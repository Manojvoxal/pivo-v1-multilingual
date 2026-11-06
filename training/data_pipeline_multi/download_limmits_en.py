"""
Download the LIMMITS'24 Indian English TTS data (IISc SYSPIN, 1 female + 1 male, ~40 h each)
onto the Modal volume `tts1-raw-datasets`, next to the SYSPIN Kannada data, and check it.

Files (from the LIMMITS'24 download portal):
  English_F.tar.gz  ~10.4 GB
  English_M.tar.gz  ~10.9 GB

Result on the volume:
  /english/limmits/archives/English_{F,M}.tar.gz   (deleted after a good extract unless --keep-archives)
  /english/limmits/English_F/...                    extracted as shipped
  /english/limmits/English_M/...
  /english/limmits/report_English_{F,M}.json        files, pairing wav<->transcript, sample rates, hours

Usage (from this folder, after you registered at https://forms.gle/ch6H2vRjFaKr84V3A):
  modal run --detach download_limmits_en.py                   # both voices, from IISc
  modal run --detach download_limmits_en.py --voices F        # one voice
  modal run download_limmits_en.py --stage inspect            # only re-run the check
  modal run --detach download_limmits_en.py --source hf       # fallback: Hugging Face mirror
                                                              # iAkashPaul/limmits-2024 (needs the
                                                              # `huggingface` secret)
  modal volume get tts1-raw-datasets english/limmits/report_English_F.json .

A broken download is resumed when you run it again (wget -c). Each archive is checked with
`gzip -t` before it is extracted.
"""

import json
import os
import subprocess
import time

import modal

RAW_VOLUME = "tts1-raw-datasets"
HF_SECRET = "huggingface"
IISC_BASE = "https://ee.iisc.ac.in/limmitsdataset/download"
HF_MIRROR = "iAkashPaul/limmits-2024"
ROOT = "/raw/english/limmits"
VOICES = ("F", "M")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "pigz", "libsndfile1")
    .pip_install("soundfile", "numpy", "huggingface_hub>=0.23")
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("limmits-en-download", image=image)
raw_vol = modal.Volume.from_name(RAW_VOLUME)
hf_secret = modal.Secret.from_name(HF_SECRET)


def _sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(volumes={"/raw": raw_vol}, secrets=[hf_secret], cpu=4, memory=8192,
              timeout=24 * 3600, retries=1)
def fetch(voice: str, source: str = "iisc", keep_archives: bool = False):
    name = f"English_{voice}"
    arch_dir = f"{ROOT}/archives"
    os.makedirs(arch_dir, exist_ok=True)
    archive = f"{arch_dir}/{name}.tar.gz"
    out_dir = f"{ROOT}/{name}"
    done_flag = f"{out_dir}/.extracted"

    if os.path.exists(done_flag):
        print(f"{name}: already extracted, skipping download", flush=True)
        return {"voice": voice, "status": "already_done"}

    t0 = time.time()
    if source == "iisc":
        _sh(["wget", "-c", "--tries=20", "--waitretry=30", "--read-timeout=120",
             "--progress=dot:giga", "-O", archive, f"{IISC_BASE}/{name}.tar.gz"])
    elif source == "hf":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_MIRROR, f"{name}.tar.gz", repo_type="dataset",
                               local_dir=arch_dir)
        if path != archive:
            os.replace(path, archive)
    else:
        raise ValueError("--source must be iisc or hf")
    raw_vol.commit()
    size_gb = os.path.getsize(archive) / 1e9
    print(f"{name}: downloaded {size_gb:.2f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)

    # integrity check before extracting (a truncated archive fails here, re-run to resume)
    _sh(["bash", "-c", f"pigz -t '{archive}'"])

    os.makedirs(out_dir, exist_ok=True)
    _sh(["bash", "-c", f"tar -I pigz -xf '{archive}' -C '{out_dir}'"])
    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if not keep_archives:
        os.remove(archive)
    raw_vol.commit()
    return {"voice": voice, "status": "extracted", "archive_gb": round(size_gb, 2),
            "minutes": round((time.time() - t0) / 60, 1)}


@app.function(volumes={"/raw": raw_vol}, cpu=8, memory=8192, timeout=2 * 3600)
def inspect(voice: str):
    """What is inside: folders, file types, wav<->transcript pairing, sample rates, hours."""
    import collections
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import soundfile as sf

    raw_vol.reload()
    name = f"English_{voice}"
    base = Path(f"{ROOT}/{name}")
    if not base.exists():
        raise FileNotFoundError(f"{base} not found: run the download first")

    exts = collections.Counter()
    dirs = collections.Counter()
    wavs, texts, other = {}, {}, []
    for p in base.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        exts[p.suffix.lower()] += 1
        dirs[str(p.parent.relative_to(base))] += 1
        if p.suffix.lower() in (".wav", ".flac"):
            wavs[p.stem] = p
        elif p.suffix.lower() in (".txt", ".lab", ".data", ".tsv", ".csv", ".json"):
            texts[p.stem] = p
        else:
            other.append(str(p.relative_to(base)))

    def info(p):
        try:
            i = sf.info(str(p))
            return i.samplerate, i.channels, i.frames / i.samplerate
        except Exception as e:  # noqa: BLE001
            return None, None, str(e)

    with ThreadPoolExecutor(32) as ex:
        infos = list(ex.map(info, wavs.values()))
    rates = collections.Counter(r for r, _, _ in infos if r)
    chans = collections.Counter(c for _, c, _ in infos if c)
    durs = [d for r, _, d in infos if r]
    bad = [str(p) for p, (r, _, _) in zip(wavs.values(), infos) if not r]

    paired = sorted(set(wavs) & set(texts))
    samples = []
    for stem in paired[:: max(1, len(paired) // 8)][:8]:
        samples.append({"id": stem, "text": texts[stem].read_text(encoding="utf-8", errors="replace").strip()[:300]})
    # transcript files that are lists (one file, many lines) instead of one file per clip
    list_files = {str(p.relative_to(base)): sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
                  for s, p in texts.items() if s not in wavs}

    report = {
        "voice": name,
        "folders": dict(dirs.most_common(20)),
        "file_types": dict(exts),
        "wav_files": len(wavs),
        "transcript_files": len(texts),
        "paired_one_file_per_clip": len(paired),
        "transcript_list_files": dict(list(list_files.items())[:20]),
        "sample_rates": dict(rates),
        "channels": dict(chans),
        "hours": round(sum(durs) / 3600, 2),
        "clip_seconds": {"min": round(min(durs), 2), "mean": round(sum(durs) / len(durs), 2),
                         "max": round(max(durs), 2)} if durs else None,
        "unreadable_wavs": bad[:20],
        "unreadable_count": len(bad),
        "other_files": other[:20],
        "sample_transcripts": samples,
    }
    with open(f"{ROOT}/report_{name}.json", "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    raw_vol.commit()
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return report


@app.local_entrypoint()
def main(stage: str = "all", voices: str = "F,M", source: str = "iisc", keep_archives: bool = False):
    vs = [v.strip().upper() for v in voices.split(",") if v.strip()]
    if any(v not in VOICES for v in vs):
        raise SystemExit("--voices takes F, M or F,M")
    if stage in ("all", "download"):
        for r in fetch.map(vs, kwargs={"source": source, "keep_archives": keep_archives}):
            print(r)
    if stage in ("all", "inspect"):
        for r in inspect.map(vs):
            print(f"{r['voice']}: {r['wav_files']} wavs, {r['hours']} h, "
                  f"sample rates {r['sample_rates']}, paired {r['paired_one_file_per_clip']}")
