"""Readers for the raw datasets. Each yields items {utt_id, audio, sr, text, extra}.

Rasa (AI4Bharat, gated on Hugging Face; needs HF_TOKEN):
    rasa_list_shards -> rasa_read_meta (text/gender/style/duration only, no audio)
    -> selection (stages.select_rasa) -> rasa_download + rasa_items for chosen rows
SYSPIN Kannada: parquet already on the raw volume (same reader as PiVo-KN)
IndicTTS English: unzipped IIT Madras folders uploaded to the raw volume
Hi-Fi TTS English: fallback / smoke test (streamed from Hugging Face)
"""

import glob
import hashlib
import io
import os
import random
import re
import time
from pathlib import Path


def gender_of(value):
    v = str(value or "").strip().lower()
    if v.startswith("f") or v in ("woman", "w"):
        return "f"
    if v.startswith("m") or v == "man":
        return "m"
    return None


def parse_duration(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def safe_id(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:120]


def stable_order(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _retry(fn, what, tries=6):
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:  # network: 429 / 5xx / resets
            if k == tries - 1:
                raise
            wait = min(120, 5 * 2 ** k)
            print(f"[retry] {what}: {type(exc).__name__}: {exc} -> retry in {wait}s", flush=True)
            time.sleep(wait)


# ------------------------------------------------------------------ Rasa
def rasa_list_shards(repo, config, split, token=None):
    from huggingface_hub import HfApi
    files = _retry(lambda: HfApi(token=token).list_repo_files(repo, repo_type="dataset"),
                   f"list {repo}")
    prefix = f"{config}/{split}-"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


META_COLUMNS = ["filename", "text", "language", "gender", "style", "duration", "wav_path"]


def rasa_read_meta(repo, path, token=None):
    """Metadata of one Rasa parquet shard WITHOUT downloading the audio column
    (pyarrow reads only the needed column chunks over HTTP range requests)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    def _read():
        fs = HfFileSystem(token=token)
        with fs.open(f"datasets/{repo}/{path}", "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in META_COLUMNS if c in pf.schema_arrow.names]
            return pf.read(columns=cols).to_pylist(), pf.schema_arrow.names

    rows, names = _retry(_read, f"meta {path}")
    stem = Path(path).stem
    out = []
    for i, r in enumerate(rows):
        fname = r.get("filename") or r.get("wav_path") or f"{stem}_{i:06d}"
        out.append({
            "shard": path, "row": i, "filename": str(fname),
            "text": r.get("text") or "", "gender": gender_of(r.get("gender")),
            "gender_raw": r.get("gender"), "style": (r.get("style") or "").strip(),
            "duration": parse_duration(r.get("duration")),
        })
    return out, names


def rasa_download(repo, path, token=None, local_dir="/tmp/rasa"):
    from huggingface_hub import hf_hub_download
    return _retry(lambda: hf_hub_download(repo, path, repo_type="dataset", token=token,
                                          local_dir=local_dir), f"download {path}")


def rasa_items(local_parquet, wanted, lang):
    """Yield (gender, item) for rows whose row index is in `wanted` {row: gender}."""
    import pyarrow.parquet as pq
    import soundfile as sf
    pf = pq.ParquetFile(local_parquet)
    names = pf.schema_arrow.names
    cols = [c for c in ("filename", "wav_path", "text", "style", "audio") if c in names]
    row = 0
    stem = Path(local_parquet).stem
    for batch in pf.iter_batches(batch_size=32, columns=cols):
        for rec in batch.to_pylist():
            idx = row
            row += 1
            g = wanted.get(idx)
            if g is None:
                continue
            a = rec.get("audio") or {}
            data = a.get("bytes") if isinstance(a, dict) else None
            if not data:
                path = a.get("path") if isinstance(a, dict) else None
                if not path or not os.path.exists(path):
                    continue
                data = open(path, "rb").read()
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            fname = rec.get("filename") or rec.get("wav_path") or f"{stem}_{idx:06d}"
            uid = f"rasa_{lang}_{g}_{safe_id(Path(str(fname)).stem)}"
            yield g, {"utt_id": uid, "audio": audio, "sr": sr, "text": rec.get("text") or "",
                      "extra": {"style": rec.get("style"), "shard": stem, "row": idx}}


# ------------------------------------------------------------------ SYSPIN (Kannada)
def iter_syspin(syspin_dir, gender, shard=0, n_shards=1, seed=0, language="Kannada"):
    import pyarrow.parquet as pq
    import soundfile as sf
    files = sorted(glob.glob(f"{syspin_dir}/**/*.parquet", recursive=True))
    random.Random(seed).shuffle(files)  # spread selection over domains/splits
    for f in files[shard::n_shards]:
        stem = Path(f).stem
        pf = pq.ParquetFile(f)
        row = 0
        for batch in pf.iter_batches(batch_size=64, columns=["path", "transcript", "domain", "gender", "language"]):
            for rec in batch.to_pylist():
                row += 1
                if rec["gender"] != gender or (rec.get("language") or language) != language:
                    continue
                audio, sr = sf.read(io.BytesIO(rec["path"]["bytes"]), dtype="float32")
                yield {"utt_id": f"syspin_{gender[0].lower()}_{stem}_{row:06d}", "audio": audio,
                       "sr": sr, "text": rec["transcript"], "extra": {"domain": rec.get("domain")}}


# ------------------------------------------------------------------ IndicTTS English
_FESTIVAL = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')


def indictts_index(folder):
    """Pair every wav under `folder` with its transcript.

    Understands the festival `txt.done.data` format used by IndicTTS
    (`( utt_id "text" )`) and simple `id<TAB>text` / `id|text` lists."""
    wavs = {}
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if p.lower().endswith((".wav", ".flac")):
            wavs.setdefault(Path(p).stem.lower(), p)
    texts = {}
    cand = [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
            if os.path.isfile(p) and (p.endswith(".data") or p.lower().endswith((".txt", ".tsv", ".csv")))]
    for p in sorted(cand):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip().lstrip("﻿")
                if not line:
                    continue
                m = _FESTIVAL.match(line)
                if m:
                    texts[m.group(1).lower()] = m.group(2)
                    continue
                for sep in ("\t", "|"):
                    if sep in line:
                        k, v = line.split(sep, 1)
                        k = Path(k.strip()).stem.lower()
                        if k in wavs:
                            texts[k] = v.strip().strip('"')
                        break
    pairs = sorted((k, wavs[k], t) for k, t in texts.items() if k in wavs)
    stats = {"wav_files": len(wavs), "transcripts": len(texts), "paired": len(pairs),
             "transcript_files": [os.path.relpath(p, folder) for p in cand][:20]}
    return pairs, stats


def iter_indictts(folder, tag, gender, shard=0, n_shards=1):
    import soundfile as sf
    pairs, _ = indictts_index(folder)
    pairs.sort(key=lambda x: stable_order(x[0]))
    for key, wav, text in pairs[shard::n_shards]:
        audio, sr = sf.read(wav, dtype="float32")
        yield {"utt_id": f"iitm_{tag}_{gender}_{safe_id(key)}", "audio": audio, "sr": sr,
               "text": text, "extra": {"file": os.path.basename(wav)}}


# ------------------------------------------------------------------ Hi-Fi TTS (fallback)
HIFI_REPO = "MikhailT/hifi-tts"


def iter_hifi(speaker, shard=0, n_shards=1):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(HIFI_REPO, repo_type="dataset")
                   if f.endswith(".parquet") and "clean" in f and "train" in f)
    if not files:
        raise RuntimeError("no train/clean parquet shards found in " + HIFI_REPO)
    for f in files[shard::n_shards]:
        local = hf_hub_download(HIFI_REPO, f, repo_type="dataset", local_dir=f"/tmp/hifi{shard}")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=64, columns=["speaker", "file", "text_normalized", "audio"]):
            for rec in batch.to_pylist():
                if str(rec["speaker"]) != speaker:
                    continue
                audio, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
                uid = Path(rec["file"]).stem
                yield {"utt_id": f"hifi{speaker}_{uid}", "audio": audio, "sr": sr,
                       "text": rec["text_normalized"]}
        os.remove(local)
