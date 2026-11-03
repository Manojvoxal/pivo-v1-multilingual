"""
Build PiVo-ready training data (Kannada + English) on Modal.

Sources
  Kannada : SYSPIN (already on volume `tts1-raw-datasets` at /kannada/syspin_hf)
  English : Hi-Fi TTS clean speakers 92 (F) and 6097 (M), streamed from
            Hugging Face `MikhailT/hifi-tts` parquet shards

Steps (all run remotely; safe to re-run a single stage)
  1. prepare  : text -> PiVo tokens, 48k/44.1k -> 16 kHz mono, loudness norm,
                writes an MFA corpus + per-speaker metadata
  2. align    : Montreal Forced Aligner 3.4.3 (same version PiVo used)
                Kannada -> train an aligner on our own data with our G2P lexicon
                English -> pretrained english_us_arpa model (falls back to training)
  3. build    : per-token frame durations, 3-frame edge trim, Opus FARGAN
                20-dim features, drop bad clips, per-speaker manifests
  4. finalize : merge into manifests/{train,val,test}.jsonl, feature stats, summary

Usage (from pivo-v1/training/data_pipeline):
  modal run prepare_pivo_v4.py --smoke              # ~10 min end-to-end test, tiny data
  modal run --detach prepare_pivo_v4.py             # full run (keeps going if laptop sleeps)
  modal run prepare_pivo_v4.py --smoke --stage align   # resume from a stage (align -> build -> finalize)
  modal run prepare_pivo_v4.py --stage build --only      # run just one stage
Output volume: `pivo-data`, folder /pivo_kn_en_v1 (or /smoke).
"""

import os
from pathlib import Path

import modal

RAW_VOLUME = "tts1-raw-datasets"
OUT_VOLUME = "pivo-data"
SYSPIN_DIR = "/raw/kannada/syspin_hf"
HIFI_REPO = "MikhailT/hifi-tts"
PIVO_REPO = "empero-ai/PiVo-1.4M"
OPUS_COMMIT = "bafd40dcbd506e12da257b8cb05c0afe1ed44719"  # Opus revision PiVo used (PROVENANCE.json)
OPUS_FALLBACK_TARBALL = "http://archive.ubuntu.com/ubuntu/pool/main/o/opus/opus_1.5.2.orig.tar.gz"

# voice_id 0/1 = English F/M so they can start from PiVo's Pi(0, F) / Vo(1, M) voice embeddings.
SPEAKERS = [
    {"name": "en_hifi92_f",   "lang": "en", "source": "hifi",   "key": "92",     "hours": 10, "voice_id": 0},
    {"name": "en_hifi6097_m", "lang": "en", "source": "hifi",   "key": "6097",   "hours": 10, "voice_id": 1},
    {"name": "kn_syspin_f",   "lang": "kn", "source": "syspin", "key": "Female", "hours": 15, "voice_id": 2},
    {"name": "kn_syspin_m",   "lang": "kn", "source": "syspin", "key": "Male",   "hours": 15, "voice_id": 3},
]
SMOKE_HOURS = 0.15

HERE = Path(__file__).parent

# Dockerfile RUN lines must be single-line, so the build is one `bash -c` command:
# try the exact Opus git revision PiVo pinned, else fall back to the 1.5.2 release.
_cfg = "./configure -q --enable-deep-plc --enable-dred --disable-doc && make -j8 fargan_demo"
opus_build = (
    "bash -c '"
    "cd /opt && "
    f"if git clone -q https://github.com/xiph/opus.git opus && cd opus && git checkout -q {OPUS_COMMIT} "
    f"&& ./autogen.sh && {_cfg}; "
    f"then echo git:{OPUS_COMMIT} > /opt/opus_version.txt; "
    f"else cd /opt && rm -rf opus && wget -q {OPUS_FALLBACK_TARBALL} -O opus.tgz && tar xzf opus.tgz "
    f"&& mv opus-1.5.2 opus && cd opus && {_cfg} && echo release:1.5.2 > /opt/opus_version.txt; fi "
    "&& cp /opt/opus/fargan_demo /usr/local/bin/fargan_demo && /usr/local/bin/fargan_demo 2>&1 | head -1; "
    "test -x /usr/local/bin/fargan_demo'"
)

image = (
    modal.Image.micromamba(python_version="3.11")
    .apt_install("git", "wget", "build-essential", "autoconf", "automake", "libtool", "pkg-config")
    .micromamba_install(
        # python + libsqlite pinned in the same solve: MFA's sqlite database needs the
        # conda libsqlite (with sqlite3_deserialize), not an older system copy.
        "python=3.11", "libsqlite>=3.45", "sqlite",
        "montreal-forced-aligner=3.4.3", "pysoundfile", "pyarrow", "soxr-python",
        "pyloudnorm", "huggingface_hub", "pandas",
        channels=["conda-forge"],
    )
    .run_commands(opus_build)  # before LD_LIBRARY_PATH, so git/wget use system libs
    .env({"MFA_ROOT_DIR": "/tmp/mfa_root", "PYTHONUNBUFFERED": "1",
          "LD_LIBRARY_PATH": "/opt/conda/lib"})
    # fail the image build early (not hours later) if sqlite or MFA are broken
    .run_commands(
        "python -c \"import sqlite3; c = sqlite3.connect(':memory:'); "
        "print('sqlite', sqlite3.sqlite_version); c.serialize()\"",
        "mfa version",
    )
    .add_local_dir(HERE / "pivo_kn_v3", remote_path="/root/pivo_kn")  # code folder, imported as pivo_kn
)

app = modal.App("pivo-kn-en-data", image=image)
raw_vol = modal.Volume.from_name(RAW_VOLUME)
out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)
VOLS = {"/raw": raw_vol, "/out": out_vol}


def _root(smoke):
    return "/out/smoke" if smoke else "/out/pivo_kn_en_v1"


def _setup_frontend():
    import sys
    sys.path.insert(0, "/root")
    from huggingface_hub import hf_hub_download
    from pivo_kn import english
    english.set_cmudict_path(hf_hub_download(PIVO_REPO, "pivo/resources/cmudict.dict"))


# --------------------------------------------------------------------------- sharding
# Every stage fans out over many small CPU containers (MFA and Opus are CPU-only, so
# GPUs would not help). Counts are per speaker (prepare/build) or per language (align).
PREP_SHARDS, ALIGN_SHARDS, BUILD_SHARDS = 12, 16, 12
SMOKE_SHARDS = 2
OVERCOLLECT = 1.5            # prepare a bit extra; finalize caps each speaker at its target hours
ALIGNER_TRAIN_HOURS = 1.5    # per Kannada speaker, to train the aligner quickly


def _n(smoke, n):
    return SMOKE_SHARDS if smoke else n


def _hours(spk, smoke):
    return SMOKE_HOURS if smoke else spk["hours"]


# --------------------------------------------------------------------------- sources
def _iter_syspin(gender, shard=0, n_shards=1, seed=0):
    import glob, io, random
    import pyarrow.parquet as pq
    import soundfile as sf
    files = sorted(glob.glob(f"{SYSPIN_DIR}/**/*.parquet", recursive=True))
    random.Random(seed).shuffle(files)  # spread selection over domains/splits
    for f in files[shard::n_shards]:
        stem = Path(f).stem
        pf = pq.ParquetFile(f)
        row = 0
        for batch in pf.iter_batches(batch_size=64, columns=["path", "transcript", "domain", "gender", "language"]):
            for rec in batch.to_pylist():
                row += 1
                if rec["gender"] != gender or (rec.get("language") or "Kannada") != "Kannada":
                    continue
                audio, sr = sf.read(io.BytesIO(rec["path"]["bytes"]), dtype="float32")
                yield {"utt_id": f"syspin_{gender[0].lower()}_{stem}_{row:06d}", "audio": audio,
                       "sr": sr, "text": rec["transcript"], "extra": {"domain": rec.get("domain")}}


def _iter_hifi(speaker, shard=0, n_shards=1):
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(HIFI_REPO, repo_type="dataset")
                   if f.endswith(".parquet") and "clean" in f and "train" in f)
    if not files:
        raise RuntimeError("no train/clean parquet shards found in " + HIFI_REPO)
    for f in files[shard::n_shards]:  # round-robin, so a speaker's contiguous files spread over shards
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
        os.remove(local)  # keep the container disk small


# --------------------------------------------------------------------------- stages
@app.function(volumes=VOLS, cpu=2, memory=4096, timeout=3600)
def reset_dirs(smoke: bool, subdirs: list):
    """Remove outputs of a stage before re-running it (so stale shards never mix in)."""
    import shutil
    root = _root(smoke)
    for sub in subdirs:
        shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
    out_vol.commit()


@app.function(volumes=VOLS, cpu=2, memory=8192, timeout=3 * 3600)
def prepare_shard(spk: dict, shard: int, n_shards: int, smoke: bool = False):
    _setup_frontend()
    from pivo_kn import stages
    if spk["source"] == "syspin":
        items = _iter_syspin(spk["key"], shard, n_shards)
    else:
        items = _iter_hifi(spk["key"], shard, n_shards)
    hours = _hours(spk, smoke) * OVERCOLLECT / n_shards
    report = stages.prepare_speaker(items, _root(smoke), spk["name"], spk["lang"], hours, tag=f"{shard:03d}")
    out_vol.commit()
    print(report)
    return report


def _mfa_jobs(cpus):
    # os.cpu_count() reports the host, not the container reservation, so pass it explicitly
    return max(1, cpus - 1)


@app.function(volumes=VOLS, cpu=32, memory=65536, timeout=6 * 3600)
def align_setup(lang: str, smoke: bool = False):
    """Write the lexicon and get an acoustic model: Kannada -> train on a small subset;
    English -> PiVo's pretrained english_us_arpa 3.0.0 (checked on a few clips first)."""
    import glob, shutil, subprocess, sys
    sys.path.insert(0, "/root")
    from pivo_kn import stages
    out_vol.reload()
    root = _root(smoke)
    dict_path, n_words = stages.write_dictionary(root, lang)
    rows = [r for name in sorted(os.listdir(os.path.join(root, "meta")))
            for r in stages._jsonl_read(os.path.join(root, "meta", name)) if r["lang"] == lang]
    print(f"{lang}: {n_words} lexicon entries, {len(rows)} utterances")
    shutil.rmtree(os.path.join(root, "align", lang), ignore_errors=True)
    os.makedirs(os.path.join(root, "align"), exist_ok=True)
    model_dst = os.path.join(root, "align", f"{lang}_acoustic_model.zip")

    def _train_subset():
        work = f"/tmp/train_{lang}"
        shutil.rmtree(work, ignore_errors=True)
        hours = min(ALIGNER_TRAIN_HOURS, SMOKE_HOURS) if smoke else ALIGNER_TRAIN_HOURS
        stages.copy_corpus_subset(root, lang, stages.pick_training_subset(rows, hours), work)
        os.makedirs(os.path.join(work, "dict"))
        shutil.copy(dict_path, os.path.join(work, "dict", f"{lang}.dict"))
        stages.run_mfa(work, lang, "train", num_jobs=_mfa_jobs(32), temp_dir="/tmp/mfa_tmp_train")
        shutil.copy(os.path.join(work, "align", f"{lang}_acoustic_model.zip"), model_dst)

    used = "train_subset"
    if lang == "en":
        try:
            try:
                subprocess.run(["mfa", "model", "download", "acoustic", "english_us_arpa",
                                "--version", "3.0.0"], check=True)
                used = "pretrained:english_us_arpa(3.0.0)"
            except subprocess.CalledProcessError:
                subprocess.run(["mfa", "model", "download", "acoustic", "english_us_arpa"], check=True)
                used = "pretrained:english_us_arpa(latest)"
            found = glob.glob("/tmp/mfa_root/**/english_us_arpa*.zip", recursive=True)
            if not found:
                raise subprocess.CalledProcessError(1, "find english_us_arpa.zip")
            # quick compatibility check on 30 clips (phone set must match our lexicon)
            work = "/tmp/check_en"
            shutil.rmtree(work, ignore_errors=True)
            stages.copy_corpus_subset(root, lang, [(r["speaker"], r["id"]) for r in rows[:30]], work)
            os.makedirs(os.path.join(work, "dict"))
            shutil.copy(dict_path, os.path.join(work, "dict", f"{lang}.dict"))
            stages.run_mfa(work, lang, f"pretrained:{found[0]}", num_jobs=_mfa_jobs(32), temp_dir="/tmp/mfa_tmp_chk")
            from pivo_kn import prep
            ok = 0
            for r in rows[:30]:
                ali = os.path.join(work, "align", lang, r["speaker"], r["id"] + ".json")
                if not os.path.exists(ali):
                    continue
                labels = [l for _, _, l in prep.parse_mfa_json(ali)[1] if l not in prep.SILENCE_LABELS]
                ok += labels == [prep.mfa_phone(p) for _, ph in r["words"] for p in ph]
            print(f"pretrained English check: {ok}/30 clips reproduce our phone sequence")
            if ok < 20:
                raise subprocess.CalledProcessError(1, "pretrained model phones do not match our lexicon")
            shutil.copy(found[0], model_dst)
        except subprocess.CalledProcessError as exc:
            print("pretrained English model unusable, training on a subset instead:", exc)
            used = "train_subset"
            _train_subset()
    else:
        _train_subset()
    out_vol.commit()
    utts = [(r["speaker"], r["id"]) for r in rows]
    print(f"{lang}: aligner = {used}")
    return {"lang": lang, "aligner": used, "model": model_dst, "utts": utts}


@app.function(volumes=VOLS, cpu=8, memory=16384, timeout=3 * 3600)
def align_shard(lang: str, utts: list, model: str, smoke: bool = False):
    import glob, shutil, sys
    sys.path.insert(0, "/root")
    from pivo_kn import stages
    out_vol.reload()
    root = _root(smoke)
    work = "/tmp/work"
    shutil.rmtree(work, ignore_errors=True)
    stages.copy_corpus_subset(root, lang, utts, work)
    os.makedirs(os.path.join(work, "dict"))
    shutil.copy(os.path.join(root, "dict", f"{lang}.dict"), os.path.join(work, "dict", f"{lang}.dict"))
    local_model = f"/tmp/{lang}_acoustic_model.zip"
    shutil.copy(model, local_model)
    stages.run_mfa(work, lang, f"pretrained:{local_model}", num_jobs=_mfa_jobs(8), temp_dir="/tmp/mfa_tmp")
    n = 0
    for src in glob.glob(os.path.join(work, "align", lang, "*", "*.json")):
        spk = os.path.basename(os.path.dirname(src))
        dst = os.path.join(root, "align", lang, spk)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
        n += 1
    out_vol.commit()
    print(f"{lang}: aligned {n}/{len(utts)}")
    return n


@app.function(volumes=VOLS, cpu=2, memory=8192, timeout=3 * 3600)
def build_shard(spk: dict, shard: int, n_shards: int, smoke: bool = False):
    import sys
    sys.path.insert(0, "/root")
    from pivo_kn import stages
    out_vol.reload()
    root = _root(smoke)
    meta = stages.speaker_meta(root, spk["name"])[shard::n_shards]
    report = stages.build_speaker(root, spk["name"], spk["voice_id"], "/usr/local/bin/fargan_demo",
                                  meta=meta, tag=f"{shard:03d}")
    out_vol.commit()
    return report


@app.function(volumes=VOLS, cpu=4, memory=32768, timeout=3600)
def finalize(smoke: bool = False):
    import collections, json, sys
    sys.path.insert(0, "/root")
    from pivo_kn import stages
    out_vol.reload()
    root = _root(smoke)
    names, build_reports = [], {}
    for spk in SPEAKERS:
        if os.path.isdir(os.path.join(root, "manifests", "shards")):
            build_reports[spk["name"]] = stages.finalize_speaker(
                root, spk["name"], spk["voice_id"], max_hours=_hours(spk, smoke))
        if os.path.exists(os.path.join(root, "manifests", "per_speaker", spk["name"] + ".jsonl")):
            names.append(spk["name"])
    counts = stages.merge_manifests(root, names)
    stats = stages.feature_stats(root)
    prepare_reports = collections.defaultdict(lambda: {"kept": 0, "kept_hours": 0.0, "dropped": collections.Counter()})
    rep_dir = os.path.join(root, "reports")
    for f in sorted(os.listdir(rep_dir)):
        if f.startswith("prepare_") and f.endswith(".jsonl"):
            rep = stages._jsonl_read(os.path.join(rep_dir, f))[0]
            agg = prepare_reports[rep["speaker"]]
            agg["kept"] += rep["kept"]
            agg["kept_hours"] = round(agg["kept_hours"] + rep["kept_hours"], 3)
            agg["dropped"].update(rep["dropped"])
    summary = {"root": root, "splits": counts, "feature_frames_for_stats": stats["frames"],
               "voices": {s["name"]: s["voice_id"] for s in SPEAKERS},
               "prepare": {k: {**v, "dropped": dict(v["dropped"])} for k, v in prepare_reports.items()},
               "build": build_reports,
               "opus": open("/opt/opus_version.txt").read().strip()}
    with open(os.path.join(root, "manifests", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    out_vol.commit()
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


@app.function(timeout=24 * 3600)
def pipeline(stage: str = "all", smoke: bool = False, only: bool = False):
    """Remote orchestrator, so `modal run --detach` keeps going without the laptop.

    `stage` is where to START (prepare -> align -> build -> finalize); later stages
    follow automatically unless `only` is set.
    """
    import time
    order = ["prepare", "align", "build", "finalize"]
    start = 0 if stage == "all" else order.index(stage)
    todo = [order[start]] if only else order[start:]
    result, t0 = None, time.time()
    if "prepare" in todo:
        reset_dirs.remote(smoke, ["meta", "corpus", "reports", "dict"])
        n = _n(smoke, PREP_SHARDS)
        list(prepare_shard.starmap([(s, i, n) for s in SPEAKERS for i in range(n)], kwargs={"smoke": smoke}))
        print(f"prepare done in {time.time() - t0:.0f}s", flush=True)
    if "align" in todo:
        langs = sorted({s["lang"] for s in SPEAKERS})
        setups = list(align_setup.map(langs, kwargs={"smoke": smoke}))
        jobs = []
        for st in setups:
            n = _n(smoke, ALIGN_SHARDS)
            jobs += [(st["lang"], st["utts"][i::n], st["model"]) for i in range(n) if st["utts"][i::n]]
            print(f"{st['lang']}: aligner {st['aligner']}, {len(st['utts'])} utterances", flush=True)
        list(align_shard.starmap(jobs, kwargs={"smoke": smoke}))
        print(f"align done at {time.time() - t0:.0f}s", flush=True)
    if "build" in todo:
        reset_dirs.remote(smoke, ["data", "manifests", "reports/build_shards"])
        n = _n(smoke, BUILD_SHARDS)
        list(build_shard.starmap([(s, i, n) for s in SPEAKERS for i in range(n)], kwargs={"smoke": smoke}))
        print(f"build done at {time.time() - t0:.0f}s", flush=True)
    if "finalize" in todo:
        result = finalize.remote(smoke=smoke)
        print(f"all done at {time.time() - t0:.0f}s", flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, only: bool = False):
    result = pipeline.remote(stage=stage, smoke=smoke, only=only)
    if result:
        print("splits:", result["splits"])
