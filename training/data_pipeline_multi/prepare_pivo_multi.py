"""
Build PiVo-ready training data for the multilingual model (11 major Indian languages + English
by default; optional tier 3 adds 5 more, tier 4 is dropped for now, see pivo_multi/langs.py)
on Modal. Successor of pivo-v1/training/data_pipeline/prepare_pivo_v4.py (Kannada + English).

Sources (see pivo_multi/config.py)
  Indian languages : Rasa (AI4Bharat, Hugging Face `ai4bharat/Rasa`, gated), 1 F + 1 M each
  Kannada          : SYSPIN (already on volume `tts1-raw-datasets` at /kannada/syspin_hf)
  English          : IndicTTS English (IIT Madras), uploaded to /english/indictts/...
                     (--english hifi = Hi-Fi TTS speakers 92/6097, used by the smoke test)

Stages (each can be re-run alone; later stages follow unless --only)
  scan     : Rasa metadata only (no audio download): style/gender/hours report per language
             and the exact clip selection; checks the IndicTTS English upload
  prepare  : download the selected clips, text -> tokens, 16 kHz mono, -20 LUFS, MFA corpus
  align    : MFA 3.4.3, one aligner trained per language (English: english_us_arpa if it fits)
  build    : per-token durations, edge trim, Opus FARGAN 20-dim features, per-speaker manifests
  finalize : manifests/{train,val,test}.jsonl, symbols/languages/voices.json, feature stats
  cleanup  : (optional, run by hand) delete corpus/ after a good build to save ~90 GB

Usage (from this folder):
  modal run prepare_pivo_multi.py --smoke                       # ~20 min end-to-end test (en, kn, hi, ta)
  modal run prepare_pivo_multi.py --stage scan --only           # look at Rasa styles first (minutes)
  modal run prepare_pivo_multi.py --stage scan --only --styles "BOOK,NEWS,WIKI"   # pin styles
  modal run --detach prepare_pivo_multi.py --styles "BOOK,NEWS,WIKI"              # full run
  modal run --detach prepare_pivo_multi.py --langs kn,en,hi,te,ta,bn            # tier-1 languages only
  modal run --detach prepare_pivo_multi.py --phases 3 --styles "..."           # later: add tier 3 (ne kok mai doi sa)
  modal run prepare_pivo_multi.py --stage build                 # resume from a stage
Output volume: `pivo-multi-data` (v2), folder /pivo_multi_v1 (smoke: /smoke).
"""

import os
from pathlib import Path

import modal

RAW_VOLUME = "tts1-raw-datasets"
OUT_VOLUME = "pivo-multi-data"     # Volumes v2: this dataset has ~1.5M small files
HF_SECRET = "huggingface"          # Modal secret holding HF_TOKEN (Rasa is gated)
PIVO_REPO = "empero-ai/PiVo-1.4M"
OPUS_COMMIT = "bafd40dcbd506e12da257b8cb05c0afe1ed44719"  # Opus revision PiVo used
OPUS_FALLBACK_TARBALL = "http://archive.ubuntu.com/ubuntu/pool/main/o/opus/opus_1.5.2.orig.tar.gz"

HERE = Path(__file__).parent

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
        "python=3.11", "libsqlite>=3.45", "sqlite",
        "montreal-forced-aligner=3.4.3", "pysoundfile", "pyarrow", "soxr-python",
        "pyloudnorm", "huggingface_hub>=0.23", "fsspec", "pandas",
        channels=["conda-forge"],
    )
    .run_commands(opus_build)
    .env({"MFA_ROOT_DIR": "/tmp/mfa_root", "PYTHONUNBUFFERED": "1",
          "LD_LIBRARY_PATH": "/opt/conda/lib"})
    .run_commands(
        "python -c \"import sqlite3; c = sqlite3.connect(':memory:'); "
        "print('sqlite', sqlite3.sqlite_version); c.serialize()\"",
        "mfa version",
    )
    .add_local_dir(HERE / "pivo_multi", remote_path="/root/pivo_multi")
)

app = modal.App("pivo-multi-data", image=image)
raw_vol = modal.Volume.from_name(RAW_VOLUME)
try:
    out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True, version=2)
except TypeError:  # older Modal client: create it yourself with `modal volume create --version=2`
    out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)
VOLS = {"/raw": raw_vol, "/out": out_vol}
hf_secret = modal.Secret.from_name(HF_SECRET)

# Fan-out sizes. Every step here (download, resampling, MFA, Opus feature extraction) is
# CPU-only, so a GPU would sit idle; speed comes from many CPU containers in parallel.
# Tuned for wall-clock time over cost (3 Nov 2026).
PREP_SHARDS, ALIGN_SHARDS, BUILD_SHARDS = 24, 32, 16   # per speaker / language / speaker
SMOKE_SHARDS = 2
SMOKE_LANGS = ["en", "kn", "hi", "ta"]
SMOKE_HOURS = 0.1
SMOKE_RASA_SHARDS = 6          # smoke: scan 6 parquet files spread evenly (Rasa files are sorted by speaker/style)
ALIGNER_TRAIN_HOURS = 1.5      # per speaker, to train each language's aligner


def _root(smoke):
    return "/out/smoke" if smoke else "/out/pivo_multi_v1"


def _n(smoke, n):
    return SMOKE_SHARDS if smoke else n


def _setup():
    import sys
    sys.path.insert(0, "/root")
    from huggingface_hub import hf_hub_download
    from pivo_multi import english
    english.set_cmudict_path(hf_hub_download(PIVO_REPO, "pivo/resources/cmudict.dict"))


def _speakers(phases, english, langs, smoke):
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import config
    if smoke:
        langs = langs or SMOKE_LANGS
    spks = config.active_speakers(phases=tuple(phases), english=english, langs=langs)
    if smoke:
        for s in spks:
            s["hours"] = SMOKE_HOURS
    return spks


# --------------------------------------------------------------------------- helpers
@app.function(volumes=VOLS, cpu=2, memory=4096, timeout=3600)
def reset_paths(paths: list):
    """Remove outputs of a stage before re-running it (so stale shards never mix in)."""
    import glob
    import shutil
    out_vol.reload()
    n = 0
    for pattern in paths:
        for p in glob.glob(pattern):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            n += 1
    out_vol.commit()
    return n


# --------------------------------------------------------------------------- stage 0: scan
@app.function(volumes=VOLS, secrets=[hf_secret], cpu=4, memory=16384, timeout=3 * 3600)
def scan_language(lang: str, config_name: str, hours: float, styles: list = None, smoke: bool = False):
    """Rasa metadata for one language (no audio) -> style report + clip selection."""
    import collections
    from concurrent.futures import ThreadPoolExecutor
    _setup()
    from pivo_multi import config, sources, stages
    token = os.environ.get("HF_TOKEN")
    shards = sources.rasa_list_shards(config.RASA_REPO, config_name, config.RASA_SPLIT, token)
    if not shards:
        raise RuntimeError(f"{lang}: no {config.RASA_SPLIT} parquet files under {config_name}/ "
                           f"(accepted the Rasa terms on Hugging Face? HF_TOKEN in the '{HF_SECRET}' secret?)")
    if smoke:
        k = min(SMOKE_RASA_SHARDS, len(shards))
        shards = [shards[round(i * (len(shards) - 1) / max(1, k - 1))] for i in range(k)]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda p: sources.rasa_read_meta(config.RASA_REPO, p, token), shards))
    rows = [r for part, _ in results for r in part]
    columns = results[0][1]
    selected, report = stages.select_rasa(rows, lang, hours, styles=styles)
    report.update(shards=len(shards), columns=columns)
    root = _root(smoke)
    out_vol.reload()
    stages._jsonl_write(os.path.join(root, "select", f"{lang}.jsonl"), selected)
    stages._json_write(os.path.join(root, "reports", "scan", f"{lang}.json"), report)
    out_vol.commit()
    by_g = collections.Counter(r["gender"] for r in selected)
    print(f"{lang}: {len(rows)} rows in {len(shards)} shards -> selected {dict(by_g)}", flush=True)
    return report


@app.function(volumes=VOLS, cpu=2, memory=4096, timeout=1800)
def check_indictts(dirs: list):
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import sources
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            out[d] = {"error": "folder not found on the raw volume"}
            continue
        _, stats = sources.indictts_index(d)
        out[d] = stats
    return out


# --------------------------------------------------------------------------- stage 1: prepare
@app.function(volumes=VOLS, secrets=[hf_secret], cpu=2, memory=8192, timeout=2 * 3600,
              max_containers=100, retries=2)
def prepare_rasa_shard(lang: str, shard: str, wanted: dict, speakers: dict, smoke: bool = False):
    """wanted: {row_index(str): gender}; speakers: {gender: speaker_name}."""
    import shutil
    _setup()
    from pivo_multi import config, sources, stages
    token = os.environ.get("HF_TOKEN")
    wanted = {int(k): v for k, v in wanted.items()}
    local = sources.rasa_download(config.RASA_REPO, shard, token, local_dir=f"/tmp/rasa_{lang}")
    tag = f"{lang}_{Path(shard).stem}"
    reports = []
    try:
        for g, name in sorted(speakers.items()):
            want_g = {k: v for k, v in wanted.items() if v == g}
            if not want_g:
                continue
            items = (it for gg, it in sources.rasa_items(local, want_g, lang) if gg == g)
            reports.append(stages.prepare_speaker(items, _root(smoke), name, lang, 1e9, tag=tag))
    finally:
        shutil.rmtree(f"/tmp/rasa_{lang}", ignore_errors=True)
    out_vol.commit()
    return reports


@app.function(volumes=VOLS, secrets=[hf_secret], cpu=2, memory=8192, timeout=3 * 3600)
def prepare_shard(spk: dict, shard: int, n_shards: int, smoke: bool = False):
    """SYSPIN Kannada / IndicTTS English / Hi-Fi English (one shard of one speaker)."""
    _setup()
    from pivo_multi import config, sources, stages
    if spk["source"] == "syspin":
        items = sources.iter_syspin(config.SYSPIN_DIR, spk["key"], shard, n_shards)
    elif spk["source"] == "indictts":
        items = sources.iter_indictts(spk["dir"], spk["name"].split("_")[2], spk["gender"], shard, n_shards)
    elif spk["source"] == "hifi":
        items = sources.iter_hifi(spk["key"], shard, n_shards)
    else:
        raise ValueError(spk["source"])
    hours = spk["hours"] * config.OVERCOLLECT / n_shards
    report = stages.prepare_speaker(items, _root(smoke), spk["name"], spk["lang"], hours, tag=f"{shard:03d}")
    out_vol.commit()
    print(report, flush=True)
    return report


# --------------------------------------------------------------------------- stage 2: align
def _mfa_jobs(cpus):
    return max(1, cpus - 1)


@app.function(volumes=VOLS, cpu=64, memory=131072, timeout=8 * 3600)
def align_setup(lang: str, speakers: list, smoke: bool = False):
    """Write the lexicon and get an acoustic model: train one on a subset of this
    language; English first tries PiVo's pretrained english_us_arpa 3.0.0."""
    import glob
    import shutil
    import subprocess
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import prep, stages
    out_vol.reload()
    root = _root(smoke)
    rows = stages.lang_meta(root, lang, speakers)
    if not rows:
        raise RuntimeError(f"{lang}: no prepared utterances (prepare stage dropped everything?)")
    dict_path, n_words = stages.write_dictionary(root, lang, rows)
    print(f"{lang}: {n_words} lexicon entries, {len(rows)} utterances", flush=True)
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
        stages.run_mfa(work, lang, "train", num_jobs=_mfa_jobs(64), temp_dir="/tmp/mfa_tmp_train")
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
            work = "/tmp/check_en"
            shutil.rmtree(work, ignore_errors=True)
            stages.copy_corpus_subset(root, lang, [(r["speaker"], r["id"]) for r in rows[:30]], work)
            os.makedirs(os.path.join(work, "dict"))
            shutil.copy(dict_path, os.path.join(work, "dict", f"{lang}.dict"))
            stages.run_mfa(work, lang, f"pretrained:{found[0]}", num_jobs=_mfa_jobs(64), temp_dir="/tmp/mfa_tmp_chk")
            ok = 0
            for r in rows[:30]:
                ali = os.path.join(work, "align", lang, r["speaker"], r["id"] + ".json")
                if not os.path.exists(ali):
                    continue
                labels = [l for _, _, l in prep.parse_mfa_json(ali)[1] if l not in prep.SILENCE_LABELS]
                ok += labels == [prep.mfa_phone(p) for _, ph in r["words"] for p in ph]
            print(f"pretrained English check: {ok}/30 clips reproduce our phone sequence", flush=True)
            if ok < 20:
                raise subprocess.CalledProcessError(1, "pretrained model phones do not match our lexicon")
            shutil.copy(found[0], model_dst)
        except subprocess.CalledProcessError as exc:
            print("pretrained English model unusable, training on a subset instead:", exc, flush=True)
            used = "train_subset"
            _train_subset()
    else:
        _train_subset()
    out_vol.commit()
    utts = [(r["speaker"], r["id"]) for r in rows]
    print(f"{lang}: aligner = {used}", flush=True)
    return {"lang": lang, "aligner": used, "model": model_dst, "utts": utts}


@app.function(volumes=VOLS, cpu=16, memory=32768, timeout=4 * 3600, retries=1)
def align_shard(lang: str, utts: list, model: str, smoke: bool = False):
    import glob
    import shutil
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import stages
    out_vol.reload()
    root = _root(smoke)
    work = "/tmp/work"
    shutil.rmtree(work, ignore_errors=True)
    stages.copy_corpus_subset(root, lang, utts, work)
    os.makedirs(os.path.join(work, "dict"))
    shutil.copy(os.path.join(root, "dict", f"{lang}.dict"), os.path.join(work, "dict", f"{lang}.dict"))
    local_model = f"/tmp/{lang}_acoustic_model.zip"
    shutil.copy(model, local_model)
    stages.run_mfa(work, lang, f"pretrained:{local_model}", num_jobs=_mfa_jobs(16), temp_dir="/tmp/mfa_tmp")
    n = 0
    for src in glob.glob(os.path.join(work, "align", lang, "*", "*.json")):
        spk = os.path.basename(os.path.dirname(src))
        dst = os.path.join(root, "align", lang, spk)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
        n += 1
    out_vol.commit()
    print(f"{lang}: aligned {n}/{len(utts)}", flush=True)
    return n


# --------------------------------------------------------------------------- stage 3: build
@app.function(volumes=VOLS, cpu=2, memory=8192, timeout=4 * 3600, retries=1)
def build_shard(spk: dict, shard: int, n_shards: int, smoke: bool = False):
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import stages
    out_vol.reload()
    root = _root(smoke)
    meta = stages.speaker_meta(root, spk["name"])[shard::n_shards]
    report = stages.build_speaker(root, spk["name"], spk["voice_id"], "/usr/local/bin/fargan_demo",
                                  meta=meta, tag=f"{shard:03d}")
    out_vol.commit()
    return report


# --------------------------------------------------------------------------- stage 4: finalize
@app.function(volumes=VOLS, cpu=4, memory=32768, timeout=2 * 3600)
def finalize(speakers: list, english: str = "indictts", smoke: bool = False):
    import collections
    import json
    import sys
    sys.path.insert(0, "/root")
    from pivo_multi import config, stages
    out_vol.reload()
    root = _root(smoke)
    build_reports = {}
    for spk in speakers:
        build_reports[spk["name"]] = stages.finalize_speaker(
            root, spk["name"], spk["voice_id"], max_hours=spk["hours"])
    all_spk = config.all_speakers(english)
    prepared = [s["name"] for s in all_spk
                if os.path.exists(os.path.join(root, "manifests", "per_speaker", s["name"] + ".jsonl"))]
    counts = stages.merge_manifests(root, prepared, all_spk)
    stats = stages.feature_stats(root)
    prep_agg = collections.defaultdict(lambda: {"kept": 0, "kept_hours": 0.0, "dropped": collections.Counter()})
    rep_dir = os.path.join(root, "reports", "prepare")
    for f in sorted(os.listdir(rep_dir)) if os.path.isdir(rep_dir) else []:
        rep = stages._jsonl_read(os.path.join(rep_dir, f))[0]
        agg = prep_agg[rep["speaker"]]
        agg["kept"] += rep["kept"]
        agg["kept_hours"] = round(agg["kept_hours"] + rep["kept_hours"], 3)
        agg["dropped"].update(rep["dropped"])
    hours_by_lang = collections.Counter()
    for name in prepared:
        rep = stages._jsonl_read(os.path.join(root, "reports", f"build_{name}.jsonl"))[0]
        hours_by_lang[name.split("_")[0]] += rep["hours"]
    summary = {"root": root, "splits": counts, "voices_prepared": len(prepared),
               "hours_total": round(sum(hours_by_lang.values()), 2),
               "hours_by_lang": {k: round(v, 2) for k, v in sorted(hours_by_lang.items())},
               "feature_stats": {k: stats[k] for k in ("frames", "files", "voices")},
               "prepare": {k: {**v, "dropped": dict(v["dropped"])} for k, v in prep_agg.items()},
               "build": build_reports,
               "opus": open("/opt/opus_version.txt").read().strip()}
    stages._json_write(os.path.join(root, "manifests", "summary.json"), summary)
    out_vol.commit()
    print(json.dumps({k: summary[k] for k in ("splits", "voices_prepared", "hours_total", "hours_by_lang")},
                     indent=1), flush=True)
    return summary


@app.function(volumes=VOLS, cpu=4, memory=8192, timeout=6 * 3600)
def cleanup(smoke: bool = False):
    """Delete corpus/ (16 kHz copies used only by MFA) once build + finalize are good."""
    import shutil
    root = _root(smoke)
    shutil.rmtree(os.path.join(root, "corpus"), ignore_errors=True)
    out_vol.commit()
    return "corpus/ removed"


# --------------------------------------------------------------------------- orchestrator
ORDER = ["scan", "prepare", "align", "build", "finalize"]


@app.function(volumes=VOLS, timeout=24 * 3600)
def pipeline(stage: str = "all", smoke: bool = False, only: bool = False, phases: list = (1, 2),
             langs: list = None, english: str = "indictts", styles: list = None):
    """Remote orchestrator, so `modal run --detach` keeps going without the laptop."""
    import collections
    import sys
    import time
    sys.path.insert(0, "/root")
    from pivo_multi import stages
    if stage == "cleanup":
        return cleanup.remote(smoke)
    root = _root(smoke)
    spks = _speakers(phases, english, langs, smoke)
    if not spks:
        raise RuntimeError("no speakers selected (check --phases / --langs)")
    rasa = [s for s in spks if s["source"] == "rasa"]
    other = [s for s in spks if s["source"] != "rasa"]
    rasa_langs = sorted({s["lang"] for s in rasa})
    print(f"{len(spks)} voices, languages: {sorted({s['lang'] for s in spks})}", flush=True)
    start = 0 if stage == "all" else ORDER.index(stage)
    todo = [ORDER[start]] if only else ORDER[start:]
    t0, result = time.time(), {}

    if "scan" in todo:
        if rasa_langs:
            args = [(l, next(s["config"] for s in rasa if s["lang"] == l),
                     next(s["hours"] for s in rasa if s["lang"] == l)) for l in rasa_langs]
            reports = list(scan_language.starmap(args, kwargs={"styles": styles, "smoke": smoke},
                                                 return_exceptions=True))
            result["scan"] = {}
            for (l, _, _), rep in zip(args, reports):
                if isinstance(rep, Exception):
                    result["scan"][l] = {"error": f"{type(rep).__name__}: {rep}"}
                else:
                    result["scan"][l] = rep
        en = [s for s in other if s["source"] == "indictts"]
        if en:
            result["indictts"] = check_indictts.remote([s["dir"] for s in en])
            bad = {d: st for d, st in result["indictts"].items() if st.get("error") or not st.get("paired")}
            if bad and not only:
                raise RuntimeError(f"IndicTTS English not ready: {bad}. Upload it (README step 3) "
                                   "or run with --english hifi")
        _print_scan(result.get("scan", {}))
        if "indictts" in result:
            print("\nIndicTTS English upload:", result["indictts"], flush=True)
        errors = {l: r["error"] for l, r in result.get("scan", {}).items() if "error" in r}
        need = 0.05 if smoke else 3.0
        low = {l: r["selected"] for l, r in result.get("scan", {}).items()
               if "error" not in r and min(r["selected"].get(g, {"hours": 0})["hours"] for g in ("f", "m")) < need}
        print(f"scan done in {time.time() - t0:.0f}s; errors: {errors or 'none'}; "
              f"languages with < 3 h selected per voice: {low or 'none'}", flush=True)
        if (errors or low) and not only:
            raise RuntimeError("scan found problems (see above); fix --styles or access, "
                               "then rerun from --stage scan")

    if "prepare" in todo:
        pats = []
        for s in spks:
            pats += [f"{root}/meta/{s['name']}__*", f"{root}/corpus/{s['lang']}/{s['name']}",
                     f"{root}/reports/prepare/{s['name']}__*"]
        reset_paths.remote(pats)
        out_vol.reload()
        jobs = []
        for l in rasa_langs:
            sel_path = os.path.join(root, "select", f"{l}.jsonl")
            if not os.path.exists(sel_path):
                raise RuntimeError(f"{l}: no selection file {sel_path}; run --stage scan first")
            by_shard = collections.defaultdict(dict)
            for r in stages._jsonl_read(sel_path):
                by_shard[r["shard"]][str(r["row"])] = r["gender"]
            names = {s["gender"]: s["name"] for s in rasa if s["lang"] == l}
            jobs += [(l, shard, wanted, names) for shard, wanted in sorted(by_shard.items())]
        n = _n(smoke, PREP_SHARDS)
        other_jobs = [(s, i, n) for s in other for i in range(n)]
        print(f"prepare: {len(jobs)} Rasa shard jobs + {len(other_jobs)} SYSPIN/English jobs", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:   # both fan-outs at the same time
            f1 = ex.submit(lambda: list(prepare_shard.starmap(
                other_jobs, kwargs={"smoke": smoke}, return_exceptions=True)))
            f2 = ex.submit(lambda: list(prepare_rasa_shard.starmap(
                jobs, kwargs={"smoke": smoke}, return_exceptions=True)))
            res = f1.result() + f2.result()
        failed = [r for r in res if isinstance(r, Exception)]
        print(f"prepare done at {time.time() - t0:.0f}s; failed jobs: {len(failed)}", flush=True)
        for r in failed[:5]:
            print("  ", type(r).__name__, str(r)[:300], flush=True)
        if len(failed) > max(2, 0.05 * len(res)):
            raise RuntimeError(f"{len(failed)} prepare jobs failed; rerun --stage prepare")

    if "align" in todo:
        out_vol.reload()
        langs_ = sorted({s["lang"] for s in spks})
        names_by_lang = {l: [s["name"] for s in spks if s["lang"] == l] for l in langs_}
        setups = list(align_setup.starmap([(l, names_by_lang[l]) for l in langs_],
                                          kwargs={"smoke": smoke}, return_exceptions=True))
        jobs = []
        for l, st in zip(langs_, setups):
            if isinstance(st, Exception):
                print(f"{l}: aligner setup FAILED: {st}", flush=True)
                continue
            n = _n(smoke, ALIGN_SHARDS)
            jobs += [(st["lang"], st["utts"][i::n], st["model"]) for i in range(n) if st["utts"][i::n]]
            print(f"{st['lang']}: aligner {st['aligner']}, {len(st['utts'])} utterances", flush=True)
        list(align_shard.starmap(jobs, kwargs={"smoke": smoke}, return_exceptions=True))
        print(f"align done at {time.time() - t0:.0f}s", flush=True)

    if "build" in todo:
        pats = []
        for s in spks:
            pats += [f"{root}/data/{s['name']}", f"{root}/manifests/shards/{s['name']}__*",
                     f"{root}/reports/build_shards/{s['name']}__*"]
        reset_paths.remote(pats)
        n = _n(smoke, BUILD_SHARDS)
        list(build_shard.starmap([(s, i, n) for s in spks for i in range(n)], kwargs={"smoke": smoke}))
        print(f"build done at {time.time() - t0:.0f}s", flush=True)

    if "finalize" in todo:
        result["finalize"] = finalize.remote(spks, english=english, smoke=smoke)
        print(f"all done at {time.time() - t0:.0f}s", flush=True)
    return result


def _print_scan(scan):
    """Human-readable scan report (styles x gender hours, selection) in the run log."""
    for lang, rep in sorted(scan.items()):
        if "error" in rep:
            print(f"\n[{lang}] ERROR {rep['error']}", flush=True)
            continue
        print(f"\n[{lang}] {rep['rows']} rows, {rep['shards']} shards, scripts {rep['script_of_first_words']}")
        print(f"  gender values: {rep['gender_values']}")
        for k, v in rep["hours_by_gender_style"].items():
            print(f"  {k:40s} {v['clips']:6d} clips {v['hours']:7.2f} h")
        print(f"  dropped: {rep['dropped']}")
        print(f"  SELECTED: {rep['selected']}   (style rule: {rep['style_rule']})")


@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, only: bool = False, phases: str = "1,2",
         langs: str = "", english: str = "", styles: str = ""):
    english = english or ("hifi" if smoke else "indictts")
    result = pipeline.remote(
        stage=stage, smoke=smoke, only=only,
        phases=[int(p) for p in phases.split(",") if p.strip()],
        langs=[l.strip() for l in langs.split(",") if l.strip()] or None,
        english=english,
        styles=[s.strip() for s in styles.split(",") if s.strip()] or None,
    )
    if not isinstance(result, dict):
        print(result)
        return
    if result.get("finalize"):
        f = result["finalize"]
        print("\nsplits:", f["splits"], "| voices:", f["voices_prepared"], "| hours:", f["hours_total"])
        print("hours by language:", f["hours_by_lang"])
