"""
Train the multilingual PiVo acoustic model on Modal (from scratch, 12 languages, 24 voices).

Reads the dataset made by data_pipeline_multi/prepare_pivo_multi.py from the
`pivo-multi-data` volume (/pivo_multi_v1, smoke: /smoke), and writes checkpoints, logs and
sample WAVs to the `pivo-multi-runs` volume. PiVo-1.4M is still baked into the image, but
only for its FARGAN vocoder (sample WAVs) and the model-code parity check; no PiVo weights
go into the new model.

Time matters more than cost: H100 first, and several widths train side by side (one GPU
each). A small CPU orchestrator builds the feature cache once, then starts the trainers and
waits for them, so `--detach` keeps every trainer alive.

Usage (from pivo-v1-multilingual/training):
  modal run train_gpu.py --smoke                            # ~10 min: widths 96 + 128, 300 steps, /smoke data
  modal run --detach train_gpu.py                           # full run: widths 96 + 128 in parallel
  modal run --detach train_gpu.py --widths 128              # one width only
  modal run --detach train_gpu.py --preset multi-fast       # batch 128, fewer steps
  (re-running the same command resumes each run from its last.pt)

Stage 2, spectral refinement of each run's best.pt (decoder only, both widths in parallel):
  modal run --detach train_gpu.py --stage refine            # multi_w96 -> multi_w96_refine, same for w128
  modal run --detach train_gpu.py --stage refine --widths 128

Run names: multi_w96, multi_w128 (smoke: smoke_w96, smoke_w128). Fetch samples / checkpoints:
  modal volume ls  pivo-multi-runs multi_w128/samples
  mkdir runs\\multi_w128
  modal volume get pivo-multi-runs multi_w128/samples runs/multi_w128/
  modal volume get pivo-multi-runs multi_w128/best.pt runs/multi_w128/best.pt
"""

import inspect
from pathlib import Path

import modal

HERE = Path(__file__).parent
DATA_VOLUME, RUNS_VOLUME = "pivo-multi-data", "pivo-multi-runs"
DATA_ROOT, SMOKE_ROOT = "/data/pivo_multi_v1", "/data/smoke"
PACK_DIR = "/runs/_feature_cache"
PIVO = "/pivo"  # PiVo-1.4M: onnx/ (FARGAN vocoder for samples) + acoustic.pt (parity test only)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "onnxruntime", "soundfile", "huggingface_hub")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('empero-ai/PiVo-1.4M', local_dir='{PIVO}', "
        "allow_patterns=['checkpoints/acoustic.pt', 'onnx/*'])\""
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE / "pivo_train", remote_path="/root/pivo_train")
    .add_local_dir(HERE / "tests", remote_path="/root/tests")
)

app = modal.App("pivo-multi-train", image=image)
data_vol = modal.Volume.from_name(DATA_VOLUME)
try:
    runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True, version=2)
except TypeError:  # older Modal client
    runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
VOLS = {"/data": data_vol, "/runs": runs_vol}

# Fastest first: Modal takes the first type that is free right now.
GPUS = ["H100", "A100-80GB", "A100"]
# the orchestrator only waits; never let Modal preempt it (if this Modal client supports it)
_ORCH = {"nonpreemptible": True} if "nonpreemptible" in inspect.signature(app.function).parameters else {}

DEFAULT_WIDTHS = "96,128"

# Presets (~295 h, ~150k clips, ~2.4k batches of 64 per epoch):
#   multi      : batch 64, LR 6e-4 -> 6e-5 cosine, bf16, 60k steps (~25 epochs). The recipe that
#                worked for PiVo-KN ("fast"), run longer for 5x the data. H100 smoke: ~27 steps/s
#                -> ~40 min + evals/samples, both widths side by side.
#   multi-fast : batch 128, LR 8e-4, 36k steps (~30 epochs). Same data seen in less wall time;
#                use when time matters more than the last bit of quality.
PRESETS = {
    "multi": dict(steps=60000, batch_size=64, lr=6e-4, min_lr=6e-5, warmup=1000, amp="bf16",
                  max_batch_frames=64 * 1200, eval_every=3000, log_every=100),
    "multi-fast": dict(steps=36000, batch_size=128, lr=8e-4, min_lr=8e-5, warmup=1000, amp="bf16",
                       max_batch_frames=128 * 1200, eval_every=3000, log_every=100),
}
REFINE_STEPS = 3000  # PiVo-KN used 1200 for 4 voices; 24 voices share the decoder here


def _gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def _parity():
    """Prove the model code still reproduces PiVo exactly (num_langs=0 path) before GPU time."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "/root/tests/test_parity.py", "--model-dir", PIVO], check=True)


def _pack(data_root):
    """One-file feature cache, built in a child process; instant when already up to date."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pivo_train.pack", "--data-root", data_root,
                    "--pack-dir", PACK_DIR], check=True, cwd="/root")
    runs_vol.commit()


def _names(widths, smoke):
    return [(int(w), f"{'smoke' if smoke else 'multi'}_w{int(w)}") for w in str(widths).split(",") if w.strip()]


@app.function(gpu=GPUS, cpu=16, memory=65536, timeout=24 * 3600, volumes=VOLS)
def train(run_name: str, width: int = 96, smoke: bool = False, preset: str = "multi", steps: int = 0,
          batch_size: int = 0, lr: float = 0.0):
    import os
    import sys
    print(f"[{run_name}] container up: GPU = {_gpu_name()}", flush=True)
    sys.path.insert(0, "/root")
    _parity()
    data_root = SMOKE_ROOT if smoke else DATA_ROOT
    _pack(data_root)  # normally built by the orchestrator already -> just a key check
    if os.path.exists(f"/runs/{run_name}/last.pt"):
        print(f"[{run_name}] found last.pt -> resuming (use another run name for a fresh run)", flush=True)
    p = dict(PRESETS[preset])
    if steps:
        p["steps"] = steps
    if batch_size:
        p["batch_size"] = batch_size
        p["max_batch_frames"] = batch_size * 1200
    if lr:
        p.update(lr=lr, min_lr=lr / 10)
    if smoke:
        p.update(steps=steps or 300, eval_every=150, log_every=25, warmup=50)
    print(f"[{run_name}] width {width}, preset {preset}: {p}", flush=True)

    from pivo_train.train import main
    argv = [
        "--data-root", data_root, "--from-scratch", "--width", str(width),
        "--onnx-dir", f"{PIVO}/onnx",
        "--out", f"/runs/{run_name}",
        "--steps", str(p["steps"]),
        "--batch-size", str(p["batch_size"]),
        "--max-batch-frames", str(p["max_batch_frames"]),
        "--lr", str(p["lr"]), "--min-lr", str(p["min_lr"]),
        "--warmup", str(p["warmup"]),
        "--amp", p["amp"],
        "--eval-every", str(p["eval_every"]),
        "--save-every", str(150 if smoke else 1000),  # last.pt often -> little lost on preemption
        "--log-every", str(p["log_every"]),
        "--num-workers", "0", "--preload", "--pack-dir", PACK_DIR,
        "--resume", "auto",  # a preempted + restarted container continues from last.pt
    ]
    result = main(argv, on_checkpoint=runs_vol.commit)
    runs_vol.commit()
    return {"run": run_name, "width": width, "gpu": _gpu_name(), **result}


@app.function(gpu=GPUS, cpu=8, memory=65536, timeout=6 * 3600, volumes=VOLS)
def refine(run_name: str, from_run: str, smoke: bool = False, steps: int = 0):
    """Stage 2 (PiVo report sec. 3.3): spectral refinement of /runs/<from_run>/best.pt.
    Only the decoder trains, against a small critic that is thrown away afterwards.
    Re-running the same run name resumes from its last.pt."""
    import sys
    print(f"[{run_name}] container up: GPU = {_gpu_name()}", flush=True)
    sys.path.insert(0, "/root")
    _parity()
    data_root = SMOKE_ROOT if smoke else DATA_ROOT
    _pack(data_root)
    init = f"/runs/{from_run}/best.pt"
    print(f"[{run_name}] stage 2 from {init}", flush=True)

    from pivo_train.refine import main
    argv = ["--data-root", data_root, "--init", init, "--onnx-dir", f"{PIVO}/onnx",
            "--out", f"/runs/{run_name}", "--preload", "--pack-dir", PACK_DIR]
    if smoke:
        argv += ["--steps", str(steps or 60), "--eval-every", "30", "--log-every", "10",
                 "--samples-per-voice", "1"]
    else:
        argv += ["--steps", str(steps or REFINE_STEPS), "--samples-per-voice", "2"]
    result = main(argv, on_checkpoint=runs_vol.commit)
    runs_vol.commit()
    return {"run": run_name, "from": from_run, "gpu": _gpu_name(), **(result or {})}


@app.function(cpu=16, memory=65536, timeout=3 * 3600, volumes=VOLS)
def pack_cache(data_root: str):
    """Feature cache (~8.5 GB for the full set): many CPUs + RAM, but a normal (cheap) container."""
    import sys
    import time
    sys.path.insert(0, "/root")
    t0 = time.time()
    _pack(data_root)
    return round(time.time() - t0)


# The orchestrator only waits on the GPU jobs. Non-preemptible containers bill CPU/RAM at 3x,
# so it stays tiny; the heavy cache build runs in pack_cache.
@app.function(cpu=1, memory=2048, timeout=24 * 3600, volumes=VOLS, **_ORCH)
def orchestrate(stage: str = "train", widths: str = DEFAULT_WIDTHS, smoke: bool = False,
                preset: str = "multi", steps: int = 0, batch_size: int = 0, lr: float = 0.0,
                gpu: str = "", from_runs: str = ""):
    """Build the feature cache once (CPU), then run one GPU job per width in parallel."""
    import json
    data_root = SMOKE_ROOT if smoke else DATA_ROOT
    for f in ("train.jsonl", "symbols.json", "feature_stats.json"):
        if not Path(data_root, "manifests", f).exists():
            raise SystemExit(f"{data_root}/manifests/{f} missing: run the data pipeline's finalize first")
    print(f"feature cache ready ({pack_cache.remote(data_root)}s)", flush=True)

    def with_gpu(fn):
        if not gpu:
            return fn
        try:
            return fn.with_options(gpu=gpu)
        except Exception as e:  # noqa: BLE001
            print(f"--gpu {gpu} ignored ({e}); using {GPUS}", flush=True)
            return fn

    if stage == "refine":
        sources = [s for s in from_runs.split(",") if s] or [name for _, name in _names(widths, smoke)]
        jobs = [(f"{src}_refine", src, smoke, steps) for src in sources]
        print(f"refining in parallel: {[j[0] for j in jobs]}", flush=True)
        results = list(with_gpu(refine).starmap(jobs, return_exceptions=True))
    else:
        jobs = [(name, w, smoke, preset, steps, batch_size, lr) for w, name in _names(widths, smoke)]
        print(f"training in parallel: {[j[0] for j in jobs]} (GPU preference {gpu or GPUS})", flush=True)
        results = list(with_gpu(train).starmap(jobs, return_exceptions=True))
    out = [r if isinstance(r, dict) else {"error": repr(r)} for r in results]
    print(json.dumps(out, indent=1), flush=True)
    return out


@app.local_entrypoint()
def run(stage: str = "train", smoke: bool = False, widths: str = DEFAULT_WIDTHS, preset: str = "multi",
        steps: int = 0, batch_size: int = 0, lr: float = 0.0, gpu: str = "", from_runs: str = ""):
    if preset not in PRESETS:
        raise SystemExit(f"--preset must be one of {list(PRESETS)}")
    if stage not in ("train", "refine"):
        raise SystemExit("--stage must be train or refine")
    res = orchestrate.remote(stage=stage, widths=widths, smoke=smoke, preset=preset, steps=steps,
                             batch_size=batch_size, lr=lr, gpu=gpu, from_runs=from_runs)
    for r in res:
        print(r)
