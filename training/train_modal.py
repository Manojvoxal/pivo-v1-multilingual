"""
Train the Kannada + English PiVo acoustic model on Modal (stage 1, supervised).

Reads the dataset made by pivo_data/prepare_pivo_v4.py from the `pivo-data` volume,
warm-starts from PiVo-1.4M's English acoustic.pt (downloaded into the image), and
writes checkpoints, logs and sample WAVs to the `pivo-runs` volume.

Usage (from pivo-v1/training):
  modal run train_modal.py --smoke                     # ~5 min GPU test on /smoke data
  modal run --detach train_modal.py --run-name kn_en_v1 # full training, PiVo recipe (batch 8)
  modal run --detach train_modal.py --run-name kn_en_fast --preset fast   # batch 64, ~20-30 min
  modal run --detach train_modal.py --run-name kn_en_v1 --resume   # continue from last.pt
  modal run train_modal.py --smoke --gpu H100          # force a specific GPU type

Listen to samples / fetch checkpoints:
  modal volume ls  pivo-runs kn_en_v1/samples
  modal volume get pivo-runs kn_en_v1/samples/step_010000 ./samples_10k
  modal volume get pivo-runs kn_en_v1/best.pt ./best.pt
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent
DATA_VOLUME, RUNS_VOLUME = "pivo-data", "pivo-runs"
PIVO = "/pivo"  # PiVo-1.4M (acoustic.pt + onnx) baked into the image

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

app = modal.App("pivo-kn-en-train", image=image)
data_vol = modal.Volume.from_name(DATA_VOLUME)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)


# GPU fallback list: Modal takes the first type that is free right now (the model is tiny,
# so any of these is plenty). Override for one run with:  --gpu H100
GPUS = ["A100", "H100", "A10G", "L4"]


# Two presets:
#   pivo : PiVo's own stage-1 recipe from the technical report -> batch 8, LR 3e-4 constant
#          after 200 warm-up steps, fp32. 80k steps ~= 18 epochs / ~640k clips (PiVo: 85,772
#          updates, ~686k clips). Slowest, safest. Rough time on A100: ~1-1.5 h.
#   fast : batch 64, LR 6e-4 with cosine decay, bf16. 9k steps. ~20-30 min.
PRESETS = {
    "pivo": dict(steps=80000, batch_size=8, lr=3e-4, min_lr=3e-4, warmup=200, amp="none",
                 max_batch_frames=8 * 1500, eval_every=4000, log_every=200),
    "fast": dict(steps=9000, batch_size=64, lr=6e-4, min_lr=6e-5, warmup=300, amp="bf16",
                 max_batch_frames=64 * 1200, eval_every=1500, log_every=100),
}


@app.function(gpu=GPUS, cpu=16, memory=65536, timeout=20 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def train(run_name: str, smoke: bool = False, preset: str = "pivo", steps: int = 0,
          batch_size: int = 0, lr: float = 0.0, resume: bool = False):
    import os
    import subprocess
    import sys
    import torch
    print(f"container up: GPU = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    sys.path.insert(0, "/root")
    # 1) prove the model code reproduces PiVo exactly before spending GPU time
    subprocess.run([sys.executable, "/root/tests/test_parity.py", "--model-dir", PIVO], check=True)

    # 2) build the one-file feature cache in a CHILD process (keeps this process free to
    #    answer Modal heartbeats); instant if the cache already exists
    data_root = "/data/smoke" if smoke else "/data/pivo_kn_en_v1"
    subprocess.run([sys.executable, "-m", "pivo_train.pack", "--data-root", data_root,
                    "--pack-dir", "/runs/_feature_cache"], check=True, cwd="/root")
    runs_vol.commit()

    if os.path.exists(f"/runs/{run_name}/last.pt"):
        print(f"found /runs/{run_name}/last.pt -> resuming (use a new --run-name for a fresh run)", flush=True)
    p = dict(PRESETS[preset])
    if steps:
        p["steps"] = steps
    if batch_size:
        p["batch_size"] = batch_size
        p["max_batch_frames"] = batch_size * 1500
    if lr:
        p["lr"] = lr
        p["min_lr"] = lr if preset == "pivo" else lr / 10
    if smoke:
        p.update(steps=300, eval_every=150, log_every=25)
    print(f"preset {preset}: {p}", flush=True)

    from pivo_train.train import main
    argv = [
        "--data-root", data_root,
        "--pivo-ckpt", f"{PIVO}/checkpoints/acoustic.pt",
        "--onnx-dir", f"{PIVO}/onnx",
        "--out", f"/runs/{run_name}",
        "--steps", str(p["steps"]),
        "--batch-size", str(p["batch_size"]),
        "--max-batch-frames", str(p["max_batch_frames"]),
        "--lr", str(p["lr"]), "--min-lr", str(p["min_lr"]),
        "--warmup", str(p["warmup"]),
        "--amp", p["amp"],
        "--eval-every", str(p["eval_every"]),
        "--save-every", str(150 if smoke else 1000),   # last.pt often -> little lost on preemption
        "--log-every", str(p["log_every"]),
        "--num-workers", "0", "--preload", "--pack-dir", "/runs/_feature_cache",
    ] + ["--resume", "auto"]   # always: if Modal preempts and restarts the container, continue from last.pt
    result = main(argv, on_checkpoint=runs_vol.commit)
    runs_vol.commit()
    return result


@app.local_entrypoint()
def run(run_name: str = "", smoke: bool = False, preset: str = "pivo", steps: int = 0,
        batch_size: int = 0, lr: float = 0.0, resume: bool = False, gpu: str = ""):
    name = run_name or ("smoke" if smoke else "kn_en_v1")
    fn = train.with_options(gpu=gpu) if gpu else train
    print(fn.remote(name, smoke=smoke, preset=preset, steps=steps, batch_size=batch_size,
                    lr=lr, resume=resume))
