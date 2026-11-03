"""Build the one-file feature caches (train/val) used by `--preload --pack-dir`.

Runs as a separate process before training, so the training process stays free to
answer Modal heartbeats. Reading ~36k small .npy files from a network volume is
latency-bound, so it fans out over several processes x threads and prints progress.

    python -m pivo_train.pack --data-root /data/pivo_kn_en_v1 --pack-dir /runs/_feature_cache
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from .data import read_manifest

MAX_UTTERANCE_FRAMES = 3000  # same filter the trainer applies (model config max_utterance_frames)


def _load_chunk(args):
    root, paths, threads = args
    load = lambda p: np.load(os.path.join(root, p)).astype(np.float32)
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(load, paths))


def pack_rows(root, rows, pack_path, procs=16, threads=16, chunk=400):
    # key must match pivo_train.data.PiVoDataset._load_all
    key = hashlib.sha1("\n".join(f"{r['features']}:{r['n_frames']}" for r in rows).encode()).hexdigest()
    if os.path.exists(pack_path + ".json"):
        try:
            with open(pack_path + ".json") as fh:
                if json.load(fh).get("key") == key:
                    print(f"{os.path.basename(pack_path)}: cache already up to date", flush=True)
                    return
        except (ValueError, OSError):
            pass  # unreadable / partial -> rebuild
    paths = [r["features"] for r in rows]
    jobs = [(root, paths[i:i + chunk], threads) for i in range(0, len(paths), chunk)]
    parts, done, t0 = [None] * len(jobs), 0, time.time()
    with ProcessPoolExecutor(procs) as ex:
        futs = {ex.submit(_load_chunk, j): k for k, j in enumerate(jobs)}
        for f in as_completed(futs):
            parts[futs[f]] = f.result()
            done += len(parts[futs[f]])
            rate = done / max(time.time() - t0, 1e-6)
            print(f"{os.path.basename(pack_path)}: {done}/{len(paths)} files "
                  f"({rate:.0f}/s, ~{(len(paths) - done) / max(rate, 1e-6) / 60:.1f} min left)", flush=True)
    feats = [a for p in parts for a in p]
    for r, a in zip(rows, feats):
        if len(a) != r["n_frames"]:
            raise ValueError(f"{r['id']}: {len(a)} frames, manifest says {r['n_frames']}")
    os.makedirs(os.path.dirname(pack_path), exist_ok=True)
    off = np.concatenate(([0], np.cumsum([len(a) for a in feats]))).tolist()
    np.save(pack_path + ".npy", np.concatenate(feats) if feats else np.zeros((0, 20), np.float32))
    with open(pack_path + ".json.tmp", "w") as fh:
        json.dump({"key": key, "offsets": off}, fh)
    os.replace(pack_path + ".json.tmp", pack_path + ".json")
    print(f"{os.path.basename(pack_path)}: wrote {len(feats)} files in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.data_root))
    train = [r for r in read_manifest(a.data_root, "train") if r["n_frames"] <= MAX_UTTERANCE_FRAMES]
    splits = {"train": train}
    for s in ("val", "test"):
        if os.path.exists(os.path.join(a.data_root, "manifests", f"{s}.jsonl")):
            splits["val"] = read_manifest(a.data_root, s)
            break
    for split, rows in splits.items():
        pack_rows(a.data_root, rows, os.path.join(a.pack_dir, f"{tag}_{split}"), a.procs, a.threads)


if __name__ == "__main__":
    main()
