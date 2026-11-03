"""Training data: rows from manifests/{train,val,test}.jsonl written by the data pipeline
(see pivo_data/DATA_FORMAT.md). Each item gives token ids, per-token frame durations,
voice id and a [frames, 20] Opus/FARGAN feature matrix."""

import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def read_manifest(root, split):
    path = os.path.join(root, "manifests", f"{split}.jsonl")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:  # cheap integrity checks; the pipeline already guarantees these
        assert len(r["token_ids"]) == len(r["durations"]), r["id"]
        assert sum(r["durations"]) == r["n_frames"], r["id"]
    return rows


class PiVoDataset(Dataset):
    def __init__(self, root, rows, feature_mean, feature_std, max_frames=None, preload=False,
                 threads=32, pack_path=None):
        """preload: keep all features in RAM. pack_path: optional single-file cache of every
        feature matrix (one .npy + offsets). Reading thousands of small files from a network
        volume is slow; the pack is built once and then loads in seconds on later runs."""
        self.root = root
        self.rows = [r for r in rows if max_frames is None or r["n_frames"] <= max_frames]
        self.mean = np.asarray(feature_mean, dtype=np.float32)
        self.std = np.asarray(feature_std, dtype=np.float32)
        self.cache = None
        if preload:
            self.cache = self._load_all(threads, pack_path)

    def _load_all(self, threads, pack_path):
        import hashlib
        key = hashlib.sha1("\n".join(f"{r['features']}:{r['n_frames']}" for r in self.rows).encode()).hexdigest()
        if pack_path and os.path.exists(pack_path + ".json"):
            with open(pack_path + ".json") as fh:
                meta = json.load(fh)
            if meta.get("key") == key:
                big = np.load(pack_path + ".npy")
                off = meta["offsets"]
                return [big[off[i]:off[i + 1]] for i in range(len(self.rows))]
        from concurrent.futures import ThreadPoolExecutor
        load = lambda r: np.load(os.path.join(self.root, r["features"])).astype(np.float32)
        with ThreadPoolExecutor(threads) as ex:
            feats = list(ex.map(load, self.rows))
        if pack_path:
            os.makedirs(os.path.dirname(pack_path), exist_ok=True)
            off = np.concatenate(([0], np.cumsum([len(f) for f in feats]))).tolist()
            np.save(pack_path + ".npy", np.concatenate(feats) if feats else np.zeros((0, 20), np.float32))
            with open(pack_path + ".json", "w") as fh:
                json.dump({"key": key, "offsets": off}, fh)
        return feats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.cache is not None:
            feats = self.cache[i]
        else:
            feats = np.load(os.path.join(self.root, r["features"])).astype(np.float32)
        if len(feats) != r["n_frames"]:
            raise ValueError(f"{r['id']}: {len(feats)} feature frames, manifest says {r['n_frames']}")
        return {
            "id": r["id"],
            "tokens": torch.tensor(r["token_ids"], dtype=torch.long),
            "durations": torch.tensor(r["durations"], dtype=torch.long),
            "voice": int(r["voice_id"]),
            "lang": r["lang"],
            "lang_id": int(r.get("lang_id", 0)),  # multilingual manifests carry lang_id
            "features": torch.from_numpy((feats - self.mean) / self.std),  # normalized like PiVo
        }


def collate(batch):
    b = len(batch)
    t_max = max(len(x["tokens"]) for x in batch)
    f_max = max(len(x["features"]) for x in batch)
    tokens = torch.zeros(b, t_max, dtype=torch.long)          # 0 = <pad>
    durations = torch.zeros(b, t_max, dtype=torch.long)
    features = torch.zeros(b, f_max, batch[0]["features"].shape[1])
    token_lengths = torch.zeros(b, dtype=torch.long)
    frame_lengths = torch.zeros(b, dtype=torch.long)
    for i, x in enumerate(batch):
        n, f = len(x["tokens"]), len(x["features"])
        tokens[i, :n] = x["tokens"]
        durations[i, :n] = x["durations"]
        features[i, :f] = x["features"]
        token_lengths[i] = n
        frame_lengths[i] = f
    return {
        "ids": [x["id"] for x in batch],
        "langs": [x["lang"] for x in batch],
        "voices": torch.tensor([x["voice"] for x in batch], dtype=torch.long),
        "lang_ids": torch.tensor([x["lang_id"] for x in batch], dtype=torch.long),
        "tokens": tokens, "durations": durations, "features": features,
        "token_lengths": token_lengths, "frame_lengths": frame_lengths,
    }


class BucketBatchSampler(Sampler):
    """Batches of similar length (less padding), reshuffled every epoch.

    Optionally caps the frames per batch so long clips get smaller batches.
    """

    def __init__(self, rows, batch_size, max_batch_frames=None, seed=0, drop_last=True):
        self.lengths = [r["n_frames"] for r in rows]
        self.batch_size = batch_size
        self.max_batch_frames = max_batch_frames
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        # sort inside large shuffled chunks -> similar lengths, still random order overall
        chunk = self.batch_size * 50
        batches = []
        for c in range(0, len(idx), chunk):
            part = sorted(idx[c:c + chunk], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in part:
                new_max = max(cur_max, self.lengths[i])
                too_many = len(cur) >= self.batch_size
                too_long = self.max_batch_frames and cur and new_max * (len(cur) + 1) > self.max_batch_frames
                if too_many or too_long:
                    batches.append(cur)
                    cur, new_max = [], self.lengths[i]
                cur.append(i)
                cur_max = new_max
            if cur and (not self.drop_last or len(cur) == self.batch_size):
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def describe(rows):
    by = {}
    for r in rows:
        k = (r["speaker"], r["voice_id"], r["lang"])
        by.setdefault(k, [0, 0.0])
        by[k][0] += 1
        by[k][1] += r["seconds"] / 3600
    return {f"{k[0]} (voice {k[1]}, {k[2]})": f"{v[0]} clips, {v[1]:.2f} h" for k, v in sorted(by.items(), key=lambda kv: kv[0][1])}
