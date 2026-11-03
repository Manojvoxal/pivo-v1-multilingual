"""Speaker discovery helpers used for SPRINGLab/IndicTTS-English (no speaker labels)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivo_multi import speakers  # noqa: E402


def test_median_f0():
    t = np.arange(16000 * 2) / 16000
    for f in (110.0, 220.0):
        x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8)).astype(np.float32)
        assert abs(speakers.median_f0(x) - f) / f < 0.05
    assert speakers.median_f0(np.zeros(16000, np.float32)) == 0.0


def test_path_keys():
    paths = [f"train_tamilfemale_{i:05d}.wav" for i in range(400)] + \
            [f"train_hindimale_{i:05d}.wav" for i in range(400)]
    keys = speakers.path_speaker_keys(paths, min_rows=300, min_speakers=2)
    assert set(keys) == {"train_tamilfemale", "train_hindimale"}
    assert speakers.path_speaker_keys(["audio.wav"] * 1000) is None
    assert speakers.path_speaker_keys([""] * 1000) is None


def test_cluster_and_pick():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 64))
    emb = np.concatenate([c + 0.15 * rng.standard_normal((200, 64)) for c in centers])
    labels, info = speakers.cluster_embeddings(emb, k_range=(2, 10), sample=1000)
    assert info["merged_to"] == 5
    truth = np.repeat(np.arange(5), 200)
    for k in range(5):  # each true speaker maps to one cluster
        assert len(set(labels[truth == k])) == 1
    summ = {"a": {"gender": "f", "hours": 25.0}, "b": {"gender": "f", "hours": 12.0},
            "c": {"gender": "m", "hours": 10.0}}
    assert speakers.pick_voices(summ) == {"f": "b", "m": "c"}          # 25 h = merged talents
    assert speakers.pick_voices(summ, ["a", "c"]) == {"f": "a", "m": "c"}
