"""Deterministic within-phone timing features for the optional acoustic control."""

import math

import numpy as np


def phoneme_phase_features(durations, max_token_frames=200):
    durations = np.asarray(durations)
    if durations.ndim != 1 or durations.dtype.kind not in "iu" or (durations < 0).any():
        raise ValueError("phase features require nonnegative integer durations")
    if max_token_frames < 1:
        raise ValueError("max token frames must be positive")
    counts = np.repeat(durations, durations).astype(np.float32)
    starts = np.repeat(np.cumsum(durations) - durations, durations)
    fraction = (np.arange(len(counts), dtype=np.float32) - starts.astype(np.float32) + 0.5) / counts
    return np.stack(
        (
            2 * fraction - 1,
            np.sin(math.pi * fraction),
            np.cos(math.pi * fraction),
            np.log1p(counts) / math.log1p(max_token_frames),
        ),
        axis=-1,
    ).astype(np.float32)
