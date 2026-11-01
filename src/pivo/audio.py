import wave
from pathlib import Path

import numpy as np

NATIVE_TRAINING_DELAY = 80


def native_feature_audio(audio, frames):
    """Match Opus dump_data's half-frame delay for native analysis features.

    Native feature k describes source samples [160*k-80, 160*k+80).
    The leading 80 target samples are zero; the unavailable tail is discarded.
    This returns ordinary PCM: pre-emphasis is applied by the trainer afterward.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or frames < 1 or not np.isfinite(audio).all():
        raise ValueError("expected finite mono audio and positive frame count")
    if not frames * 160 <= len(audio) < (frames + 1) * 160:
        raise ValueError("native feature/audio lengths disagree")
    return np.pad(audio, (NATIVE_TRAINING_DELAY, 0))[: frames * 160]


def read_wav(path, expected_rate=16000):
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != expected_rate
        ):
            raise ValueError(f"expected mono PCM16 WAV at {expected_rate} Hz: {path}")
        audio = (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        )
    if audio.size < 160 or not np.isfinite(audio).all():
        raise ValueError("audio is too short or non-finite")
    return audio


def write_wav(path, audio, sample_rate=16000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("refusing to save non-finite audio")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio.clip(-1, 1) * 32767).round().astype("<i2").tobytes())


def pitch_periods(features):
    """Pinned FARGAN feature 18 is log2(256/period) - 1.5."""
    import torch

    return (
        (256 / torch.pow(2.0, features[..., 18].clamp(-10, 10) + 1.5)).round().clamp(32, 255).long()
    )


def preemphasis(audio, coefficient=0.85):
    import torch

    return torch.cat((audio[..., :1], audio[..., 1:] - coefficient * audio[..., :-1]), -1)


def deemphasis(audio, state=0.0, coefficient=0.85):
    """Non-neural DSP; state must persist across output chunks."""
    audio = np.asarray(audio, dtype=np.float32)
    out = np.empty_like(audio)
    for i, sample in enumerate(audio):
        state = float(sample) + coefficient * state
        out[i] = state
    return out, state
