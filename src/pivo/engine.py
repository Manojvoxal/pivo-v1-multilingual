"""Validated local inference for the pinned PiVo ONNX release."""
import hashlib
import io
import json
import math
import wave
from pathlib import Path

import numpy as np

from .runtime import Runtime
from .text import SPECIAL, phonemize

MODEL_FILES = {f"onnx/{name}" for name in (
    "metadata.json", "encoder.onnx", "decoder.onnx", "conditioning.onnx", "vocoder_step.onnx"
)}


def verify_model(directory):
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / "model_manifest.json").read_text())
    if manifest.get("format") != "pivo-model-manifest-v1":
        raise ValueError("unsupported PiVo model manifest")
    rows = manifest.get("files", [])
    if len(rows) != len(MODEL_FILES) or {r["path"] for r in rows} != MODEL_FILES:
        raise ValueError("model manifest must pin the five PiVo runtime assets")
    for row in rows:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or unsafe model asset: {row['path']}")
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"model asset size mismatch: {row['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"model asset hash mismatch: {row['path']}")
    return root, manifest


def wav_bytes(audio, sample_rate=16000):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("expected finite mono audio")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio.clip(-1, 1) * 32767).round().astype("<i2").tobytes())
    return output.getvalue()


class PiVo:
    """Reuse one set of CPU sessions; each utterance gets independent state."""
    def __init__(self, model_dir, *, threads=1, max_chars=2000, max_tokens=512):
        if threads < 1 or max_chars < 1 or max_tokens < 2:
            raise ValueError("threads and input budgets must be positive")
        self.model_dir, self.manifest = verify_model(model_dir)
        self.runtime = Runtime(self.model_dir / "onnx", threads=threads)
        self.sample_rate = self.runtime.metadata["config"]["sample_rate"]
        self.max_chars, self.max_tokens = max_chars, max_tokens

    def _validate(self, text, voice, speed, overrides):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input must contain spoken text")
        if len(text) > self.max_chars:
            raise ValueError(f"input exceeds {self.max_chars} characters; split it into clauses")
        if voice not in ("Pi", "Vo"):
            raise ValueError("voice must be Pi or Vo")
        if not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0.5 <= speed <= 2:
            raise ValueError("speed must be finite and between 0.5 and 2.0")
        if overrides is not None:
            if not isinstance(overrides, dict) or len(overrides) > 128:
                raise ValueError("overrides must be a dictionary with at most 128 entries")
            for word, phones in overrides.items():
                if not isinstance(word, str) or len(word) > 64 or not isinstance(phones, list) or len(phones) > 64:
                    raise ValueError("invalid pronunciation override")
                if not all(isinstance(phone, str) for phone in phones):
                    raise ValueError("override phones must be ARPAbet strings")
        front = phonemize(text, overrides=overrides)
        if not any(s not in SPECIAL for s in front.symbols):
            raise ValueError("input must contain spoken text")
        if len(front.token_ids) > self.max_tokens:
            raise ValueError(f"input exceeds {self.max_tokens} frontend tokens; split it into clauses")

    def iter_audio(self, text, *, voice="Pi", speed=1.0, overrides=None, chunk_frames=16):
        """Yield float32, mono 10 ms PCM frames after full-text duration prediction."""
        self._validate(text, voice, speed, overrides)
        if not isinstance(chunk_frames, int) or chunk_frames < 1:
            raise ValueError("chunk_frames must be a positive integer")
        for chunk in self.runtime.iter_audio(text, voice=voice, speed=speed,
                                            overrides=overrides, chunk_frames=chunk_frames):
            if not np.isfinite(chunk).all():
                raise RuntimeError("model produced non-finite audio")
            yield chunk

    def synthesize(self, text, **kwargs):
        chunks = list(self.iter_audio(text, **kwargs))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def save(self, path, text, **kwargs):
        path = Path(path)
        audio = self.synthesize(text, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes(audio, self.sample_rate))
        return path
