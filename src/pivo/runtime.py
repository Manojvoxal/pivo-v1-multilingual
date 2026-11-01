"""Reference CPU ONNX runtime. Exported graphs also expose a native-host interface."""

import json
from pathlib import Path

import numpy as np

from .audio import deemphasis
from .phase import phoneme_phase_features
from .text import SPECIAL, VOICE_IDS, phonemize


class Runtime:
    def __init__(self, directory, *, threads=1, allow_untrained=False):
        import onnxruntime as ort

        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["format"] != "pivo-onnx-v1":
            raise ValueError("unsupported export format")
        if not self.metadata["trained"] and not allow_untrained:
            raise ValueError(
                "this export is untrained; supply trained checkpoints before synthesis"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.sessions = {
            name: ort.InferenceSession(
                str(directory / f"{name}.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name in ["encoder", "decoder", "conditioning", "vocoder_step"]
        }

    @staticmethod
    def periods(features):
        return (
            np.rint(256 / np.power(2.0, np.clip(features[..., 18], -10, 10) + 1.5))
            .clip(32, 255)
            .astype(np.int64)
        )

    def iter_audio(self, text, *, voice="Pi", speed=1.0, chunk_frames=16, overrides=None):
        if voice not in VOICE_IDS or not 0.5 <= speed <= 2.0 or chunk_frames < 1:
            raise ValueError("invalid voice, speed, or chunk size")
        front = phonemize(text, overrides=overrides)
        if not any(
            s not in ("<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":")
            for s in front.symbols
        ):
            return
        c = self.metadata["config"]
        tokens = np.array([front.token_ids], dtype=np.int64)
        encoded, log_durations = self.sessions["encoder"].run(
            None,
            {
                "tokens": tokens,
                "voice": np.array([VOICE_IDS[voice]], dtype=np.int64),
                "lengths": np.array([len(front.token_ids)], dtype=np.int64),
            },
        )
        durations = (
            np.rint(np.expm1(log_durations.clip(0, np.log1p(c["max_token_frames"]))) / speed)
            .clip(0, c["max_token_frames"])
            .astype(np.int64)
        )
        durations = np.maximum(durations, (tokens >= len(SPECIAL)).astype(np.int64))
        durations[tokens <= 2] = 0
        if durations.sum() > c["max_utterance_frames"]:
            raise ValueError("text exceeds the utterance frame budget; split it into clauses")
        expanded = np.repeat(encoded[0], durations[0], axis=0)[None]
        if c.get("phoneme_phase", False):
            phase = phoneme_phase_features(durations[0], c["max_token_frames"])[None]
            expanded = np.concatenate((expanded, phase), axis=-1)
        cache = {
            name: np.zeros((1, c["kv_heads"], 0, c["width"] // c["heads"]), dtype=np.float32)
            for name in self.metadata["cache_names"]
        }
        state = {
            name: np.zeros((1, width), dtype=np.float32)
            for name, width in zip(self.metadata["step_state_names"], (256, 160, 128, 128, 164))
        }
        queue = []
        previous = None
        deemphasis_state = 0.0
        for position in range(0, expanded.shape[1], chunk_frames):
            result = self.sessions["decoder"].run(
                None,
                {
                    "expanded": expanded[:, position : position + chunk_frames],
                    "position": np.array(position, dtype=np.int64),
                    **cache,
                },
            )
            cache = dict(zip(self.metadata["cache_names"], result[1:]))
            queue.extend(result[0][0])
            final = position + chunk_frames >= expanded.shape[1]
            while len(queue) >= 2 or (final and queue):
                current = queue.pop(0)
                following = queue[0] if queue else current
                previous = current if previous is None else previous
                context = np.stack((previous, current, following))[None].astype(np.float32)
                periods = self.periods(context)
                conditioning = self.sessions["conditioning"].run(
                    None, {"features": context, "periods": periods}
                )[0]
                frame = []
                for subframe in range(4):
                    result = self.sessions["vocoder_step"].run(
                        None,
                        {
                            "conditioning": conditioning[:, 0, subframe * 80 : (subframe + 1) * 80],
                            "period": periods[:, 1],
                            **state,
                        },
                    )
                    state = dict(zip(self.metadata["step_state_names"], result[1:]))
                    frame.append(result[0][0])
                pcm, deemphasis_state = deemphasis(np.concatenate(frame), deemphasis_state)
                previous = current
                yield pcm

    def synthesize(self, text, **kwargs):
        chunks = list(self.iter_audio(text, **kwargs))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
