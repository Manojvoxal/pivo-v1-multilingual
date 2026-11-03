"""Turn 20-dim Opus/FARGAN features into 16 kHz audio with PiVo's released vocoder ONNX
(conditioning.onnx + vocoder_step.onnx). Mirrors the vocoder half of pivo/runtime.py, so
what you hear during training is exactly what the deployed runtime would produce."""

import numpy as np

STEP_STATE = (("memory", 256), ("gru1", 160), ("gru2", 128), ("gru3", 128), ("frame_state", 164))


def periods(features):
    return (np.rint(256 / np.power(2.0, np.clip(features[..., 18], -10, 10) + 1.5))
            .clip(32, 255).astype(np.int64))


def deemphasis(audio, state=0.0, coefficient=0.85):
    out = np.empty_like(audio)
    for i, sample in enumerate(audio):
        state = float(sample) + coefficient * state
        out[i] = state
    return out, state


class Vocoder:
    def __init__(self, onnx_dir, threads=2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        mk = lambda n: ort.InferenceSession(f"{onnx_dir}/{n}.onnx", sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.conditioning = mk("conditioning")
        self.step = mk("vocoder_step")

    def __call__(self, features):
        """features: [frames, 20] float32 (NOT normalized). Returns float32 PCM at 16 kHz."""
        features = np.asarray(features, dtype=np.float32)
        state = {name: np.zeros((1, width), dtype=np.float32) for name, width in STEP_STATE}
        out, de_state = [], 0.0
        n = len(features)
        for i in range(n):
            prev = features[i - 1] if i > 0 else features[i]
            nxt = features[i + 1] if i + 1 < n else features[i]
            context = np.stack((prev, features[i], nxt))[None].astype(np.float32)
            per = periods(context)
            cond = self.conditioning.run(None, {"features": context, "periods": per})[0]
            frame = []
            for sub in range(4):
                res = self.step.run(None, {"conditioning": cond[:, 0, sub * 80:(sub + 1) * 80],
                                           "period": per[:, 1], **state})
                state = dict(zip([s for s, _ in STEP_STATE], res[1:]))
                frame.append(res[0][0])
            pcm, de_state = deemphasis(np.concatenate(frame), de_state)
            out.append(pcm)
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def write_wav(path, audio, sr=16000):
    import wave
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).round().astype("<i2").tobytes())
