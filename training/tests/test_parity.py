"""Parity check: our PyTorch PiVoAcoustic loaded with PiVo's acoustic.pt must reproduce the
official encoder.onnx / decoder.onnx outputs. Run:

    python tests/test_parity.py --model-dir <PiVo-1.4M dir>

Exits non-zero if any max-abs difference exceeds the tolerance.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pivo_train.model import load_pivo_checkpoint, phase_features  # noqa: E402

# PiVo English token ids for a fixed sentence (from pivo.text.phonemize), so the test
# needs no frontend: "NASA is ready for the next mission."
SENTENCES = {
    "nasa": [1, 21, 38, 25, 40, 3, 62, 32, 3, 24, 53, 12, 64, 3, 14, 44, 24, 3, 13, 40, 3, 21, 53, 18, 25, 27,
             3, 20, 62, 26, 40, 21, 4, 2],
    # "Hello, my name is Vo. How many parameters does a voice need?"
    "hello": [1, 16, 40, 19, 68, 5, 3, 20, 50, 3, 21, 59, 20, 3, 62, 32, 3, 29, 68, 4, 3, 16, 47, 3, 20, 53, 21,
              64, 3, 23, 55, 38, 20, 40, 27, 55, 32, 3, 12, 41, 32, 3, 40, 3, 29, 71, 25, 3, 21, 65, 12, 6, 2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--tokens", default=None, help="comma-separated token ids (optional)")
    a = ap.parse_args()
    import onnxruntime as ort

    model, _ = load_pivo_checkpoint(os.path.join(a.model_dir, "checkpoints", "acoustic.pt"), num_voices=2)
    model.eval()
    enc_s = ort.InferenceSession(os.path.join(a.model_dir, "onnx", "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(os.path.join(a.model_dir, "onnx", "decoder.onnx"), providers=["CPUExecutionProvider"])

    token_sets = [[int(x) for x in a.tokens.split(",")]] if a.tokens else list(SENTENCES.values())
    worst = 0.0
    for toks in token_sets:
        for voice in (0, 1):
            tokens = np.array([toks], dtype=np.int64)
            o_enc, o_logd = enc_s.run(None, {"tokens": tokens, "voice": np.array([voice], np.int64),
                                             "lengths": np.array([len(toks)], np.int64)})
            with torch.no_grad():
                t_enc, t_logd = model.encode(torch.from_numpy(tokens), torch.tensor([voice]), torch.tensor([len(toks)]))
            d_enc = np.abs(t_enc.numpy() - o_enc).max()
            d_logd = np.abs(t_logd.numpy() - o_logd).max()

            # durations exactly as pivo/runtime.py computes them
            dur = model.predict_durations(torch.from_numpy(o_logd), torch.from_numpy(tokens))[0]
            expanded = np.repeat(o_enc[0], dur.numpy(), axis=0)[None]
            phase = phase_features(dur).numpy()[None]
            expanded = np.concatenate((expanded, phase), axis=-1).astype(np.float32)

            # ONNX streaming decode (chunks of 16, like the runtime)
            cache = {n: np.zeros((1, 2, 0, 24), np.float32) for n in ("key0", "value0", "key1", "value1")}
            o_feats = []
            for p in range(0, expanded.shape[1], 16):
                r = dec_s.run(None, {"expanded": expanded[:, p:p + 16], "position": np.array(p, np.int64), **cache})
                o_feats.append(r[0])
                cache = dict(zip(("key0", "value0", "key1", "value1"), r[1:]))
            o_feats = np.concatenate(o_feats, axis=1)
            with torch.no_grad():
                t_full = model.denormalize(model.decode(torch.from_numpy(expanded))).numpy()
                caches = [torch.zeros(1, 2, 0, 24) for _ in range(4)]
                t_stream = []
                for p in range(0, expanded.shape[1], 16):
                    f, caches = model.decode_step(torch.from_numpy(expanded[:, p:p + 16]), torch.tensor(p), caches)
                    t_stream.append(f)
                t_stream = torch.cat(t_stream, 1).numpy()
            d_full = np.abs(t_full - o_feats).max()
            d_stream = np.abs(t_stream - o_feats).max()
            scale = np.abs(o_feats).max()
            print(f"voice {voice}: tokens={len(toks)} frames={expanded.shape[1]} | encoded {d_enc:.2e} | "
                  f"log_durations {d_logd:.2e} | features(full) {d_full:.2e} | features(stream) {d_stream:.2e} "
                  f"(feature scale {scale:.2f})")
            worst = max(worst, d_enc, d_logd, d_full, d_stream)
    ok = worst <= a.tol
    print("PARITY", "OK" if ok else "FAILED", f"(worst {worst:.2e}, tol {a.tol:.0e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
