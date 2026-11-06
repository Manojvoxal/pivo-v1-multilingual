"""Synthesize the 24 voices with one variant and compare it with FP32.

usage: python evaluate.py VARIANTS_ROOT SENTENCES_JSON OUT_DIR ACOUSTIC_VARIANT VOCODER_VARIANT [--samples]

ACOUSTIC_VARIANT supplies encoder + decoder, VOCODER_VARIANT supplies conditioning + vocoder_step,
so the two halves can be tested separately. Same inference steps as web/app/engine.js.
"""
import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT, SENT, OUT, AC, VO = sys.argv[1:6]
SAVE = "--samples" in sys.argv
NOQDQ = "--noqdq" in sys.argv
STATE = [("memory", 256), ("gru1", 160), ("gru2", 128), ("gru3", 128), ("frame_state", 164)]
CACHE = ["key0", "value0", "key1", "value1"]
LMAX = math.log1p(200)


def js_round(x):
    return math.floor(x + 0.5)


def sess(variant, name):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    if NOQDQ:
        # dequantize the weights once at load (constant folding) instead of fusing into integer kernels
        o.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(f"{ROOT}/{variant}/{name}.onnx", o, providers=["CPUExecutionProvider"])


class Model:
    def __init__(self, ac, vo):
        self.enc, self.dec = sess(ac, "encoder"), sess(ac, "decoder")
        self.cond, self.step = sess(vo, "conditioning"), sess(vo, "vocoder_step")
        meta = json.loads(Path(f"{ROOT}/fp32/metadata.json").read_text(encoding="utf-8"))
        self.hd, self.kv = meta["config"]["head_dim"], meta["config"]["kv_heads"]
        self.voices, self.langs = meta["voices"], meta["language_ids"]
        self.step_out = [o.name for o in self.step.get_outputs()]

    def durations(self, ids, voice, lang):
        i64 = lambda a: np.array(a, dtype=np.int64)
        enc, logd = self.enc.run(None, {"tokens": i64([ids]), "voice": i64([voice]), "lang": i64([lang]),
                                        "lengths": i64([len(ids)])})
        dur = []
        for i, tok in enumerate(ids):
            if tok <= 2:
                dur.append(0)
                continue
            d = min(max(js_round(math.expm1(min(max(float(logd[0][i]), 0.0), LMAX))), 0), 200)
            dur.append(max(d, 1) if tok >= 10 else d)
        return enc[0], dur

    def features(self, enc, dur):
        rows = []
        for i, n in enumerate(dur):
            for k in range(n):
                f = (k + 0.5) / n
                rows.append(np.concatenate([enc[i], np.array([2 * f - 1, math.sin(math.pi * f), math.cos(math.pi * f),
                                                              math.log1p(n) / LMAX], np.float32)]))
        x = np.stack(rows).astype(np.float32)[None]
        cache = {c: np.zeros((1, self.kv, 0, self.hd), np.float32) for c in CACHE}
        out = []
        for p in range(0, x.shape[1], 16):
            r = self.dec.run(None, {"expanded": x[:, p:p + 16], "position": np.array(p, np.int64), **cache})
            out.append(r[0][0])
            cache = dict(zip(CACHE, r[1:]))
        return np.concatenate(out).astype(np.float32)

    def vocode(self, feats):
        F = len(feats)
        st = {n: np.zeros((1, w), np.float32) for n, w in STATE}
        de, out = 0.0, []
        per = lambda v: min(255, max(32, js_round(256 / 2 ** (min(10, max(-10, float(v))) + 1.5))))
        for i in range(F):
            rows = [max(i - 1, 0), i, min(i + 1, F - 1)]
            ctx = np.stack([feats[r] for r in rows])[None].astype(np.float32)
            pp = np.array([[per(feats[r][18]) for r in rows]], np.int64)
            c = self.cond.run(None, {"features": ctx, "periods": pp})[0].reshape(-1)
            for s in range(4):
                res = dict(zip(self.step_out, self.step.run(None, {"conditioning": c[s * 80:s * 80 + 80][None],
                                                                     "period": pp[0, 1:2], **st})))
                st = {n: res[n + "_out"] for n, _ in STATE}
                for v in res["excitation"].reshape(-1):
                    de = float(v) + 0.85 * de
                    out.append(de)
        return np.array(out, np.float32)


def logmel(x, n_mels=80):
    n_fft, hop = 1024, 160
    x = np.pad(x, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    pts = hz(np.linspace(mel(0), mel(8000), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / 16000).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return 10 * np.log10(np.maximum(spec @ fb.T, 1e-10))


def save_wav(path, pcm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.round(np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    sents = json.loads(Path(SENT).read_text(encoding="utf-8"))
    ref = Model("fp32", "fp32")
    m = Model(AC, VO)
    tag = (AC if AC == VO else f"ac-{AC}_vo-{VO}") + ("@noqdq" if NOQDQ else "")
    results, synth_t, audio_s = [], 0.0, 0.0
    for lang, s in sents.items():
        for g in ("f", "m"):
            vid, lid = ref.voices[f"{lang}_{g}"], ref.langs[lang]
            nat, forced_ref, forced = [], [], []
            same_tok = tot_tok = 0
            d_feat, d_pitch, n_voiced = [], [], 0
            for ids in s["chunks"]:
                enc_r, dur_r = ref.durations(ids, vid, lid)
                t0 = time.perf_counter()
                enc_v, dur_v = m.durations(ids, vid, lid)
                fv = m.features(enc_v, dur_v)
                nat.append(m.vocode(fv))
                synth_t += time.perf_counter() - t0
                audio_s += len(nat[-1]) / 16000
                spoken = [i for i, t in enumerate(ids) if t >= 10]
                same_tok += sum(dur_r[i] == dur_v[i] for i in spoken)
                tot_tok += len(spoken)
                # forced: both use the FP32 durations, so audio lines up frame by frame
                fr = ref.features(enc_r, dur_r)
                ff = m.features(enc_v, dur_r)
                d_feat.append(np.sqrt(np.mean((fr[:, :18] - ff[:, :18]) ** 2)))
                voiced = fr[:, 19] > 0.3
                n_voiced += voiced.sum()
                d_pitch.extend(np.abs(fr[voiced, 18] - ff[voiced, 18]) * 1200)   # cents
                forced_ref.append(ref.vocode(fr))
                forced.append(m.vocode(ff))
            a, b = np.concatenate(forced_ref), np.concatenate(forced)
            lsd = float(np.mean(np.abs(logmel(a) - logmel(b))))
            pcm = np.concatenate(nat)
            r = dict(voice=f"{lang}_{g}", seconds=round(len(pcm) / 16000, 2),
                     dur_match=round(same_tok / tot_tok, 4),
                     cep_rmse=round(float(np.mean(d_feat)), 4),
                     pitch_cents=round(float(np.mean(d_pitch)), 1),
                     mel_db=round(lsd, 3),
                     wave_corr=round(float(np.corrcoef(a, b)[0, 1]), 5),
                     peak=round(float(np.abs(pcm).max()), 3))
            results.append(r)
            print(tag, r, flush=True)
            if SAVE:
                save_wav(Path(OUT) / "samples" / tag / f"{lang}_{g}.wav", pcm)
    Path(OUT, "metrics").mkdir(parents=True, exist_ok=True)
    Path(OUT, "metrics", f"{tag}.json").write_text(json.dumps(
        dict(tag=tag, acoustic=AC, vocoder=VO, rtf=synth_t / audio_s, voices=results), indent=1))
    print(tag, "RTF", round(synth_t / audio_s, 3))


main()
