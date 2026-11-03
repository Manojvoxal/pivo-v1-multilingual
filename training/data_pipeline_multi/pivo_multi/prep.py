"""Audio, alignment and feature helpers for building PiVo training data.
(Copied unchanged from the Kannada pipeline pivo_kn_v3/prep.py, except that
mfa_phone now lives in symbols.py and also maps the nasal-vowel "~".)

Conventions (taken from PiVo-1.4M's own code):
  * 16 kHz mono, 10 ms frames, hop 160 samples;
  * acoustic features = first 20 of the 36 floats Opus `fargan_demo -features`
    writes per frame (18 cepstral coefs, pitch feature, pitch correlation);
    feature k describes samples [160k-80, 160k+80) of the clip;
  * per-token integer durations in frames: <bos>/<eos> get 0, every phone >= 1,
    <space>/punctuation tokens carry the pauses between words (may be 0);
  * sum(durations) == number of feature frames.
"""

import json
import math
import os
import subprocess
import tempfile

import numpy as np

from .symbols import PIVO_SPECIAL, mfa_phone  # noqa: F401  (mfa_phone re-exported)

SR = 16000
HOP = 160
MAX_TOKEN_FRAMES = 200
FEATURE_DIM = 20
OPUS_TOTAL_FEATURES = 36
_WARNED_SIZE = False


class DropClip(Exception):
    """Raised when a clip fails a check; the message is the drop reason."""


# --------------------------------------------------------------------------- audio
def to_16k_mono(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        try:
            import soxr
            audio = soxr.resample(audio, sr, SR, quality="VHQ").astype(np.float32)
        except ImportError:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), SR)
            audio = resample_poly(audio, SR // g, int(sr) // g).astype(np.float32)
    return audio


def check_and_normalize(audio, target_lufs=-20.0, peak=0.95):
    """Reject clipped / silent / non-finite audio, then loudness-normalize."""
    if not np.isfinite(audio).all():
        raise DropClip("non_finite_audio")
    if len(audio) < SR // 2:
        raise DropClip("too_short_raw")
    clipped = np.mean(np.abs(audio) >= 0.999)
    if clipped > 1e-3:
        raise DropClip("clipping")
    import pyloudnorm as pyln
    loudness = pyln.Meter(SR).integrated_loudness(audio.astype(np.float64))
    if not np.isfinite(loudness) or loudness < -60:
        raise DropClip("near_silent")
    out = audio * (10 ** ((target_lufs - loudness) / 20))
    top = np.max(np.abs(out))
    if top > peak:
        out = out * (peak / top)
    return out.astype(np.float32)


def to_pcm16(audio):
    return (np.clip(audio, -1, 1) * 32767).round().astype("<i2")


def write_wav(path, pcm16):
    import wave
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())


def read_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


# --------------------------------------------------------------------------- MFA
def mfa_lab_text(phonemized):
    return " ".join(word for word, _ in phonemized.words)


def parse_mfa_json(path):
    """Return (words, phones) as lists of (start, end, label) from MFA json output."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    tiers = data["tiers"]

    def entries(name):
        for key, tier in tiers.items():
            if key.lower().endswith(name):
                return [(float(s), float(e), str(l)) for s, e, l in tier["entries"]]
        raise DropClip(f"mfa_missing_{name}_tier")

    return entries("words"), entries("phones")


SILENCE_LABELS = {"", "sil", "sp", "<eps>", "silence"}


def frame_energy_db(pcm16):
    """Per-frame energy (dB re. full scale) on the feature grid: frame k = samples [160k-80, 160k+80)."""
    x = np.pad(np.asarray(pcm16, dtype=np.float32) / 32768.0, (HOP // 2, 0))
    n = len(pcm16) // HOP
    frames = x[: n * HOP].reshape(n, HOP)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def durations_from_alignment(phonemized, phones, num_audio_frames, margin=3,
                             energy_db=None, energetic_rel_db=-35.0, max_extend=30):
    """Map MFA phone intervals onto the PiVo token sequence.

    Trimming follows PiVo's edge-corrected corpus: keep the spoken region, `margin`
    frames on each side, and any *energetic* frames touching the edges (e.g. a
    breath-free plosive release the aligner left outside the last phone), cutting
    only at whole-frame boundaries.

    Returns (start_frame, end_frame, durations) where the clip should be cut to
    frames [start_frame, end_frame) and `durations` has one entry per token.
    """
    tokens = list(phonemized.symbols)
    speech = [(s, e, l) for s, e, l in phones if l not in SILENCE_LABELS]
    if any(l == "spn" for _, _, l in speech):
        raise DropClip("mfa_spoken_noise")
    expected = [mfa_phone(p) for _, ph in phonemized.words for p in ph]
    if [l for _, _, l in speech] != expected:
        raise DropClip("mfa_phone_mismatch")

    to_f = lambda t: int(round(t * SR / HOP))
    first, last = to_f(speech[0][0]), to_f(speech[-1][1])
    start = max(0, first - margin)
    end = min(num_audio_frames, last + margin)
    if energy_db is not None and len(energy_db):
        threshold = float(np.max(energy_db)) + energetic_rel_db
        lo = max(0, start - max_extend)
        while start > lo and energy_db[start - 1] > threshold:
            start -= 1
        hi = min(num_audio_frames, len(energy_db), end + max_extend)
        while end < hi and energy_db[end] > threshold:
            end += 1
    n = end - start
    if n <= 0:
        raise DropClip("empty_after_trim")

    phone_token_idx = [i for i, t in enumerate(tokens) if t not in PIVO_SPECIAL]
    assert len(phone_token_idx) == len(speech)
    dur = [0] * len(tokens)
    bounds = [(to_f(s) - start, to_f(e) - start) for s, e, _ in speech]
    bounds[0] = (0, bounds[0][1])                     # leading margin -> first phone

    for k, ti in enumerate(phone_token_idx):
        b0, b1 = bounds[k]
        dur[ti] += b1 - b0
        if k + 1 < len(phone_token_idx):
            gap = bounds[k + 1][0] - b1               # silence (or rounding) before next phone
            between = tokens[ti + 1: phone_token_idx[k + 1]]
            if not between:
                dur[ti] += gap                        # inside a word: keep with the phone
            else:
                punct = [j for j in range(ti + 1, phone_token_idx[k + 1]) if tokens[j] not in ("<space>",)]
                target = punct[0] if punct else ti + between.index("<space>") + 1
                dur[target] += gap
    # trailing silence + margin -> final punctuation if present, else last phone
    tail = n - bounds[-1][1]
    last_ti = phone_token_idx[-1]
    after = [j for j in range(last_ti + 1, len(tokens)) if tokens[j] not in ("<eos>", "<space>")]
    dur[after[0] if after else last_ti] += tail

    if any(d < 0 for d in dur):
        raise DropClip("negative_duration")
    # every phone needs >= 1 frame: borrow from the longer neighbouring phone
    for k, ti in enumerate(phone_token_idx):
        if dur[ti] == 0:
            nbrs = [phone_token_idx[j] for j in (k - 1, k + 1) if 0 <= j < len(phone_token_idx)]
            donor = max(nbrs, key=lambda x: dur[x], default=None)
            if donor is None or dur[donor] < 2:
                raise DropClip("zero_duration_phone")
            dur[donor] -= 1
            dur[ti] = 1
    if sum(dur) != n:
        raise DropClip("duration_sum_mismatch")
    if max(dur) > MAX_TOKEN_FRAMES:
        raise DropClip("token_longer_than_2s")
    internal = [dur[i] for i in range(phone_token_idx[0], last_ti) if tokens[i] in PIVO_SPECIAL]
    if internal and max(internal) > 100:
        raise DropClip("internal_pause_over_1s")
    return start, end, dur


# --------------------------------------------------------------------------- features
def extract_features(pcm16, fargan_demo):
    """Run Opus `fargan_demo -features` and return (frames, 20) float32.

    The per-frame record width depends on the Opus revision: the 1.5.x releases
    write 36 floats (20 features + 16 LPC), newer master (incl. the revision PiVo
    pinned) writes just the 20 features. Both give one record per full 160-sample
    frame, so the width is recovered from the expected frame count.
    """
    frames = len(pcm16) // HOP
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.pcm"), os.path.join(tmp, "out.f32")
        np.asarray(pcm16, dtype="<i2").tofile(src)
        subprocess.run([fargan_demo, "-features", src, dst], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        feats = np.fromfile(dst, dtype="<f4")
    width = None
    for w in (FEATURE_DIM, OPUS_TOTAL_FEATURES):
        if frames and feats.size == frames * w:
            width = w
    if width is None:
        global _WARNED_SIZE
        if not _WARNED_SIZE:
            _WARNED_SIZE = True
            print(f"[features] unexpected size: {feats.size} floats for {frames} frames "
                  f"({feats.size / max(frames, 1):.2f} per frame)", flush=True)
        raise DropClip("feature_file_size")
    feats = feats.reshape(frames, width)[:, :FEATURE_DIM].copy()
    if not np.isfinite(feats).all():
        raise DropClip("non_finite_features")
    return feats


def pitch_periods(features):
    return np.clip(np.rint(256 / np.power(2.0, np.clip(features[..., 18], -10, 10) + 1.5)), 32, 255)
