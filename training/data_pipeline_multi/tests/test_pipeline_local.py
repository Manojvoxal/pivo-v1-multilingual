"""Offline end-to-end test of the data pipeline (no Modal, no network, no MFA, no Opus).

Fake Rasa / SYSPIN / IndicTTS inputs go through select -> prepare -> (fake MFA
alignment) -> build (fake fargan_demo) -> finalize, and the resulting manifests are
checked against the rules the trainer relies on.

Run:  PIVO_CMUDICT=.../cmudict.dict python tests/test_pipeline_local.py
"""

import io
import json
import os
import random
import stat
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivo_multi import config, english, prep, sources, stages  # noqa: E402
from pivo_multi.langs import LANG_TO_ID  # noqa: E402
from pivo_multi.symbols import SYMBOL_TO_ID, VOCAB_SIZE, mfa_phone  # noqa: E402

HI = ["भारत एक विशाल देश है।", "मैं हर दिन किताब पढ़ता हूँ।", "आज मौसम बहुत अच्छा है।",
      "हमें समय पर स्कूल जाना चाहिए।", "उसने मुझे एक सुंदर कहानी सुनाई।", "यह सड़क शहर की ओर जाती है।",
      "बच्चे मैदान में खेल रहे हैं।", "किसान खेत में काम करता है।", "Doctor ने दवाई दी।",
      "इस साल 2026 में चुनाव है।"]
TA = ["தமிழ் ஒரு பழமையான மொழி.", "நான் தினமும் பள்ளிக்குச் செல்கிறேன்.", "அவன் வீட்டில் இருக்கிறான்.",
      "இன்று வானிலை நன்றாக உள்ளது.", "மழை பெய்தது."]
KN = ["ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ.", "ಇದು ಒಂದು ಸುಂದರ ದಿನ.", "ನಮ್ಮ ಮನೆ ಹತ್ತಿರ ಶಾಲೆ ಇದೆ."]
EN = ["The weather is pleasant today.", "Please close the door.", "India is a large country."]


def fake_audio(sec, sr, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    return (0.08 * env * rng.standard_normal(len(t))).astype(np.float32)


def wav_bytes(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_rasa_parquet(path, texts, lang, seed):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    rows = []
    for i in range(60):
        text = texts[i % len(texts)]
        style = rng.choice(["BOOK", "NEWS", "HAPPY", "CONV", "WIKI"])
        g = "Female" if i % 2 == 0 else "Male"
        dur = rng.uniform(2.0, 5.0)
        rows.append({"filename": f"{lang}_{g}_{style}_{seed}_{i:04d}.wav", "text": text, "language": lang,
                     "gender": g, "style": style, "duration": f"{dur:.2f}", "wav_path": f"x/{i}.wav",
                     "audio": {"bytes": wav_bytes(fake_audio(dur, 48000, seed * 1000 + i), 48000),
                               "path": f"{i}.wav"}})
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=16)


def make_fargan(tmp):
    """Executable that mimics `fargan_demo -features in.pcm out.f32` (20 floats per frame)."""
    path = os.path.join(tmp, "fake_fargan")
    with open(path, "w") as fh:
        fh.write(f"""#!{sys.executable}
import sys, os, numpy as np
src, dst = sys.argv[2], sys.argv[3]
n = os.path.getsize(src) // 2 // 160
np.random.default_rng(n).standard_normal((n, 20)).astype('<f4').tofile(dst)
""")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def fake_align(root, meta):
    """Write MFA-style json: phones spread evenly over the clip (minus 0.2 s edges)."""
    for m in meta:
        pcm = prep.read_wav(os.path.join(root, "corpus", m["lang"], m["speaker"], m["id"] + ".wav"))
        dur = len(pcm) / prep.SR
        phones = [mfa_phone(p) for _, ph in m["words"] for p in ph]
        t0, t1 = 0.2, dur - 0.2
        step = (t1 - t0) / len(phones)
        entries = [[round(t0 + k * step, 3), round(t0 + (k + 1) * step, 3), p] for k, p in enumerate(phones)]
        words = [[t0, t1, " ".join(w for w, _ in m["words"])]]
        out = os.path.join(root, "align", m["lang"], m["speaker"], m["id"] + ".json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"tiers": {"words": {"entries": words}, "phones": {"entries": entries}}}, fh)


def run(tmp):
    cmu = os.environ.get("PIVO_CMUDICT")
    assert cmu, "set PIVO_CMUDICT"
    english.set_cmudict_path(cmu)
    root = os.path.join(tmp, "out")
    fargan = make_fargan(tmp)
    spk_all = config.all_speakers("indictts")
    by_name = {s["name"]: s for s in spk_all}

    # ---------------- Rasa: hi + ta, two shards each, metadata scan + selection
    for lang, texts in (("hi", HI), ("ta", TA)):
        shards = []
        for k in range(2):
            p = os.path.join(tmp, f"{lang}_train-{k:05d}.parquet")
            make_rasa_parquet(p, texts, lang, seed=k + (10 if lang == "ta" else 0))
            shards.append(p)
        # rasa_read_meta works on HF paths; read the same columns locally instead
        import pyarrow.parquet as pq
        meta_rows = []
        for p in shards:
            pf = pq.ParquetFile(p)
            for i, r in enumerate(pf.read(columns=["filename", "text", "gender", "style", "duration"]).to_pylist()):
                meta_rows.append({"shard": p, "row": i, "filename": r["filename"], "text": r["text"],
                                  "gender": sources.gender_of(r["gender"]), "gender_raw": r["gender"],
                                  "style": r["style"], "duration": sources.parse_duration(r["duration"])})
        sel, rep = stages.select_rasa(meta_rows, lang, hours_per_voice=0.02)
        assert all(r["style"] in ("BOOK", "NEWS", "WIKI") for r in sel), rep
        assert rep["dropped"].get("style_excluded", 0) > 0
        if lang == "hi":
            assert rep["dropped"].get("text_digits", 0) > 0     # "2026" sentence dropped
        sel2, _ = stages.select_rasa(meta_rows, lang, 0.02, styles=["book"])
        assert sel2 and all(r["style"] == "BOOK" for r in sel2)
        for p in shards:
            wanted = {r["row"]: r["gender"] for r in sel if r["shard"] == p}
            for g in ("f", "m"):
                name = f"{lang}_rasa_{g}"
                items = (it for gg, it in sources.rasa_items(p, {k: v for k, v in wanted.items() if v == g}, lang)
                         if gg == g)
                stages.prepare_speaker(items, root, name, lang, 1e9, tag=f"{lang}_{os.path.basename(p)[:-8]}")
        print(f"{lang}: selected {rep['selected']}")

    # ---------------- SYSPIN-like Kannada parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    sysdir = os.path.join(tmp, "syspin")
    os.makedirs(sysdir)
    rows = []
    for i in range(12):
        g = "Female" if i % 2 == 0 else "Male"
        rows.append({"path": {"bytes": wav_bytes(fake_audio(3.0, 48000, 500 + i), 48000), "path": None},
                     "transcript": KN[i % len(KN)], "domain": "news", "gender": g, "language": "Kannada"})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(sysdir, "train-00000.parquet"))
    for g, key in (("f", "Female"), ("m", "Male")):
        stages.prepare_speaker(sources.iter_syspin(sysdir, key), root, f"kn_syspin_{g}", "kn", 1.0, tag="000")

    # ---------------- IndicTTS English (festival txt.done.data)
    for g in ("f", "m"):
        d = os.path.join(tmp, f"indictts_{g}", "english")
        os.makedirs(os.path.join(d, "wav"))
        with open(os.path.join(d, "txt.done.data"), "w") as fh:
            for i, t in enumerate(EN * 2):
                uid = f"train_tamil{g}_{i:05d}"
                fh.write(f'( {uid} "{t}" )\n')
                import soundfile as sf
                sf.write(os.path.join(d, "wav", uid + ".wav"), fake_audio(3.0, 48000, 900 + i), 48000)
        pairs, st = sources.indictts_index(os.path.dirname(d))
        assert st["paired"] == len(EN) * 2, st
        stages.prepare_speaker(sources.iter_indictts(os.path.dirname(d), "tamil", g), root,
                               f"en_iitm_tamil_{g}", "en", 1.0, tag="000")

    # ---------------- dictionaries (as align_setup does) + fake MFA + build + finalize
    active = [s for s in spk_all if s["lang"] in ("en", "kn", "hi", "ta")]
    for lang in ("en", "kn", "hi", "ta"):
        names = [s["name"] for s in active if s["lang"] == lang]
        rows = stages.lang_meta(root, lang, names)
        path, n = stages.write_dictionary(root, lang, rows)
        print(f"{lang}: {len(rows)} utts, {n} words in lexicon; e.g. {open(path).readline().strip()}")
        fake_align(root, rows)
    for s in active:
        meta = stages.speaker_meta(root, s["name"])
        stages.build_speaker(root, s["name"], s["voice_id"], fargan, meta=meta, tag="000")
        stages.finalize_speaker(root, s["name"], s["voice_id"], max_hours=s["hours"])
    prepared = [s["name"] for s in active]
    counts = stages.merge_manifests(root, prepared, spk_all)
    stats = stages.feature_stats(root) if counts.get("train") else None
    print("splits:", counts, "| stats over", stats and stats["voices"], "voices")

    # ---------------- checks the trainer relies on
    n_rows = 0
    for split in counts:
        for line in open(os.path.join(root, "manifests", f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            n_rows += 1
            assert len(r["token_ids"]) == len(r["durations"]), r["id"]
            assert sum(r["durations"]) == r["n_frames"], r["id"]
            assert r["durations"][0] == 0 and r["durations"][-1] == 0
            assert all(0 <= t < VOCAB_SIZE for t in r["token_ids"])
            assert all(SYMBOL_TO_ID[s] == t for s, t in zip(r["symbols"], r["token_ids"]))
            feats = np.load(os.path.join(root, r["features"]))
            assert feats.shape == (r["n_frames"], 20)
            assert by_name[r["speaker"]]["voice_id"] == r["voice_id"]
            assert r["lang_id"] == LANG_TO_ID[r["lang"]] and r["lang"] == by_name[r["speaker"]]["lang"]
    voices = json.load(open(os.path.join(root, "manifests", "voices.json")))
    langs = json.load(open(os.path.join(root, "manifests", "languages.json")))
    sym = json.load(open(os.path.join(root, "manifests", "symbols.json")))
    assert len(voices) == 46 and len(langs) == 23 and sym["vocab_size"] == VOCAB_SIZE
    print(f"OK: {n_rows} manifest rows checked; {len(voices)} voices, {len(langs)} languages, "
          f"{len(sym['symbols'])} symbols")
    reps = {}
    for f in sorted(os.listdir(os.path.join(root, "reports", "prepare"))):
        r = json.loads(open(os.path.join(root, "reports", "prepare", f)).readline())
        reps[r["speaker"]] = reps.get(r["speaker"], 0) + r["kept"]
    print("prepared clips per speaker:", reps)
    return n_rows


def test_pipeline_local():
    if not os.environ.get("PIVO_CMUDICT"):
        import pytest
        pytest.skip("set PIVO_CMUDICT")
    with tempfile.TemporaryDirectory() as tmp:
        assert run(tmp) > 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run(tmp)


def test_speech_up_to_the_last_frame_keeps_the_clip():
    """Speech that runs to the end of the recording (no trailing silence, final punctuation):
    the last phone's end rounds past the last whole frame. The clip must be kept, with the
    overshoot taken from the last phone (this used to raise negative_duration)."""
    from pivo_multi import prep
    from pivo_multi.frontend import phonemize
    from pivo_multi.symbols import mfa_phone
    ph = phonemize("ನಮಸ್ಕಾರ, ನೀವು.", "kn")
    labels = [mfa_phone(p) for _, w in ph.words for p in w]
    phones, t = [(0.0, 0.2, "sil")], 0.2
    for label in labels:
        phones.append((round(t, 3), round(t + 0.05, 3), label))
        t += 0.05
    n_frames = int((t - 0.004) * 16000) // 160
    start, end, dur = prep.durations_from_alignment(ph, phones, n_frames, margin=3)
    assert sum(dur) == end - start
    assert min(d for d, s in zip(dur, ph.symbols) if s.startswith("in_")) >= 1
