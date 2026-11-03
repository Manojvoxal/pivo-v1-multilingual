"""Pipeline stages (adapted from the Kannada pipeline pivo_kn_v3/stages.py).

Every stage reads/writes plain files under one root folder, so it can be re-run or
resumed independently on Modal. Layout under ROOT:
  select/<lang>.jsonl                      Rasa rows chosen by the scan stage
  corpus/<lang>/<speaker>/<utt>.wav|.lab   16 kHz loudness-normalised audio + MFA word transcript
  meta/<speaker>__<tag>.jsonl              one record per kept utterance (text, tokens, words)
  dict/<lang>.dict                         MFA pronunciation dictionary (word<TAB>phones)
  align/<lang>/<speaker>/<utt>.json        MFA alignments
  data/<speaker>/wavs/<utt>.wav            trimmed audio
  data/<speaker>/feats/<utt>.npy           (frames, 20) float32 FARGAN features
  manifests/{train,val,test}.jsonl         training manifests (see DATA_FORMAT.md)
  reports/*.json(l)                        scan, drop reasons, statistics
"""

import collections
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

from . import prep
from .config import DEFAULT_STYLES, OVERCOLLECT, style_allowed
from .frontend import FrontendError, drop_reason, phonemize
from .langs import LANG, LANG_CODES, LANG_TO_ID
from .scripts import script_of_word
from .symbols import (ENGLISH_ID_RANGE, INDIC_ID_RANGE, NUM_SPECIAL, SYMBOL_TO_ID, SYMBOLS,
                      VOCAB_SIZE, mfa_phone)


def _jsonl_write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _jsonl_read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _json_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------- stage 0 (Rasa)
def select_rasa(meta_rows, lang, hours_per_voice, styles=None, min_s=1.5, max_s=16.0):
    """Choose Rasa rows for one language from metadata only.

    Keeps rows with an allowed style, a plausible duration and text our frontend can
    read (strict: no digits, no unknown characters, English words in the CMU lexicon),
    then takes a deterministic pseudo-random subset per gender up to
    hours_per_voice * OVERCOLLECT. Returns (selected_rows, report)."""
    explicit = None if styles is None else {s.strip().lower() for s in styles}
    by_style = collections.defaultdict(lambda: [0, 0.0])
    scripts = collections.Counter()
    drops = collections.Counter()
    cands = collections.defaultdict(list)
    genders_raw = collections.Counter()
    for r in meta_rows:
        g, dur = r["gender"], r["duration"]
        genders_raw[str(r.get("gender_raw"))] += 1
        key = f"{g or '?'}|{r['style'] or '<none>'}"
        by_style[key][0] += 1
        by_style[key][1] += (dur or 0.0) / 3600
        for w in r["text"].split()[:3]:
            w = w.strip(".,?!;:'\"()[]-–—।॥|…")
            if w:
                scripts[script_of_word(w) or "other"] += 1
        if g is None:
            drops["unknown_gender"] += 1
            continue
        if not style_allowed(r["style"], explicit):
            drops["style_excluded"] += 1
            continue
        if dur is None or not (min_s <= dur <= max_s):
            drops["duration"] += 1
            continue
        try:
            phonemize(r["text"], lang)
        except FrontendError as exc:
            drops["text_" + drop_reason(exc)] += 1
            continue
        cands[g].append(r)
    selected, chosen = [], {}
    for g, rows in sorted(cands.items()):
        rows.sort(key=lambda r: hashlib.sha1(f"{r['shard']}|{r['filename']}".encode()).hexdigest())
        total, cap = 0.0, hours_per_voice * OVERCOLLECT * 3600
        for r in rows:
            if total >= cap:
                break
            selected.append(r)
            total += r["duration"]
        chosen[g] = {"clips": sum(1 for r in selected if r["gender"] == g),
                     "hours": round(total / 3600, 2),
                     "eligible_hours": round(sum(r["duration"] for r in rows) / 3600, 2)}
    report = {
        "lang": lang, "rows": len(meta_rows),
        "hours_by_gender_style": {k: {"clips": v[0], "hours": round(v[1], 2)}
                                  for k, v in sorted(by_style.items())},
        "gender_values": dict(genders_raw),
        "script_of_first_words": dict(scripts.most_common()),
        "dropped": dict(drops), "selected": chosen,
        "style_rule": sorted(explicit) if explicit is not None else "default " + ",".join(DEFAULT_STYLES),
    }
    return selected, report


# --------------------------------------------------------------------------- stage 1
def prepare_speaker(items, root, speaker, lang, target_hours, min_s=1.5, max_s=15.0, tag=None):
    """items: iterable of dicts {utt_id, audio (float array), sr, text, extra}.

    Phonemizes, resamples, normalizes and writes the MFA corpus for one speaker,
    stopping once `target_hours` of audio have been kept."""
    drops = collections.Counter()
    meta, kept_seconds = [], 0.0
    corpus_dir = os.path.join(root, "corpus", lang, speaker)
    os.makedirs(corpus_dir, exist_ok=True)
    for item in items:
        if kept_seconds >= target_hours * 3600:
            break
        utt = item["utt_id"]
        try:
            ph = phonemize(item["text"], lang)
            audio = prep.to_16k_mono(item["audio"], item["sr"])
            seconds = len(audio) / prep.SR
            if not (min_s <= seconds <= max_s + 1.0):  # +1 s slack: edges get trimmed later
                raise prep.DropClip("raw_length_out_of_range")
            audio = prep.check_and_normalize(audio)
        except FrontendError as exc:
            drops["frontend_" + drop_reason(exc)] += 1
            continue
        except prep.DropClip as exc:
            drops[str(exc)] += 1
            continue
        except Exception as exc:  # undecodable audio etc.: never kill the shard
            drops["error_" + type(exc).__name__] += 1
            continue
        prep.write_wav(os.path.join(corpus_dir, utt + ".wav"), prep.to_pcm16(audio))
        with open(os.path.join(corpus_dir, utt + ".lab"), "w", encoding="utf-8") as fh:
            fh.write(prep.mfa_lab_text(ph))
        meta.append({
            "id": utt, "speaker": speaker, "lang": lang, "text": item["text"],
            "normalized_text": ph.normalized_text, "symbols": list(ph.symbols),
            "words": [[w, list(p)] for w, p in ph.words], "raw_seconds": round(seconds, 3),
            **({"extra": item["extra"]} if item.get("extra") else {}),
        })
        kept_seconds += seconds
    name = speaker if tag is None else f"{speaker}__{tag}"
    _jsonl_write(os.path.join(root, "meta", name + ".jsonl"), meta)
    report = {"speaker": speaker, "shard": tag, "lang": lang, "kept": len(meta),
              "kept_hours": round(kept_seconds / 3600, 3), "dropped": dict(drops)}
    _jsonl_write(os.path.join(root, "reports", "prepare", f"{name}.jsonl"), [report])
    return report


def speaker_meta(root, speaker):
    """All metadata rows of a speaker (unsharded file and/or <speaker>__<tag> files)."""
    rows, d = [], os.path.join(root, "meta")
    if not os.path.isdir(d):
        return rows
    for name in sorted(os.listdir(d)):
        if name == speaker + ".jsonl" or name.startswith(speaker + "__"):
            rows.extend(_jsonl_read(os.path.join(d, name)))
    return rows


def lang_meta(root, lang, speakers):
    rows = []
    for spk in speakers:
        rows.extend(r for r in speaker_meta(root, spk) if r["lang"] == lang)
    return rows


def write_dictionary(root, lang, rows):
    """Collect every (word, phones) pair used by this language's utterances."""
    entries = {}
    for row in rows:
        for word, phones in row["words"]:
            prev = entries.setdefault(word, phones)
            if prev != phones:
                raise RuntimeError(f"inconsistent pronunciation for {word!r}: {prev} vs {phones}")
    path = os.path.join(root, "dict", f"{lang}.dict")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for word in sorted(entries):
            fh.write(f"{word}\t{' '.join(mfa_phone(p) for p in entries[word])}\n")
    return path, len(entries)


# --------------------------------------------------------------------------- stage 2
def run_mfa(root, lang, mode, mfa_bin="mfa", num_jobs=8, temp_dir=None):
    """mode='train': train an aligner on this corpus and align it.
    mode='pretrained:<model>': align with a given MFA acoustic model."""
    corpus = os.path.join(root, "corpus", lang)
    dictionary = os.path.join(root, "dict", f"{lang}.dict")
    out = os.path.join(root, "align", lang)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    common = ["--output_format", "json", "-j", str(num_jobs), "--clean",
              "--no_use_postgres", "--beam", "100", "--retry_beam", "400"]
    if temp_dir:
        common += ["--temporary_directory", temp_dir]
    if mode == "train":
        model_out = os.path.join(root, "align", f"{lang}_acoustic_model.zip")
        cmd = [mfa_bin, "train", corpus, dictionary, model_out, "--output_directory", out] + common
    elif mode.startswith("pretrained:"):
        cmd = [mfa_bin, "align", corpus, dictionary, mode.split(":", 1)[1], out] + common
    else:
        raise ValueError(mode)
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def copy_corpus_subset(root, lang, utts, work):
    """Copy (speaker, utt) wav+lab pairs into work/corpus/<lang>/<speaker>/ for MFA."""
    for spk, utt in utts:
        src = os.path.join(root, "corpus", lang, spk)
        dst = os.path.join(work, "corpus", lang, spk)
        os.makedirs(dst, exist_ok=True)
        for ext in (".wav", ".lab"):
            shutil.copy(os.path.join(src, utt + ext), os.path.join(dst, utt + ext))


def pick_training_subset(rows, hours_per_speaker, seed=0):
    """Random subset (balanced per speaker) used to train an aligner quickly."""
    import random
    by_spk = collections.defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    picked = []
    for spk, rs in sorted(by_spk.items()):
        random.Random(seed).shuffle(rs)
        total = 0.0
        for r in rs:
            if total >= hours_per_speaker * 3600:
                break
            picked.append((spk, r["id"]))
            total += r["raw_seconds"]
    return picked


# --------------------------------------------------------------------------- stage 3
def _split_of(text, val_frac, test_frac):
    h = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def build_speaker(root, speaker, voice_id, fargan_demo, meta, tag, val_frac=0.015,
                  test_frac=0.015, min_s=1.5, max_s=15.0, margin=3):
    """Build training rows for one shard of a speaker; finalize_speaker merges shards."""
    drops = collections.Counter()
    rows = []
    for m in meta:
        lang, utt = m["lang"], m["id"]
        try:
            ali = os.path.join(root, "align", lang, speaker, utt + ".json")
            if not os.path.exists(ali):
                raise prep.DropClip("mfa_not_aligned")
            _, phones = prep.parse_mfa_json(ali)
            pcm = prep.read_wav(os.path.join(root, "corpus", lang, speaker, utt + ".wav"))

            class _Ph:  # lightweight view with the fields durations_from_alignment needs
                symbols = tuple(m["symbols"])
                words = tuple((w, tuple(p)) for w, p in m["words"])

            start, end, dur = prep.durations_from_alignment(
                _Ph, phones, len(pcm) // prep.HOP, margin=margin,
                energy_db=prep.frame_energy_db(pcm))
            clip = pcm[start * prep.HOP: end * prep.HOP]
            seconds = len(clip) / prep.SR
            if not (min_s <= seconds <= max_s):
                raise prep.DropClip("trimmed_length_out_of_range")
            feats = prep.extract_features(clip, fargan_demo)
            if len(feats) != end - start:
                raise prep.DropClip("feature_frame_mismatch")
        except prep.DropClip as exc:
            drops[str(exc)] += 1
            continue
        except Exception as exc:
            drops["error_" + type(exc).__name__] += 1
            continue
        wav_rel = f"data/{speaker}/wavs/{utt}.wav"
        feat_rel = f"data/{speaker}/feats/{utt}.npy"
        prep.write_wav(os.path.join(root, wav_rel), clip)
        os.makedirs(os.path.dirname(os.path.join(root, feat_rel)), exist_ok=True)
        np.save(os.path.join(root, feat_rel), feats.astype(np.float32))
        n_phones = sum(1 for s in m["symbols"] if SYMBOL_TO_ID[s] >= NUM_SPECIAL)
        rows.append({
            "id": utt, "speaker": speaker, "voice_id": voice_id,
            "lang": lang, "lang_id": LANG_TO_ID[lang],
            "text": m["text"], "normalized_text": m["normalized_text"],
            "symbols": m["symbols"],
            "token_ids": [SYMBOL_TO_ID[s] for s in m["symbols"]],
            "durations": dur, "n_frames": int(end - start),
            "seconds": round(seconds, 3), "phones_per_second": round(n_phones / seconds, 3),
            "wav": wav_rel, "features": feat_rel,
            "split": _split_of(m["normalized_text"], val_frac, test_frac),
            **({"style": m["extra"]["style"]} if m.get("extra", {}).get("style") else {}),
        })
    _jsonl_write(os.path.join(root, "manifests", "shards", f"{speaker}__{tag}.jsonl"), rows)
    _jsonl_write(os.path.join(root, "reports", "build_shards", f"{speaker}__{tag}.jsonl"),
                 [{"speaker": speaker, "shard": tag, "input": len(meta), "kept": len(rows),
                   "dropped": dict(drops)}])
    return {"speaker": speaker, "shard": tag, "kept": len(rows), "dropped": dict(drops)}


def finalize_speaker(root, speaker, voice_id, max_hours=None):
    """Merge shard rows, drop speaking-rate outliers (beyond 3 sigma: bad alignment or
    misread transcript), cap total hours, write the per-speaker manifest."""
    drops = collections.Counter()
    rows, n_input = [], 0
    d = os.path.join(root, "manifests", "shards")
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.startswith(speaker + "__"):
            rows.extend(_jsonl_read(os.path.join(d, name)))
    rd = os.path.join(root, "reports", "build_shards")
    for name in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
        if name.startswith(speaker + "__"):
            rep = _jsonl_read(os.path.join(rd, name))[0]
            n_input += rep["input"]
            drops.update(rep["dropped"])
    rows.sort(key=lambda r: r["id"])

    def _delete(r):
        for rel in (r["wav"], r["features"]):
            try:
                os.remove(os.path.join(root, rel))
            except FileNotFoundError:
                pass

    if len(rows) > 50:
        rates = np.array([r["phones_per_second"] for r in rows])
        mu, sd = rates.mean(), rates.std()
        keep = [abs(r["phones_per_second"] - mu) <= 3 * sd for r in rows]
        for r, k in zip(rows, keep):
            if not k:
                drops["speaking_rate_outlier"] += 1
                _delete(r)
        rows = [r for r, k in zip(rows, keep) if k]
    if max_hours is not None:
        import random
        random.Random(0).shuffle(rows)
        kept, total = [], 0.0
        for r in rows:
            if total < max_hours * 3600:
                kept.append(r)
                total += r["seconds"]
            else:
                drops["over_hour_cap"] += 1
                _delete(r)
        rows = sorted(kept, key=lambda r: r["id"])
    _jsonl_write(os.path.join(root, "manifests", "per_speaker", speaker + ".jsonl"), rows)
    report = {
        "speaker": speaker, "voice_id": voice_id, "input": n_input, "kept": len(rows),
        "hours": round(sum(r["seconds"] for r in rows) / 3600, 3),
        "splits": dict(collections.Counter(r["split"] for r in rows)),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2) if rows else 0,
        "dropped": dict(drops),
    }
    _jsonl_write(os.path.join(root, "reports", f"build_{speaker}.jsonl"), [report])
    return report


def merge_manifests(root, speakers, all_speakers):
    """Write manifests/{train,val,test}.jsonl, symbols.json, languages.json, voices.json."""
    by_split = collections.defaultdict(list)
    for spk in speakers:
        path = os.path.join(root, "manifests", "per_speaker", spk + ".jsonl")
        if os.path.exists(path):
            for r in _jsonl_read(path):
                by_split[r["split"]].append(r)
    for split, rows in by_split.items():
        rows.sort(key=lambda r: (r["voice_id"], r["id"]))
        _jsonl_write(os.path.join(root, "manifests", f"{split}.jsonl"), rows)
    _json_write(os.path.join(root, "manifests", "symbols.json"), {
        "symbols": list(SYMBOLS), "vocab_size": VOCAB_SIZE, "num_special": NUM_SPECIAL,
        "english_ids": list(ENGLISH_ID_RANGE), "indic_ids": list(INDIC_ID_RANGE),
        "mfa_names": {s: mfa_phone(s) for s in SYMBOLS[NUM_SPECIAL:]},
    })
    _json_write(os.path.join(root, "manifests", "languages.json"), [
        {"lang_id": LANG_TO_ID[c], "code": c, "name": LANG[c]["name"] or "English",
         "script": LANG[c]["script"], "phase": LANG[c]["phase"]} for c in LANG_CODES])
    _json_write(os.path.join(root, "manifests", "voices.json"), [
        {"voice_id": s["voice_id"], "speaker": s["name"], "lang": s["lang"], "gender": s["gender"],
         "source": s["source"], "prepared": s["name"] in speakers} for s in all_speakers])
    return {k: len(v) for k, v in by_split.items()}


def feature_stats(root, split="train", max_files=4000):
    """Per-dim mean/std over a sample balanced across voices (the manifest is sorted
    by voice, so taking the first N rows would see only a few voices)."""
    path = os.path.join(root, "manifests", f"{split}.jsonl")
    if not os.path.exists(path):
        raise RuntimeError(f"no {split} manifest: every clip was dropped; see reports/build_*.jsonl")
    rows = _jsonl_read(path)
    by_voice = collections.defaultdict(list)
    for r in rows:
        by_voice[r["voice_id"]].append(r)
    per_voice = max(20, max_files // max(1, len(by_voice)))
    import random
    picked = []
    for v, rs in sorted(by_voice.items()):
        random.Random(v).shuffle(rs)
        picked.extend(rs[:per_voice])
    allf = np.concatenate([np.load(os.path.join(root, r["features"])) for r in picked])
    stats = {"mean": allf.mean(0).tolist(), "std": allf.std(0).tolist(), "frames": int(len(allf)),
             "files": len(picked), "voices": len(by_voice)}
    _json_write(os.path.join(root, "manifests", "feature_stats.json"), stats)
    return stats
