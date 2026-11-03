"""Pipeline stages. Each stage reads/writes plain files under one root folder,
so it can be re-run or resumed independently on Modal.

Layout under ROOT:
  corpus/<lang>/<speaker>/<utt>.wav|.lab   16 kHz loudness-normalized audio + MFA word transcript
  meta/<speaker>.jsonl                     one record per kept utterance (text, tokens, words)
  dict/<lang>.dict                         MFA pronunciation dictionary (word<TAB>phones)
  align/<lang>/<speaker>/<utt>.json        MFA alignments
  data/<speaker>/wavs/<utt>.wav            trimmed audio used for training
  data/<speaker>/feats/<utt>.npy           (frames, 20) float32 FARGAN features
  manifests/{train,val,test}.jsonl         training manifests (see DATA_FORMAT.md)
  reports/*.json                           drop reasons and statistics
"""

import collections
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

from . import prep
from .frontend import FrontendError, phonemize
from .symbols import SYMBOLS, SYMBOL_TO_ID, PIVO_SYMBOLS, NUM_SPECIAL


def _jsonl_write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- stage 1
def prepare_speaker(items, root, speaker, lang, target_hours, min_s=1.5, max_s=15.0, tag=None):
    """items: iterable of dicts {utt_id, audio (float array), sr, text}.

    Phonemizes, resamples, normalizes and writes the MFA corpus for one speaker,
    stopping once `target_hours` of audio have been kept.
    """
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
        except FrontendError:
            drops["frontend"] += 1
            continue
        except prep.DropClip as exc:
            drops[str(exc)] += 1
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
    _jsonl_write(os.path.join(root, "reports", f"prepare_{name}.jsonl"), [report])
    return report


def speaker_meta(root, speaker):
    """All metadata rows of a speaker (unsharded file and/or <speaker>__<shard> files)."""
    rows, d = [], os.path.join(root, "meta")
    for name in sorted(os.listdir(d)):
        if name == speaker + ".jsonl" or name.startswith(speaker + "__"):
            rows.extend(_jsonl_read(os.path.join(d, name)))
    return rows


def write_dictionary(root, lang):
    """Collect every (word, phones) pair used by this language's speakers."""
    entries = {}
    for name in sorted(os.listdir(os.path.join(root, "meta"))):
        for row in _jsonl_read(os.path.join(root, "meta", name)):
            if row["lang"] != lang:
                continue
            for word, phones in row["words"]:
                prev = entries.setdefault(word, phones)
                if prev != phones:
                    raise RuntimeError(f"inconsistent pronunciation for {word!r}")
    path = os.path.join(root, "dict", f"{lang}.dict")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for word in sorted(entries):
            fh.write(f"{word}\t{' '.join(prep.mfa_phone(p) for p in entries[word])}\n")
    return path, len(entries)


# --------------------------------------------------------------------------- stage 2
def run_mfa(root, lang, mode, mfa_bin="mfa", num_jobs=8, temp_dir=None, run_as=None):
    """mode='train': train an aligner on this corpus and align it (used for Kannada).
    mode='pretrained:<model>': align with a downloaded MFA acoustic model."""
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
    if run_as:
        cmd = ["runuser", "-u", run_as, "--"] + cmd
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


def build_speaker(root, speaker, voice_id, fargan_demo, val_frac=0.015, test_frac=0.015,
                  min_s=1.5, max_s=15.0, margin=3, meta=None, tag=None):
    """Build training rows for a speaker (or one shard of it when meta/tag are given).

    Unsharded calls also run finalize_speaker (rate filter); sharded calls leave
    that to the caller once every shard is done.
    """
    drops = collections.Counter()
    rows = []
    if meta is None:
        meta = speaker_meta(root, speaker)
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
        wav_rel = f"data/{speaker}/wavs/{utt}.wav"
        feat_rel = f"data/{speaker}/feats/{utt}.npy"
        prep.write_wav(os.path.join(root, wav_rel), clip)
        os.makedirs(os.path.dirname(os.path.join(root, feat_rel)), exist_ok=True)
        np.save(os.path.join(root, feat_rel), feats.astype(np.float32))
        n_phones = sum(1 for s in m["symbols"] if s not in prep.PIVO_SPECIAL)
        rows.append({
            "id": utt, "speaker": speaker, "voice_id": voice_id, "lang": lang,
            "text": m["text"], "normalized_text": m["normalized_text"],
            "symbols": m["symbols"],
            "token_ids": [SYMBOL_TO_ID[s] for s in m["symbols"]],
            "durations": dur, "n_frames": int(end - start),
            "seconds": round(seconds, 3), "phones_per_second": round(n_phones / seconds, 3),
            "wav": wav_rel, "features": feat_rel,
            "split": _split_of(m["normalized_text"], val_frac, test_frac),
        })
    if tag is not None:
        _jsonl_write(os.path.join(root, "manifests", "shards", f"{speaker}__{tag}.jsonl"), rows)
        _jsonl_write(os.path.join(root, "reports", "build_shards", f"{speaker}__{tag}.jsonl"),
                     [{"speaker": speaker, "shard": tag, "input": len(meta), "kept": len(rows),
                       "dropped": dict(drops)}])
        return {"speaker": speaker, "shard": tag, "kept": len(rows), "dropped": dict(drops)}
    return finalize_speaker(root, speaker, voice_id, rows=rows, drops=drops, n_input=len(meta))


def finalize_speaker(root, speaker, voice_id, rows=None, drops=None, n_input=None, max_hours=None):
    """Merge shard rows, drop speaking-rate outliers (beyond 3 sigma: bad alignment or
    misread transcript), optionally cap total hours, write the per-speaker manifest."""
    drops = collections.Counter(drops or {})
    if rows is None:
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


def merge_manifests(root, speakers):
    by_split = collections.defaultdict(list)
    for spk in speakers:
        for r in _jsonl_read(os.path.join(root, "manifests", "per_speaker", spk + ".jsonl")):
            by_split[r["split"]].append(r)
    for split, rows in by_split.items():
        rows.sort(key=lambda r: (r["speaker"], r["id"]))
        _jsonl_write(os.path.join(root, "manifests", f"{split}.jsonl"), rows)
    with open(os.path.join(root, "manifests", "symbols.json"), "w", encoding="utf-8") as fh:
        json.dump({"symbols": list(SYMBOLS), "pivo_english_ids": [0, len(PIVO_SYMBOLS) - 1],
                   "kannada_ids": [len(PIVO_SYMBOLS), len(SYMBOLS) - 1],
                   "num_special": NUM_SPECIAL, "vocab_size": 128}, fh, ensure_ascii=False, indent=1)
    return {k: len(v) for k, v in by_split.items()}


def feature_stats(root, split="train", max_files=3000):
    path = os.path.join(root, "manifests", f"{split}.jsonl")
    if not os.path.exists(path):
        raise RuntimeError(f"no {split} manifest: every clip was dropped; see reports/build_*.jsonl")
    rows = _jsonl_read(path)[:max_files]
    acc = [np.load(os.path.join(root, r["features"])) for r in rows]
    allf = np.concatenate(acc)
    stats = {"mean": allf.mean(0).tolist(), "std": allf.std(0).tolist(), "frames": int(len(allf))}
    with open(os.path.join(root, "manifests", "feature_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    return stats
