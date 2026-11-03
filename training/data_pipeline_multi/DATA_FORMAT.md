# PiVo multilingual training data: format spec

Written by `prepare_pivo_multi.py` to the Modal volume **`pivo-multi-data`** (Volumes v2),
folder **`/pivo_multi_v1`** (smoke runs: `/smoke`). Same row format as the Kannada + English
dataset (`pivo-v1/training/data_pipeline/DATA_FORMAT.md`) plus `lang_id`, a new token
table, 23 languages and 46 voices.

## Folder layout

```
pivo_multi_v1/
  manifests/
    train.jsonl  val.jsonl  test.jsonl   one JSON object per utterance (fields below)
    symbols.json                         token table (157 symbols, vocab 160) + MFA phone names
    languages.json                       23 languages: lang_id, code, name, script, phase
    voices.json                          46 voices: voice_id, speaker, lang, gender, source, prepared
    feature_stats.json                   per-dim mean/std of the 20 features (balanced over voices)
    summary.json                         hours per language, drop reasons, Opus build
    per_speaker/<speaker>.jsonl          the same rows, split by speaker
  data/<speaker>/wavs/<id>.wav           16 kHz mono PCM16, edge-trimmed
  data/<speaker>/feats/<id>.npy          float32 [n_frames, 20]
  select/<lang>.jsonl                    Rasa rows picked by the scan stage
  reports/scan/<lang>.json               Rasa styles / genders / scripts / hours per language
  corpus/ meta/ dict/ align/ reports/    intermediate files (training ignores them)
```

## Languages (`languages.json`, ids are fixed)

| id | code | language | script | phase | source |
|---|---|---|---|---|---|
| 0 | en | English (Indian) | Latin | 1 | IndicTTS English (IIT Madras); Hi-Fi TTS fallback |
| 1 | kn | Kannada | Kannada | 1 | SYSPIN |
| 2 | hi | Hindi | Devanagari | 1 | Rasa |
| 3 | te | Telugu | Telugu | 1 | Rasa |
| 4 | ta | Tamil | Tamil | 1 | Rasa |
| 5 | bn | Bengali | Bengali | 1 | Rasa |
| 6 | mr | Marathi | Devanagari | 2 | Rasa |
| 7 | ne | Nepali | Devanagari | 2 | Rasa |
| 8 | kok | Konkani | Devanagari | 2 | Rasa |
| 9 | mai | Maithili | Devanagari | 2 | Rasa |
| 10 | doi | Dogri | Devanagari | 2 | Rasa |
| 11 | brx | Bodo | Devanagari | 2 | Rasa |
| 12 | sa | Sanskrit | Devanagari | 2 | Rasa |
| 13 | as | Assamese | Bengali-Assamese | 2 | Rasa |
| 14 | gu | Gujarati | Gujarati | 2 | Rasa |
| 15 | pa | Punjabi | Gurmukhi | 2 | Rasa |
| 16 | or | Odia | Odia | 2 | Rasa |
| 17 | ml | Malayalam | Malayalam | 2 | Rasa |
| 18 | mni | Manipuri | Meetei Mayek (or Bengali) | 2 | Rasa |
| 19 | sat | Santali | Ol Chiki (or Devanagari) | 2 | Rasa |
| 20 | ur | Urdu | Perso-Arabic | 3 | Rasa (G2P not written yet) |
| 21 | ks | Kashmiri | Perso-Arabic | 3 | Rasa (G2P not written yet) |
| 22 | sd | Sindhi | Perso-Arabic | 3 | Rasa (G2P not written yet) |

## Voices (`voices.json`, ids are fixed)

`voice_id = 2 * lang_id + (0 for female, 1 for male)`. Speaker names:
`en_iitm_tamil_f/m` (or `en_hifi92_f` / `en_hifi6097_m` with `--english hifi`),
`kn_syspin_f/m`, `<code>_rasa_f/m` for the rest. 46 voices; ids of languages not
prepared yet are reserved, so adding phase 3 later never renumbers anything.

## Manifest row

```json
{
  "id": "rasa_hi_f_Hindi_F_BOOK_00012",
  "speaker": "hi_rasa_f", "voice_id": 4, "lang": "hi", "lang_id": 2,
  "text": "original transcript", "normalized_text": "frontend-normalised text",
  "symbols":   ["<bos>", "in_bh", "in_aa", "in_r", "in_a", "in_t", "<space>", "...", ".", "<eos>"],
  "token_ids": [1, 131, 80, 134, 79, 123, 3, "...", 4, 2],
  "durations": [0, 6, 9, 4, 5, 7, 0, ..., 3, 0],
  "n_frames": 412, "seconds": 4.12, "phones_per_second": 11.4,
  "wav": "data/hi_rasa_f/wavs/<id>.wav",
  "features": "data/hi_rasa_f/feats/<id>.npy",
  "split": "train",
  "style": "BOOK"
}
```

## Tokens (`symbols.json`)

- **0–9** PiVo specials `<pad> <bos> <eos> <space> . , ? ! ; :` (same ids as PiVo).
- **10–78** ARPAbet with stress, for English sentences and English words inside Indic text.
- **79–156** 78 shared Indic phones, prefix `in_`: 15 vowels (incl. `ae` æ, `aw` ɔ, `ix` ɨ),
  14 nasal vowels (`aa~` …), 49 consonants (the Kannada set + `q x gx rr rrh zh rx tx nx`
  + Sindhi implosives `bq dq jq gq`). The same sound has the same token in every language;
  the language embedding carries the accent.
- vocab_size 160 (3 spare rows).
- MFA lexicons write phones without `_` and with `~` → `N` (`in_aa~` → `inaaN`).

## Text frontend (rule-based, `pivo_multi/`)

- `brahmic.py`: one parser for all nine Brahmic scripts (shared ISCII offsets) + per-language
  rules: inherent vowel (/a/ or /ɔ/), schwa deletion (none / final / final + VC_CV),
  anusvara (homorganic nasal, nasal vowel, /m/ or /ŋ/), Tamil positional voicing,
  Malayalam chillus / samvruthokaram / alveolar ന, Bengali-Assamese ya/ba-phala, sibilants.
- `alphabetic.py`: Ol Chiki (Santali) and Meetei Mayek (Manipuri).
- English: PiVo's normaliser + CMU lexicon (unchanged).
- Training data is strict: clips with digits, unsupported characters or English words
  missing from CMU are dropped (numbers will be expanded by the runtime normaliser later).

## Durations, features, filters

Identical to the Kannada + English dataset:
`len(durations) == len(token_ids)`, `sum(durations) == n_frames == len(features)`,
`<bos>/<eos>` = 0, every phone ≥ 1 frame, no token > 200 frames, pauses on `<space>` /
punctuation, 3-frame edge margin; Opus `fargan_demo -features` (first 20 floats per 10 ms
frame, schema `opus-fargan-20-v1`); clips 1.5–15 s, −20 LUFS; MFA 3.4.3 must reproduce the
exact phone sequence, internal pauses < 1 s, speaking-rate outliers (3σ per speaker)
dropped; deterministic 1.5 % val / 1.5 % test split on the normalised-text hash.

## What the trainer has to change (new fork)

1. `CONFIG`: `vocab_size=160`, `num_voices=46`; add `num_langs=23`.
2. Model: add `self.langs = nn.Embedding(num_langs, width)` and add it next to the voice
   embedding in `encode()` (`x = tokens + voices + langs`). Cost: 23 × width params.
3. Data: `PiVoDataset.__getitem__` returns `r["lang_id"]`; `collate` stacks it.
4. Train from scratch: build `PiVoAcoustic(config)` directly, no `load_pivo_checkpoint`
   (English and Indic share nothing with PiVo's old token rows anymore).
5. Data root `/data/pivo_multi_v1` on volume `pivo-multi-data`.
6. Recommended: sample batches balanced by voice (or language) so small voices are not
   drowned out; try width 96 and 128.
7. The feature cache (`pack.py`) for ~550 h is ~16 GB; the training container has 64 GB.
