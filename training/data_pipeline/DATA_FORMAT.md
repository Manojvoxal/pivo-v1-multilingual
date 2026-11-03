# PiVo Kannada + English training data: format spec

This is what `prepare_pivo_v4.py` writes to the Modal volume **`pivo-data`**,
folder **`/pivo_kn_en_v1`** (smoke runs go to `/smoke`). The training code should
read only what's listed here.

## Folder layout

```
pivo_kn_en_v1/
  manifests/
    train.jsonl  val.jsonl  test.jsonl   one JSON object per utterance (fields below)
    symbols.json                         token table (127 symbols, vocab 128)
    feature_stats.json                   per-dim mean/std of the 20 features (train subset)
    summary.json                         counts, drop reasons, voices, Opus version used
    per_speaker/<speaker>.jsonl          the same rows, split by speaker
  data/<speaker>/wavs/<id>.wav           16 kHz mono PCM16, already edge-trimmed
  data/<speaker>/feats/<id>.npy          float32 [n_frames, 20]
  align/                                 MFA outputs and the trained Kannada aligner (for reference)
  corpus/ meta/ dict/ reports/           intermediate files (can be ignored by training)
```

## Voices

| voice_id | speaker | language | source |
|---|---|---|---|
| 0 | `en_hifi92_f` | English | Hi-Fi TTS speaker 92 (F), clean |
| 1 | `en_hifi6097_m` | English | Hi-Fi TTS speaker 6097 (M), clean |
| 2 | `kn_syspin_f` | Kannada | SYSPIN Kannada female |
| 3 | `kn_syspin_m` | Kannada | SYSPIN Kannada male |

Voice ids 0 and 1 are deliberately English female and male, so the voice embedding
can be warm-started from PiVo's `Pi` (0, F) and `Vo` (1, M). Rows 2 and 3 are new.
The table grows from 2 to 4 entries, which adds 2 × 96 parameters.

## Manifest row

```json
{
  "id": "syspin_f_train-00012-of-00093_000123",
  "speaker": "kn_syspin_f", "voice_id": 2, "lang": "kn",
  "text": "original transcript", "normalized_text": "frontend-normalized text",
  "symbols":   ["<bos>", "kn_t", "kn_oo", "kn_ll", "kn_a", "<space>", "...", ".", "<eos>"],
  "token_ids": [1, 106, 88, 120, 79, 3, "...", 4, 2],
  "durations": [0, 4, 9, 6, 7, 0, ..., 3, 0],
  "n_frames": 412, "seconds": 4.12, "phones_per_second": 11.4,
  "wav": "data/kn_syspin_f/wavs/<id>.wav",
  "features": "data/kn_syspin_f/feats/<id>.npy",
  "split": "train"
}
```

Paths are relative to the dataset root.

## Tokens (`symbols.json`)

- **ids 0–78** are PiVo's original table, unchanged: 10 specials (`<pad> <bos> <eos> <space> . , ? ! ; :`), then the ARPAbet consonants, then stressed vowels. English rows of the pretrained 128×96 embedding can be loaded as is.
- **ids 79–126** are 48 new Kannada phones (`kn_a … kn_z`). They use unused rows of the same 128-entry table, so **the embedding shape does not change**.
- Token layout matches PiVo's frontend exactly: `<bos>`, then words separated by `<space>`, punctuation tokens where they occur, then `<eos>`. English rows are produced by PiVo's own `text.py` logic and CMU dictionary (a parity check runs on every row).
- The Kannada G2P is rule-based (`pivo_kn_v3/kannada.py`). It keeps the inherent vowel, places anusvara by the following consonant, maps visarga to /h/, and reads ಋ as /ru/.

## Durations

- `len(durations) == len(token_ids)` and `sum(durations) == n_frames == len(features)`.
- `<bos>` and `<eos>` are always 0. Every phone is ≥ 1 frame (PiVo: no zero-duration phones). No token is longer than 200 frames, which is PiVo's `max_token_frames`.
- `<space>` and punctuation tokens hold the pauses between words. They may be 0. A pause goes to the punctuation token when there is one, otherwise to `<space>`.
- Edge trim follows PiVo's edge-corrected corpus: keep the spoken region, 3 frames of margin, and any energetic frames touching the edges (within 35 dB of the clip peak, up to 30 extra frames). Cuts fall on whole-frame boundaries only.
- The leading margin is part of the first phone. The trailing silence plus its 3-frame margin goes to the final punctuation token, or to the last phone if the utterance has no final punctuation.
- Duration loss target, as in the report: `log(1 + d)`.
- Phase features at training time: `pivo.phase.phoneme_phase_features(durations)`, unchanged.

## Features

- 20 floats per 10 ms frame from Opus `fargan_demo -features` (the git revision PiVo pinned writes 20; the 1.5.2 release writes 36 and we keep the first 20): 18 cepstral coefficients, then the pitch feature (index 18), then pitch correlation (index 19). Schema: `opus-fargan-20-v1`.
- Pitch period for FARGAN: `round(256 / 2**(clip(f[18], -10, 10) + 1.5))`, clipped to 32–255. This is the same as PiVo's `audio.pitch_periods`.
- Timing: feature `k` describes wav samples `[160k − 80, 160k + 80)`. To get the matching target audio for vocoder or critic training, use PiVo's `audio.native_feature_audio(wav, n_frames)` (pads 80 samples at the start), then pre-emphasis 0.85.
- `summary.json → opus` records which Opus build produced the features. Preferred is the git revision PiVo pinned (`bafd40dc…`); the fallback is the 1.5.2 release. Use the same build when you synthesize evaluation audio.

## Filters already applied

- Clips between 1.5 and 15 s after trimming, loudness-normalised to −20 LUFS (peak ≤ 0.95), no clipping, frontend-clean text. Clips with digits or out-of-dictionary English words are dropped.
- MFA 3.4.3 alignment (English: pretrained `english_us_arpa` 3.0.0, as in PiVo; Kannada: an aligner trained on this corpus with the G2P lexicon) must reproduce the exact phone sequence (no `spn`). Internal pauses must be under 1 s, and each speaker's speaking rate must be within 3σ of that speaker's mean.
- The split is deterministic on the normalised text hash: about 1.5% val and 1.5% test. Identical sentences always land in the same split.
