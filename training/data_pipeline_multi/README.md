# PiVo multilingual data pipeline (11 major Indian languages + English)

Builds the training set for the new multilingual PiVo model (trained from scratch).
PiVo-KN-1.4M (Kannada + English) stays a separate model; nothing here touches it.

| Part | What |
|---|---|
| `prepare_pivo_multi.py` | Modal app: scan → prepare → align → build → finalize |
| `pivo_multi/` | frontend (G2P for all scripts), sources, stages, config |
| `pivo_multi/numnorm.py`, `numwords.json` | number reader for inference (hi mr gu pa bn or as ta te ml): numbers, dates, years, times, money, %, units, ordinals, suffixes, acronyms, symbols; `frontend.phonemize(text, lang, strict=False, numbers=True)` |
| `NUMBER_REVIEW.xlsx` | every number-word choice per language, for native speakers to confirm or correct |
| `DATA_FORMAT.md` | what the trainer reads + the trainer changes needed |
| `tests/` | G2P tests (73), number reader tests, offline end-to-end test with fake data |

## Data plan (from the project plan)

Languages are in tiers (`pivo_multi/langs.py`). The default run is tiers 1 + 2:
**12 languages, 24 voices, ~295 h**.

| Tier | Languages | Prepared by default? |
|---|---|---|
| 1 | English, Kannada, Hindi, Telugu, Tamil, Bengali | yes |
| 2 | Marathi, Gujarati, Punjabi, Malayalam, Odia, Assamese | yes |
| 3 | Nepali, Konkani, Maithili, Dogri, Sanskrit (easy to read, niche) | no: add later with `--phases 3` |
| 4 | Urdu, Kashmiri, Sindhi (Perso-Arabic: short vowels not written), Manipuri, Bodo (unwritten tones), Santali (unverified G2P) | no: dropped for now |

| Dataset | Used for | Voices | Hours used |
|---|---|---|---|
| Rasa (AI4Bharat, HF `ai4bharat/Rasa`, gated) | the Indian languages except Kannada | 1 F + 1 M each | 12 h per voice, neutral/read styles only |
| SYSPIN | Kannada | `kn_syspin_f/m` | 14.5 h per voice |
| SPRINGLab/IndicTTS-English (IIT Madras, HF, public) | Indian English | 1 F + 1 M talent, found automatically | up to 13 h each |

Ids are append-only (tier order), so adding tier 3 later never renumbers voices or languages.

## One-time setup (your side)

1. **Rasa access**: open https://huggingface.co/datasets/ai4bharat/Rasa while logged in
   and accept the terms.
2. **Hugging Face token on Modal** (read token from https://huggingface.co/settings/tokens):
   ```
   modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
   ```
   (Already have a secret with HF_TOKEN under another name? Change `HF_SECRET` at the top
   of `prepare_pivo_multi.py`.)
3. **Output volume** (Volumes v2, because the dataset has ~1.5M small files):
   ```
   modal volume create pivo-multi-data --version=2
   ```
4. **English** needs nothing: the default source is `SPRINGLab/IndicTTS-English` on Hugging
   Face (IIT Madras SPRING Lab's public copy of the IndicTTS English recordings). It has no
   speaker labels, so the scan stage separates the voice talents itself (speaker embeddings +
   pitch) and picks the female and the male talent with the most hours.
   Alternatives: `--english hifi` (American Hi-Fi TTS) or `--english indictts` (IITM zips
   uploaded to `tts1-raw-datasets:/english/indictts/tamil_female|tamil_male`).

## Run (from this folder)

```
# 0) offline tests on your laptop (no Modal)
pip install pytest pyarrow soundfile soxr pyloudnorm numpy
$env:PIVO_CMUDICT="C:\Users\Manoj\pivo-v1\src\pivo\resources\cmudict.dict"
python -m pytest tests -q

# 1) smoke test on Modal (~20 min): en (Hi-Fi), kn (SYSPIN), hi + ta (Rasa, 6 parquet files each), tiny hours
modal run prepare_pivo_multi.py --smoke

# 2) scan all Rasa metadata (a few minutes, no audio download) and check the table:
#    hours per voice x style, scripts, what got selected
modal run prepare_pivo_multi.py --stage scan --only

# 3) full run (keeps going if the laptop sleeps)
modal run --detach prepare_pivo_multi.py

#    English on its own (speaker scan of SPRINGLab ~10-15 min, then prepare/align/build):
modal run --detach prepare_pivo_multi.py --langs en
#    listen to the talents it found, and override the pick if you prefer others:
modal volume get pivo-multi-data pivo_multi_v1/reports/english_speakers ./en_samples
modal run --detach prepare_pivo_multi.py --langs en --en-speakers "c07,c12"
#    when several runs were started separately, merge everything at the end:
modal run prepare_pivo_multi.py --stage finalize --only

#    or tier-1 languages first:
modal run --detach prepare_pivo_multi.py --langs kn,en,hi,te,ta,bn
#    later, optional tier 3 (finalize merges everything prepared so far):
modal run --detach prepare_pivo_multi.py --phases 3

# resume from a stage / run one stage
modal run prepare_pivo_multi.py --stage align
modal run prepare_pivo_multi.py --stage finalize --only

# after a good finalize: free ~90 GB of MFA corpus copies
modal run prepare_pivo_multi.py --stage cleanup
```

Paste the printed scan table and the final summary back into the chat.

## How it works

1. **scan** reads only Rasa's text/gender/style/duration columns (no audio), keeps the
   read-speech styles **WIKI, BOOK, NEWS, INDIC** (emotions, CONV, PROPER NOUN and the
   ALEXA/BB/DIGI/UMANG commands are dropped; override with `--styles "A,B"`), runs the
   frontend on every sentence, and selects 12 h × 1.25 per voice of clips that pass.
   It stops the run if a voice has < 3 h selected.
2. **prepare** downloads only the parquet files holding selected clips (48 in parallel),
   resamples to 16 kHz mono, −20 LUFS, writes the MFA corpus.
3. **align** trains one MFA aligner per language on 1.5 h per voice, then aligns all clips.
4. **build** computes per-token durations and Opus FARGAN features (same code as PiVo-KN).
5. **finalize** caps each voice at its hours, merges manifests, writes symbols / languages /
   voices / feature stats / summary.

Expected scale (tiers 1 + 2): ~295 h, ~150k clips. Most of the cost is CPU time (MFA + feature
extraction); no GPU.

## Reading numbers (inference only)

Training text has no digits (clips with digits are dropped), so the model never needs a number
reader for training. At inference the web app and `frontend.phonemize(..., numbers=True)` write
numbers out as words first: Kannada and English with PiVo's own readers, the other ten languages
with `pivo_multi/numnorm.py` (formal / news style, same rule order as PiVo's Kannada reader).

The words live in `pivo_multi/numwords.json`, one block per language: 0-99, hundreds, thousand /
lakh / crore forms (with Tamil / Malayalam sandhi and Telugu oblique forms), years, months, clock,
currency, units, ordinals, letter names, abbreviations. 0-99 come from Unicode CLDR 49 (AI4Bharat
indic-numtowords for Odia and Assamese) with the fixes listed in `NUMBER_REVIEW.xlsx`. Changing a
word needs no retraining: edit the JSON, run `pytest tests/test_numnorm.py`, rebuild the site
(`python web/build.py` copies it into `web/app/numwords.js`). A rule change in `numnorm.py` must be
made in `web/app/numbers_multi.js` too (see `web/tests/numbers_parity.mjs`).
