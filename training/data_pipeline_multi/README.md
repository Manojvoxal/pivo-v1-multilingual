# PiVo multilingual data pipeline (11 major Indian languages + English)

Builds the training set for the new multilingual PiVo model (trained from scratch).
PiVo-KN-1.4M (Kannada + English) stays a separate model; nothing here touches it.

| Part | What |
|---|---|
| `prepare_pivo_multi.py` | Modal app: scan → prepare → align → build → finalize |
| `pivo_multi/` | frontend (G2P for all scripts), sources, stages, config |
| `DATA_FORMAT.md` | what the trainer reads + the trainer changes needed |
| `tests/` | G2P tests (73) + offline end-to-end test with fake data |

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
| IndicTTS English (IIT Madras) | Indian English | Tamil F + Tamil M talents | all (~12 h each) |

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
4. **IndicTTS English** (can come later; use `--english hifi` until then):
   - Download from https://www.iitm.ac.in/donlab/indictts/database (accept the IITM
     license) the **Tamil female English** and **Tamil male English** sets, or e-mail
     smtiitm@gmail.com, subject "Indic Database Download Request", body: Language Tamil,
     Gender Male and Female, Type English.
   - Listen to a few clips. To use other talents (Assamese F 12.05 h / M 11.30 h,
     Gujarati M 10.13 h…), edit `INDICTTS_EN` in `pivo_multi/config.py`.
   - Unzip and upload (any layout with wav files + `txt.done.data` works):
     ```
     modal volume put tts1-raw-datasets "C:\path\to\Tamil_female_english" /english/indictts/tamil_female
     modal volume put tts1-raw-datasets "C:\path\to\Tamil_male_english"   /english/indictts/tamil_male
     ```

## Run (from this folder)

```
# 0) offline tests on your laptop (no Modal)
pip install pytest pyarrow soundfile soxr pyloudnorm numpy
set PIVO_CMUDICT=C:\Users\Manoj\pivo-v1\src\pivo\resources\cmudict.dict
python -m pytest tests -q

# 1) smoke test on Modal (~20 min): en (Hi-Fi), kn (SYSPIN), hi + ta (Rasa, 2 shards), tiny hours
modal run prepare_pivo_multi.py --smoke

# 2) look at Rasa's style labels before downloading audio (a few minutes)
modal run prepare_pivo_multi.py --stage scan --only
#    -> prints, per language: hours by gender x style, scripts used, what got selected.
#    Pick the read-speech styles, then pin them:
modal run prepare_pivo_multi.py --stage scan --only --styles "BOOK,NEWS,WIKI"

# 3) full run (keeps going if the laptop sleeps); add --english hifi if IndicTTS is not uploaded yet
modal run --detach prepare_pivo_multi.py --styles "BOOK,NEWS,WIKI"

#    or tier-1 languages first:
modal run --detach prepare_pivo_multi.py --langs kn,en,hi,te,ta,bn --styles "BOOK,NEWS,WIKI"
#    later, optional tier 3 (finalize merges everything prepared so far):
modal run --detach prepare_pivo_multi.py --phases 3 --styles "BOOK,NEWS,WIKI"

# resume from a stage / run one stage
modal run prepare_pivo_multi.py --stage align
modal run prepare_pivo_multi.py --stage finalize --only

# after a good finalize: free ~90 GB of MFA corpus copies
modal run prepare_pivo_multi.py --stage cleanup
```

Paste the printed scan table and the final summary back into the chat.

## How it works

1. **scan** reads only Rasa's text/gender/style/duration columns (no audio), runs the
   frontend on every sentence, and selects 12 h × 1.25 per voice of clips that pass.
   It stops the run if a language has < 3 h selected for a voice (wrong `--styles`,
   or no access).
2. **prepare** downloads only the parquet files holding selected clips (48 in parallel),
   resamples to 16 kHz mono, −20 LUFS, writes the MFA corpus.
3. **align** trains one MFA aligner per language on 1.5 h per voice, then aligns all clips.
4. **build** computes per-token durations and Opus FARGAN features (same code as PiVo-KN).
5. **finalize** caps each voice at its hours, merges manifests, writes symbols / languages /
   voices / feature stats / summary.

Expected scale (tiers 1 + 2): ~295 h, ~150k clips. Most of the cost is CPU time (MFA + feature
extraction); no GPU.
