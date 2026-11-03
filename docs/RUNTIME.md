# Runtime reference

## Model files

A model directory contains `model_manifest.json` and five files under
`onnx/`: `metadata.json`, `encoder.onnx`, `decoder.onnx`,
`conditioning.onnx` and `vocoder_step.onnx`. `PiVo` checks each file's size
and SHA256 hash against the manifest before opening any ONNX Runtime
session. The PyTorch checkpoints in the model repository are never loaded.

## Python API

- `PiVo(model_dir, threads=1, max_chars=2000, max_tokens=512)` loads one
  reusable set of sessions.
- `synthesize(text, voice=None, speed=1.0, overrides=None)` returns float32
  PCM at 16 kHz. `voice=None` uses the model's default voice (`Pi` for
  PiVo-1.4M, `kn_f` for PiVo-KN-1.4M); `voices` and `voice_info` list them.
- `save(path, text, ...)` writes a 16 kHz mono PCM16 WAV.
- `iter_audio(...)` yields 160-sample (10 ms) frames. Vocoder and
  de-emphasis state carry over between frames within an utterance, and
  every call starts from zero state. `chunk_frames` controls how many
  acoustic frames are decoded at once; it does not change the output.

## Text frontend

The frontend normalizes English text and looks up ARPAbet phones in the
bundled CMU dictionary. Numbers, supported currency amounts, ordinals and
ISO dates are expanded deterministically. Short all-caps words are spelled
out as initialisms, except NASA, which keeps its dictionary pronunciation.
Unknown words, ambiguous slash dates or fractions, and unsupported
characters raise `ValueError`; unknown words can be handled with
pronunciation overrides.

### Kannada + English models

`metadata.json` names the frontend. PiVo-1.4M uses the English one above
(`pivo-en-v1`, 79 symbols). PiVo-KN-1.4M uses `pivo-kn-en-v1` (`kannada.py`,
127 symbols, 4 voices):

- Text without Kannada script goes through the English frontend unchanged.
- Text with Kannada script is read word by word. Kannada words go through
  rule-based phonemization: inherent vowel, virama, anusvara by place of
  articulation, and visarga as /h/.
- Latin words inside Kannada text use the CMU dictionary or overrides.
- Danda is read as a full stop.
- Before phonemization, `normalize_kn.py` writes numbers and symbols out as
  words in formal / news style. In Kannada text this covers:
  - Kannada and ASCII digits, using the Indian system (ಸಾವಿರ, ಲಕ್ಷ, ಕೋಟಿ)
  - case suffixes on numbers (2026ರಲ್ಲಿ, 5ಕ್ಕೆ, 3ನೇ)
  - decimals, percent (ಶೇಕಡಾ), and ₹/Rs./ರೂ. amounts with paise
  - times with AM/PM, and DD-MM-YYYY or ISO dates
  - phone and PIN numbers (read digit by digit), ranges, units, common
    abbreviations (ಡಾ., ಕಿ.ಮೀ.), initials (ಎಸ್.ಎಂ.) and English acronyms
- English text gets Indian conventions on top of the rules above: ₹, lakh and
  crore grouping, clock times, day-first dates, phone numbers and units.
- Latin words missing from the CMU dictionary (usually Indian names such as
  Bengaluru or Hubballi) are read with romanization rules instead of raising.
  Pass `guess_unknown=False` to `kannada.phonemize` for the strict behaviour.

The runtime refuses a model whose symbol table does not match its frontend.

## Durations

The encoder predicts `log(1 + duration)` for each phone. Durations are
rounded, every spoken phone gets at least one frame, and PAD, BOS and EOS
get none. `max_token_frames=200` and `max_utterance_frames=3000` bound the
expansion. The decoder attends to at most 64 previous frames, so its
key/value cache stays bounded. Within-phone timing features are computed
on the host.

## Vocoder features

The vocoder takes 20 features at 100 Hz: 18 cepstral coefficients, log
pitch and pitch correlation. The pitch period in samples is
`round(256 / 2**(feature[18] + 1.5))`, clipped to 32–255. Each feature
frame produces four 40-sample subframes, using one frame of lookahead.
De-emphasis uses coefficient 0.85 and keeps its state across chunks. The
sample rate is 16 kHz.

## HTTP API

`POST /v1/audio/speech` returns a complete WAV file. The JSON body accepts
`input`, `voice`, `speed`, `response_format` (`"wav"`), an optional
`model` (`"pivo-1.4m"`) and optional `overrides`. The route follows the
shape of OpenAI's speech endpoint but does not implement the rest of that
API.

Empty text, inputs over the limits and unknown words return `422`. A
request that arrives while another is being synthesized returns `429`.
Setting `PIVO_API_KEY` enables bearer authentication on `/v1` routes.
Each worker process loads its own copy of the model.
