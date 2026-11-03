# PiVo Kannada + English training

PiVo-1.4M ships no training code. This folder holds two things: the data pipeline that builds a Kannada + English dataset in PiVo's format, and a PyTorch rebuild of the PiVo acoustic model with the code that trains it. Everything heavy runs on Modal.

## Layout

```
training/
  data_pipeline/            build the dataset (Modal, CPU)
    prepare_pivo_v4.py      prepare -> MFA align -> Opus features -> manifests
    pivo_kn_v3/             text frontend: Kannada G2P, English (PiVo's CMUdict logic), token table
    DATA_FORMAT.md          what the dataset on the `pivo-data` volume looks like
  pivo_train/               model + stage-1 trainer + stage-2 refinement (imported by train_gpu.py)
  tests/test_parity.py      our model vs PiVo's official ONNX graphs
  train_gpu.py              Modal entrypoint for training (GPU)
  runs/<run-name>/          downloaded samples + checkpoints (gitignored)
```

## 1. Data (`data_pipeline/`)

Sources: SYSPIN Kannada (F + M) from the `tts1-raw-datasets` volume, and Hi-Fi TTS speakers 92 (F) and 6097 (M) for English. The output goes to the `pivo-data` volume under `/pivo_kn_en_v1`. You only need to re-run this to change the data.

```bash
cd training/data_pipeline
modal run prepare_pivo_v4.py --smoke        # small end-to-end test -> /smoke
modal run --detach prepare_pivo_v4.py       # full build -> /pivo_kn_en_v1
```

## 2. Training (`train_gpu.py`)

| File | What it does |
|---|---|
| `pivo_train/model.py` | `PiVoAcoustic`: PiVo-1.4M's acoustic model, rebuilt from the released ONNX graphs. Its parameter names and shapes match `acoustic.pt` exactly (584,965 params). It also handles the warm start: voices grow from 2 to 4, and Kannada tokens 79–126 start from similar English phones. |
| `tests/test_parity.py` | Loads `acoustic.pt` into our model and compares it with `encoder.onnx` / `decoder.onnx`. Every run checks this first. |
| `pivo_train/data.py` | Reads `manifests/*.jsonl`, normalises features with PiVo's mean/std, builds length-bucketed batches. |
| `pivo_train/pack.py` | Packs every feature file into one cache on the `pivo-runs` volume. This is done once; later runs load it in seconds. |
| `pivo_train/train.py` | Stage-1 loss: SmoothL1(features) + 0.1 × MSE(log durations), with AdamW. Saves checkpoints, val losses, and one WAV per voice at each eval. |
| `pivo_train/refine.py` | Stage 2: spectral refinement (see below). |
| `pivo_train/vocoder.py` | PiVo's FARGAN vocoder (ONNX), used to render the sample WAVs. |

```bash
cd training
modal run train_gpu.py --smoke                          # quick GPU test on /smoke
modal run --detach train_gpu.py --run-name kn_en_v1     # PiVo recipe: batch 8, lr 3e-4, 80k steps
modal run --detach train_gpu.py --run-name kn_en_v1 --resume
```

The run writes to the `pivo-runs` volume under `<run-name>/`:

- `train_log.jsonl`
- `samples/step_XXXXXX/voice{N}_{speaker}.wav` + `.txt`, every 4,000 steps
- `best.pt`, `last.pt`, `step_XXXXXX.pt`

### Download samples and checkpoints

Create the folder first and keep the trailing `/`. Without it, Modal writes every file into one file.

```powershell
mkdir runs/kn_en_v1
modal volume get pivo-runs kn_en_v1/samples runs/kn_en_v1/          # -> runs/kn_en_v1/samples/step_*/
modal volume get pivo-runs kn_en_v1/best.pt runs/kn_en_v1/best.pt
```

## Voices

| id | speaker | warm start |
|---|---|---|
| 0 | `en_hifi92_f` | PiVo voice Pi |
| 1 | `en_hifi6097_m` | PiVo voice Vo |
| 2 | `kn_syspin_f` | copy of Pi |
| 3 | `kn_syspin_m` | copy of Vo |

## Run `kn_en_v1` (3 Nov 2026)

The run used an A100, the `pivo` preset, and 80,000 steps (~18 epochs over 36,073 clips, ~50 min). The val score is val_feature + 0.1 × val_duration, which is the rule `best.pt` uses.

| step | val_feature | val_duration | score |
|---|---|---|---|
| 0 | 0.4961 | 0.3651 | 0.5326 |
| 4,000 | 0.2483 | 0.1267 | 0.2610 |
| 40,000 | 0.2346 | 0.1055 | 0.2452 |
| **80,000** | **0.2311** | 0.1052 | **0.2417** (best) |

`best.pt` = `step_080000.pt`. Val was still improving at the end, so the model didn't overfit.

## 3. Stage 2: spectral refinement (`pivo_train/refine.py`)

This follows the PiVo report, sec. 3.3. It starts from `/runs/<from-run>/best.pt` and trains only the decoder against a small critic, which pushes back on the over-smoothing that makes speech sound metallic.

- **From the report:** 1,200 updates, batch 4, 128-frame segments, lr 1e-4. The critic is a two-scale temporal critic (~110k params) that sees the 18 cepstra + voice id. Loss = smooth-L1 + 0.01 × adversarial + 0.1 × feature matching. The encoder and duration predictor stay frozen; the run checks afterwards that all 61 tensors are unchanged.
- **Not in the report**, so these are our choices: LSGAN losses, AdamW (0.8, 0.99) for both nets, and the critic layout (109,890 params).
- **Model size:** the critic is only used in training, so the model size doesn't change.

```bash
modal run train_gpu.py --stage refine --smoke      # quick test on /smoke
modal run --detach train_gpu.py --stage refine     # kn_en_v1/best.pt -> /runs/kn_en_v1_refine
```

Outputs in `/runs/kn_en_v1_refine/`:

- **`refined.pt`** is the final model (= `step_001200.pt`).
- **`refine_log.jsonl`** has, at each eval, `val_feature` (should stay close to stage 1) and `smoothness`. Smoothness compares how fast the predicted cepstra change frame to frame against real speech: 1.0 matches real speech, and released PiVo scores ~0.65.
- **`samples/ceiling/*_real.wav`** are real features played through the vocoder. They show the vocoder's own floor for metallic sound.
- **`samples/step_000000/`** (before refinement) and **`samples/step_001200/`** (after) each have `*_tf.wav` (real timing) and `*_tts.wav` (normal TTS), 3 sentences per voice.

```powershell
mkdir runs/kn_en_v1_refine
modal volume get pivo-runs kn_en_v1_refine/samples runs/kn_en_v1_refine/
modal volume get pivo-runs kn_en_v1_refine/refined.pt runs/kn_en_v1_refine/refined.pt
```

## 4. Export to the runtime (`export_onnx.py`)

This turns `refined.pt` into a model folder that `src/pivo` runs on CPU with no PyTorch:

- **Exported:** a new `encoder.onnx` and `decoder.onnx`.
- **Copied unchanged:** PiVo's vocoder graphs.
- **Written:** `metadata.json` (127 symbols, voices `en_f en_m kn_f kn_m`, frontend `pivo-kn-en-v1`) and `model_manifest.json` (sha256 of every runtime file).

The script refuses to finish unless the ONNX graphs match PyTorch: encoder ~3e-6, features ~1e-4, decoded in chunk sizes 1, 16 and 47.

```bash
cd training
python export_onnx.py --ckpt runs/kn_en_v1_refine/refined.pt --pivo-dir ../models/PiVo-1.4M --out ../models/PiVo-KN-1.4M
pivo synthesize --model-dir ../models/PiVo-KN-1.4M --text "ನಮಸ್ಕಾರ" --output hello.wav
```

Tests: `pytest` from the repo root. Set `PIVO_TEST_KN_MODEL_DIR=models/PiVo-KN-1.4M` to also run the 4-voice tests.

## Not done yet

- Optional: a short run with the lr decaying from 3e-4 to 3e-5, starting from 80k.
- A Kannada speech-recognition check over the val sentences.
- Reading numbers in Kannada text. of the trained model, and updating `src/pivo` (Kannada frontend, 127 symbols, 4 voices) so the runtime can speak Kannada.
