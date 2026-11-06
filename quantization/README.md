# PiVo-Multi-1.8M: FP16, INT8 and INT4 versions

Open `compare.html` to listen to all 24 voices in FP32, FP16, INT8 and INT4 side by side.

## Current format

`models/multi_w128_refine` is **FP32 everywhere**. All 1,847,722 weights are float32
(ONNX opset 17, exported from PyTorch). There are no quantized or float16 tensors.

| graph | weights | size |
|---|---|---|
| encoder.onnx | 732,353 | 3.05 MB |
| decoder.onnx | 298,812 | 1.24 MB |
| conditioning.onnx | 70,272 | 0.28 MB |
| vocoder_step.onnx | 746,325 | 3.00 MB |
| **total** | **1,847,722** (acoustic 1,031,125 + vocoder 816,597) | **7.57 MB** |

## Sizes (MB = 1,000,000 bytes)

| precision | model MB | model MB, gzip | whole website download | vs FP32 |
|---|---|---|---|---|
| FP32 (now) | 7.57 | 6.93 | 10.8 MB | 100% |
| FP16 | 3.89 | 3.48 | 7.4 MB | 51% |
| INT8 | 2.11 | 1.76 | 5.7 MB | 28% |
| INT4 | 1.49 | 1.12 | 5.0 MB | 20% |

The website download also includes ONNX Runtime Web (~3 MB gzip), the CMU dictionary and the page,
so it does not shrink as fast as the model.

INT4 is not 1/8 of FP32: each block of 32 weights also stores a float32 scale (~1 extra bit per weight),
and biases, norm gains and the graph itself stay as they are.

## Quality vs FP32 (all 24 voices, the welcome sentence of each language)

| precision | phone durations unchanged | pitch error | spectral distance |
|---|---|---|---|
| FP16 | 99.9% | 0.1 cents | 0.17 dB |
| INT8 | 97.5% | 3.6 cents | 1.35 dB |
| INT4 | 67.0% | 50.1 cents | 5.22 dB |

- **Phone durations unchanged:** share of phones whose predicted length (in 10 ms frames) is identical
  to FP32. A change means slightly different timing / rhythm.
- **Pitch error:** mean pitch difference on voiced frames, using the FP32 timing. 100 cents = one semitone.
- **Spectral distance:** mean absolute log-mel difference with the FP32 timing. For scale: adding tiny
  random noise (0.01) to the FP32 features already gives about 1.4 dB, because the vocoder is recurrent
  and small changes make a different but equally clean waveform.

Which half is sensitive (other half kept FP32):

| | acoustic model only | vocoder only |
|---|---|---|
| INT8 | 1.13 dB, pitch 3.6 cents | 1.02 dB |
| INT4 | 3.05 dB, pitch 50 cents, 1/3 of durations change | 4.60 dB, peaks above 1.0 (clipping) |

Two INT4 variations were also tried: blocks of 16 (1.75 MB, 4.31 dB), and int4 only for the matrix
weights with int8 for the conv kernels and embedding tables (1.82 MB, 4.25 dB). Neither is close to INT8.
Plain post-training INT4 is too coarse for a model this small. It would need quantization-aware
fine-tuning to be usable.

## Speed

Weight-only INT8 / INT4 store small weights but compute in float32. Two ways to run them:

1. **Dequantize once at load (recommended).** Same speed as FP32, and the quality numbers above.
   Needs one change in `web/app/engine.js`:
   ```js
   const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all",
     // int8 / int4 weights (DequantizeLinear): dequantize once at load instead of every call
     extra: { session: { disable_quant_qdq: "1" } } };
   ```
   Tested in Chromium with ONNX Runtime Web 1.20.1: INT8 and INT4 run as fast as FP32 (≈5× faster than
   real time) and match Python ONNX Runtime on all tested voices (correlation ≥ 0.99995).
2. **Without that change**, ONNX Runtime Web 1.20.1 dequantizes the weights on every vocoder step:
   about **4–5× slower** in the browser (7.4 s of speech took 7 s instead of 1.5 s), and the output
   no longer matches Python. Desktop ONNX Runtime 1.29 instead fuses them into integer kernels, which
   is ~2× faster on the vocoder but adds error (INT8 1.74 dB instead of 1.35 dB).

FP16 needs no code change. It is stored as float16 and cast to float32 at load, so it runs at FP32 speed.
A fully float16 graph (FP16 maths) was not built: the CPU / WebAssembly backend has no speed gain from it,
and the converter could not handle this graph's Range / Cast nodes.

## Recommendation

- **INT8** for the website: 3.6× smaller model (6.93 → 1.76 MB gzip), whole first download 10.8 → 5.7 MB,
  differences at the level of the vocoder's own run-to-run variation. Add the `engine.js` line above.
- **FP16** if you want no measurable change at all: half the size.
- **INT4**: not as it is. Listen to `compare.html`: pitch and rhythm drift and the vocoder gets rougher.

## Files

- `models/multi_w128_refine_fp16`, `_int8`, `_int4`: full model folders (same layout as
  `multi_w128_refine`, with `model_manifest.json` hashes and licence files). For the website:
  `python web/build.py --model-dir models/multi_w128_refine_int8/onnx` (plus the `engine.js` change).
- `samples/<precision>/<lang>_<f|m>.wav`: the sentences played in `compare.html`.
- `make_variants.py`: builds the variants from the FP32 export.
  `python quantization/make_variants.py models/multi_w128_refine/onnx OUT fp16,int8,int4`
- `evaluate.py`: synthesizes the 24 voices with a variant and measures it against FP32
  (`--noqdq` = dequantize at load).
- `results.json`: all sizes and per-voice numbers, including the extra variants.

Only objective measurements were made here. Nobody has listened to these samples yet, so please judge
INT8 by ear before switching the site.
