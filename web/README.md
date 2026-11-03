# Voxal Studio: Kannada and English text to speech in the browser

A web page where anyone can type Kannada or English and hear it spoken by PiVo-KN-1.4M. The
**Language** switch picks the voice: **Kannada** uses the model's `kn_m` voice and **English** uses
`en_m` (set in `build.py`, `SITE_VOICES`). The model runs **in the visitor's browser, on their own CPU** (ONNX Runtime Web,
WebAssembly). The server only hands out static files, so there is no GPU, no inference server
and nothing to overload. Text and audio never leave the visitor's computer.

```
visitor's browser                                   Modal (or any static host)
┌───────────────────────────────────────────┐       ┌────────────────────────┐
│ page (main.js) ── text ──▶ Web Worker      │ ◀──── │ HTML/JS, 4 ONNX files, │
│   ▲                       frontend.js      │ once  │ ORT wasm, dictionary   │
│   │                       (normalize + G2P)│       └────────────────────────┘
│   │                       engine.js        │
│   └── PCM blocks ◀─────── (ONNX, WASM)     │
│ player.js (plays while generating) → WAV   │
└───────────────────────────────────────────┘
```

| | |
|---|---|
| **First visit** | about 9 MB download (model 5.3 MB + runtime 3 MB, gzip), then cached by the browser |
| **Speed** | about 7× faster than real time on a 2-core CPU; audio starts in about 0.3 s |
| **Same model** | the exact `models/PiVo-KN-1.4M/onnx` graphs; browser audio matches `pivo` within 3/32768 (both voices) |
| **Same text rules** | `app/frontend.js` is a line-by-line port of `kannada.py` + `normalize_kn.py`; identical token ids on 11,085 test sentences |
| **Browsers** | recent Chrome, Edge, Firefox, Safari 16.4+ (desktop and phone) |

## Run it on your PC

```bash
pip install uvicorn
python web/build.py          # gathers everything into web/dist (downloads ONNX Runtime Web once)
python web/server.py         # open http://localhost:8000
```

## Deploy on Modal

```bash
python web/build.py
modal deploy web/modal_app.py
```

Modal prints the public URL (`https://<workspace>--voxal-studio.modal.run`). The app is one small
CPU container (0.5 CPU, 512 MB) that sleeps when nobody visits. The first visitor after a quiet
period waits a second or two for it to start. To avoid that, keep one container warm (small idle
cost): `VOXAL_MIN_CONTAINERS=1 modal deploy web/modal_app.py`.

Re-run both commands after any change to `web/app/` or the model.

## Files

| file | what it does |
|---|---|
| `app/index.html`, `app/styles.css` | page layout and Voxal theme |
| `app/main.js` | text box, Kannada / English switch, status, generate / stop, waveform, download |
| `app/worker.js` | Web Worker: downloads the model with progress, splits text into sentences, streams audio |
| `app/engine.js` | encoder → durations → decoder → FARGAN vocoder, same steps as `src/pivo/runtime.py` |
| `app/frontend.js` | Kannada / Indian-English normalization and G2P, port of the Python frontend |
| `app/player.js` | plays audio while it is generated; pause, seek, WAV export |
| `build.py` | builds `dist/`: checks the model, pins ONNX Runtime Web 1.20.1 by SHA-256, gzip copies, manifest |
| `server.py` | dependency-free static server (gzip, ETag, long caching for hashed files, CSP) |
| `modal_app.py` | Modal deployment of `server.py` + `dist/` |
| `tests/parity_ref.py`, `tests/parity.mjs` | frontend parity: Python vs JavaScript on 11,085 sentences |
| `tests/e2e.py` | real-browser test: load, both voices vs Python, stop, speed, language switch, mobile |

## How the page handles text

- **Long text** is read sentence by sentence (then at commas, then between words), with short
  pauses between pieces. Audio starts as soon as the first sentence is ready.
- **Numbers, dates, times, ₹ amounts, percent, phones, units** are written out by the same
  rules as the Python package (formal / news style). "How it was read" shows the result.
- **English voice:** text goes through Indian-English normalization (₹1,25,000 → "one lakh twenty
  five thousand rupees", 03/11/2026 → "third November, two thousand twenty six") and the CMU
  dictionary. If the text and the chosen language do not match, the page suggests switching.
- **English words** inside Kannada use the CMU dictionary (fetched in the background after
  the model loads). Names not in it (Hubballi, Shetty) are read with Indian spelling rules.
- **Emoji and unsupported symbols** are skipped instead of failing, and listed on the page.

## Tests

```bash
# text frontend: JavaScript must give the same token ids as Python
python web/tests/parity_ref.py web/tests/parity_ref.json
node web/tests/parity.mjs web/tests/parity_ref.json web/dist/lexicon/cmudict.txt

# whole site in Chromium (pip install playwright && playwright install chromium)
python web/tests/e2e.py --model-dir models/PiVo-KN-1.4M/onnx
```

If you change a normalization or G2P rule in `src/pivo/`, make the same change in
`app/frontend.js` and run the parity test.
