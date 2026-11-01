# PiVo

How many parameters does a voice need? PiVo is a complete English
text-to-speech model with two voices, **Pi** and **Vo**, in
**1,401,562 learned parameters**, vocoder included. This repository holds
the inference code: a Python API, a command-line tool and a small HTTP
server that returns WAV audio.

PiVo runs on a CPU with NumPy and ONNX Runtime and needs no PyTorch or GPU.
Output is 16 kHz mono PCM16 WAV.

Author: Leon Lehmann · Empero Research (research@empero.org) · [Credits](AUTHORS.md)

## Install

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[hub]'
```

The model weights are published separately on Hugging Face. Download them
once; after that, everything runs offline:

```bash
pivo download --repo ORG/PiVo-1.4M --revision main --output models/pivo
```

For reproducible deployments, pass a specific commit SHA as `--revision`.
On every start, PiVo checks the size and SHA256 hash of all five model
files.

## Synthesize

```bash
pivo synthesize --model-dir models/pivo --voice Pi \
  --text "NASA is ready for the next mission." --output pi.wav
```

From Python:

```python
from pivo import PiVo

model = PiVo("models/pivo", threads=1)
model.save("vo.wav", "Hello, my name is Vo.", voice="Vo")

for pcm in model.iter_audio("This is a short example.", voice="Pi"):
    ...  # float32 mono frames of 10 ms each
```

The full text is encoded first, then `iter_audio` yields audio frame by
frame, so playback can start before the whole utterance is generated.

## Pronunciation

PiVo looks words up in the CMU pronouncing dictionary. If a word is
missing, it stops with an error rather than guessing. You can supply
pronunciations as ARPAbet phones with stress markers:

```bash
pivo synthesize --model-dir models/pivo --voice Vo \
  --text "Hello, Zzvo." --overrides examples/pronunciations.json --output name.wav
```

where `examples/pronunciations.json` contains
`{"zzvo": ["Z", "IY1", "V", "OW0"]}`. See [docs/RUNTIME.md](docs/RUNTIME.md)
for the full text-handling rules.

## Serve over HTTP

```bash
python -m pip install '.[server]'
pivo serve --model-dir models/pivo --host 127.0.0.1 --port 8000
```

```bash
curl --fail http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"NASA is ready for the next mission.","voice":"Pi","response_format":"wav"}' \
  --output pi.wav
```

Interactive API docs are at `/docs`, the voice list at `/v1/voices` and a
health check at `/health`. To require a bearer token on the `/v1` routes,
set `PIVO_API_KEY` before starting the server.

The server is a simple reference implementation. Each process runs one
synthesis at a time; a concurrent request gets `429` with a `Retry-After`
header. Invalid input, unknown words or text that is too long get `422`.
Limits are 2,000 characters, 512 frontend tokens and 30 seconds of audio
per request, with speed between 0.5 and 2.0. The server listens on
localhost by default; put it behind your usual TLS and authentication
before exposing it.

## Docker

```bash
docker build -t pivo .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(realpath models/pivo):/models:ro" pivo
```

The image contains the code and server but no weights. Mount the model
directory read-only at `/models`.

## Development

```bash
python -m pip install '.[server,dev]'
pytest
```

The unit tests need no model files. To also run the integration tests
against the real model:

```bash
PIVO_TEST_MODEL_DIR="$(realpath models/pivo)" pytest
```

## Documentation

- Technical report: `technical_report.pdf` in the model repository covers
  architecture, training data, evaluation and the experiments that did not work.
- [Runtime reference](docs/RUNTIME.md): model files, state, features and
  text rules.
- [Evaluation results](docs/evaluation.json).

PiVo is a research model. Its speech is very intelligible (0.53% word
error rate on our held-out test set), but some sentences still sound
metallic or robotic. We have not yet measured speed or memory on ARM or
mobile hardware.

## License

Our code and the acoustic model weights are released under Apache-2.0. The
FARGAN-derived vocoder keeps the Xiph/Opus BSD license, and the
pronunciation dictionary keeps its CMU license. See
[THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES/](LICENSES/).

## Citation

```bibtex
@techreport{lehmann2026pivo,
  title = {PiVo: two English voices in 1.4 million inference parameters},
  author = {Lehmann, Leon},
  institution = {Empero Research},
  year = {2026},
  month = {November},
  note = {Technical report, version 1.0}
}
```
