"""Build the Voxal Studio website into web/dist/ (plain static files).

The site runs PiVo-KN-1.4M in each visitor's browser, so the build only gathers files:

* the page and its scripts (web/app/)
* the four ONNX graphs (models/PiVo-KN-1.4M/onnx/)
* ONNX Runtime Web 1.20.1 (downloaded once, checked against pinned SHA-256 hashes)
* the English dictionary, first CMU pronunciation per word (from src/pivo/resources/)
* licence texts, a manifest with per-file hashes, and .gz copies for fast transfer

    python web/build.py
    python web/build.py --model-dir models/PiVo-KN-1.4M/onnx --ort-dir path/to/onnxruntime-web/dist
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
APP_FILES = ["index.html", "styles.css", "main.js", "player.js", "worker.js", "engine.js", "frontend.js", "favicon.svg"]
MODEL_FILES = ["encoder", "decoder", "conditioning", "vocoder_step"]
SITE_VOICES = [  # voices offered on the site: (model voice id, name shown, language)
    ("kn_m", "Kannada", "kn"),
    ("en_m", "English", "en"),
]
ORT_VERSION = "1.20.1"
ORT_FILES = {  # sha256 of the files published in the onnxruntime-web@1.20.1 npm package
    "ort.wasm.min.mjs": "f53ed4792e758e7232f779d479eb8931b97ff3ab2e3b9a33ee00d1251ffdaad6",
    "ort-wasm-simd-threaded.mjs": "745eb7c0ce6f18a6aa521971b2877babc7ffb27eecb58ab3bc6e5ef4692672e8",
    "ort-wasm-simd-threaded.wasm": "207d02be4591c156b0a98f024f3d58005b5b04c92274d759fb390338c63559ea",
}
ORT_MIRRORS = ["https://cdn.jsdelivr.net/npm/onnxruntime-web@{version}/dist/{name}",
               "https://unpkg.com/onnxruntime-web@{version}/dist/{name}"]
COMPRESSIBLE = {".html", ".js", ".mjs", ".css", ".svg", ".json", ".txt", ".wasm", ".onnx"}
MARKER = ".voxal-dist"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fail(message):
    sys.exit(f"build failed: {message}")


def check_model(model_dir):
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        fail(f"{meta_path} not found. Pass --model-dir pointing at the PiVo-KN-1.4M onnx folder.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("frontend") != "pivo-kn-en-v1":
        fail("this is not a PiVo-KN export (expected frontend pivo-kn-en-v1)")
    missing = [v for v, _, _ in SITE_VOICES if v not in meta.get("voices", {})]
    if missing:
        fail(f"the model has no voice {', '.join(missing)}")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from pivo import kannada
    except ImportError as error:
        fail(f"could not import the pivo package to check the symbol table ({error}). Run `pip install -e .` first.")
    if tuple(meta["symbols"]) != tuple(kannada.SYMBOLS):
        fail("the model's symbol table does not match the Kannada frontend")
    for name in MODEL_FILES:
        if not (model_dir / f"{name}.onnx").exists():
            fail(f"missing {name}.onnx in {model_dir}")
    return meta


def ort_file(name, ort_dir, cache_dir):
    """Return the bytes of one ONNX Runtime Web file: local folder, cache, or download."""
    expected = ORT_FILES[name]
    candidates = ([ort_dir / name] if ort_dir else []) + [cache_dir / name]
    for path in candidates:
        if path.exists():
            data = path.read_bytes()
            if sha256(data) == expected:
                return data
            print(f"  ignoring {path}: hash does not match onnxruntime-web {ORT_VERSION}")
    for mirror in ORT_MIRRORS:
        url = mirror.format(version=ORT_VERSION, name=name)
        try:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except OSError as error:
            print(f"  download failed: {error}")
            continue
        if sha256(data) != expected:
            print("  download has the wrong hash, skipping this mirror")
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / name).write_bytes(data)
        return data
    fail(f"could not get {name} from onnxruntime-web {ORT_VERSION}. Download it by hand into {cache_dir}/ "
         f"(see {ORT_MIRRORS[0].format(version=ORT_VERSION, name=name)}) or pass --ort-dir.")


def lexicon_text():
    """First CMU pronunciation of each word the frontend can look up (same parsing as pivo.text.lexicon)."""
    raw = (ROOT / "src/pivo/resources/cmudict.dict").read_text(encoding="utf-8")
    first = {}
    for line in raw.splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        if word not in first and re.fullmatch(r"[a-z]+(?:'[a-z]+)*", word):
            first[word] = " ".join(parts[1:])
    return "".join(f"{word} {phones}\n" for word, phones in first.items())


def licences(model_root):
    sections = [("Voxal Studio / PiVo-KN-1.4M", model_root / "LICENSE"), ("NOTICE", model_root / "NOTICE"),
                ("Third-party components", model_root / "THIRD_PARTY.md")]
    sections += [(p.stem, p) for p in sorted((model_root / "LICENSES").glob("*")) if p.is_file()]
    sections += [("CMU Pronouncing Dictionary", ROOT / "src/pivo/resources/CMUDICT_LICENSE")]
    sections += [(p.stem, p) for p in sorted((WEB / "licenses").glob("*.txt"))]
    out = []
    for title, path in sections:
        if path.exists():
            out.append(f"{'=' * 78}\n{title}\n{'=' * 78}\n\n{path.read_text(encoding='utf-8').strip()}\n")
        else:
            print(f"  note: {path} not found, skipped in licenses.txt")
    return "\n\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/PiVo-KN-1.4M/onnx")
    parser.add_argument("--ort-dir", type=Path, help="folder with onnxruntime-web dist files (skips the download)")
    parser.add_argument("--out", type=Path, default=WEB / "dist")
    args = parser.parse_args()
    started = time.time()
    model_dir, out = args.model_dir.resolve(), args.out.resolve()

    print(f"model:  {model_dir}")
    meta = check_model(model_dir)
    if out.exists():
        if not (out / MARKER).exists() and any(out.iterdir()):
            fail(f"{out} exists and was not made by this script; choose another --out")
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "vendor").mkdir()
    (out / "lexicon").mkdir()
    (out / MARKER).write_text("made by web/build.py; deleted and rebuilt on every build\n")

    files = {}  # manifest entries for the large files the worker fetches

    def put(rel, data, key=None):
        path = out / rel
        path.write_bytes(data)
        if key:
            files[key] = {"path": rel, "size": len(data), "hash": sha256(data)[:16]}

    for name in APP_FILES:
        put(name, (WEB / "app" / name).read_bytes())
    for name in MODEL_FILES:
        put(f"model/{name}.onnx", (model_dir / f"{name}.onnx").read_bytes(), key=name)
    print(f"onnxruntime-web {ORT_VERSION}:")
    cache = WEB / ".cache" / f"onnxruntime-web-{ORT_VERSION}"
    for name in ORT_FILES:
        put(f"vendor/{name}", ort_file(name, args.ort_dir, cache), key="wasm" if name.endswith(".wasm") else None)
    put("lexicon/cmudict.txt", lexicon_text().encode("utf-8"), key="lexicon")
    put("licenses.txt", licences(model_dir.parent).encode("utf-8"))

    version = sha256(b"".join(sha256((out / p).read_bytes()).encode()
                              for p in sorted(str(q.relative_to(out)) for q in out.rglob("*") if q.is_file())))[:12]
    manifest = {"version": version, "model": meta.get("parameters", {}).get("provisional_name", "PiVo-KN-1.4M"),
                "voices": [{"id": v, "index": meta["voices"][v], "name": name, "language": lang}
                           for v, name, lang in SITE_VOICES],
                "sample_rate": 16000,
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}
    put("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    index = (out / "index.html").read_text(encoding="utf-8").replace("__VERSION__", version)
    put("index.html", index.encode("utf-8"))

    raw_total = gz_total = 0
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in COMPRESSIBLE):
        data = path.read_bytes()
        packed = gzip.compress(data, compresslevel=9, mtime=0)
        raw_total += len(data)
        if len(packed) < 0.95 * len(data):
            path.with_name(path.name + ".gz").write_bytes(packed)
            gz_total += len(packed)
        else:
            gz_total += len(data)
    print(f"\nbuilt {out}  (version {version}, {time.time() - started:.1f} s)")
    print(f"  {raw_total / 1e6:.1f} MB of files, {gz_total / 1e6:.1f} MB over the network with gzip")
    first_visit = sum(f["size"] for k, f in files.items() if k != "lexicon")
    print(f"  model + runtime: {first_visit / 1e6:.1f} MB before compression (downloaded once, then cached)")


if __name__ == "__main__":
    main()
