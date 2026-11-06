"""End-to-end test of the multilingual Voxal Studio site in a real browser (Chromium, Playwright).

Serves web/dist, picks every language and voice through the drop-downs, generates speech,
downloads the WAV and compares it with an independent Python ONNX Runtime implementation of the
same steps (encoder with language id -> durations -> streaming decoder -> FARGAN vocoder).
Also checks numbers / code-mixing (the page's "read as" text must equal what the Python
normaliser pivo_multi/numnorm.py gives), the language hint, the download name and the console.

    pip install playwright uvicorn onnxruntime numpy && playwright install chromium
    python web/build.py
    python web/tests/e2e_multi.py --model-dir models/multi_w128_refine/onnx [--langs hi,kn] [--both-voices]
"""

import argparse
import io
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

WEB = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parent / "training" / "data_pipeline_multi"))
LANGS = ["kn", "en", "hi", "te", "ta", "bn", "mr", "gu", "pa", "ml", "or", "as"]
NAMES = {"kn": "kannada", "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "bn": "bengali",
         "mr": "marathi", "gu": "gujarati", "pa": "punjabi", "ml": "malayalam", "or": "odia", "as": "assamese"}
EXTRA = {"hi": "मेरा phone कहाँ है? कीमत ₹1,250 है और 72% छूट 10:30 बजे तक है।",
         "kn": "ನನ್ನ phone ಬ್ಯಾಟರಿ 20% ಇದೆ, ಶುಲ್ಕ ₹1,250 ಮಾತ್ರ."}
# number-heavy text per language: the page must read it exactly as pivo_multi/numnorm.py does
NUMBERS = {
    "hi": "15 अगस्त 1947 को भारत आज़ाद हुआ। 2024 में GDP 6.5% बढ़ी। टिकट ₹1,250 का है, कॉल 9876543210।",
    "mr": "१५ ऑगस्ट १९४७ रोजी भारत स्वतंत्र झाला. २०२४ मध्ये ६.५% वाढ झाली. तिकीट रु. १,२५० आहे.",
    "gu": "૧૫ ઓગસ્ટ ૧૯૪૭ના રોજ ભારત આઝાદ થયું. ૨૦૨૪માં ૬.૫% વધારો. ટિકિટ રૂ. ૧,૨૫૦ની છે.",
    "pa": "੧੫ ਅਗਸਤ ੧੯੪੭ ਨੂੰ ਭਾਰਤ ਆਜ਼ਾਦ ਹੋਇਆ। ੨੦੨੪ ਵਿੱਚ ੬.੫% ਵਾਧਾ। ਟਿਕਟ ₹੧,੨੫੦ ਦੀ ਹੈ।",
    "bn": "১৯৪৭ সালের ১৫ই আগস্ট ভারত স্বাধীন হয়। ২০২৪ সালে ৬.৫% বৃদ্ধি। সকাল ১০:৩০-এ সভা।",
    "or": "୧୯୪୭ ମସିହା ଅଗଷ୍ଟ ୧୫ରେ ଭାରତ ସ୍ୱାଧୀନ ହେଲା। ୨୦୨୪ରେ ୬.୫% ବୃଦ୍ଧି। ଟିକେଟ ₹୧,୨୫୦।",
    "as": "১৯৪৭ চনৰ ১৫ আগষ্টত ভাৰত স্বাধীন হয়। ২০২৪ চনত ৬.৫% বৃদ্ধি। টিকটৰ দাম ₹১,২৫০।",
    "ta": "1947ஆம் ஆண்டு ஆகஸ்ட் 15 அன்று இந்தியா சுதந்திரம் பெற்றது. 2024இல் 6.5% வளர்ச்சி.",
    "te": "1947 ఆగస్టు 15న భారతదేశానికి స్వాతంత్ర్యం వచ్చింది. 2024లో 6.5% వృద్ధి. 5వ తరగతి.",
    "ml": "1947 ഓഗസ്റ്റ് 15-ന് ഇന്ത്യ സ്വതന്ത്രമായി. 2024-ൽ 6.5% വളർച്ച. 5-ാം ക്ലാസ്.",
}
STATE = [("memory", 256), ("gru1", 160), ("gru2", 128), ("gru3", 128), ("frame_state", 164)]
CACHE = ["key0", "value0", "key1", "value1"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(dist, port):
    import uvicorn
    from server import StaticSite
    server = uvicorn.Server(uvicorn.Config(StaticSite(dist), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int32)


def js_round(x):
    return math.floor(x + 0.5)


class Reference:
    """The steps of web/app/engine.js, written again in Python on onnxruntime (CPU)."""

    def __init__(self, model_dir):
        import onnxruntime as ort
        self.s = {n: ort.InferenceSession(str(Path(model_dir) / f"{n}.onnx"), providers=["CPUExecutionProvider"])
                  for n in ("encoder", "decoder", "conditioning", "vocoder_step")}
        meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
        self.head_dim, self.kv_heads = meta["config"]["head_dim"], meta["config"]["kv_heads"]
        self.voices, self.langs = meta["voices"], meta["language_ids"]

    def features(self, ids, voice, lang, speed=1.0):
        i64 = lambda a: np.array(a, dtype=np.int64)
        enc, logd = self.s["encoder"].run(None, {"tokens": i64([ids]), "voice": i64([voice]), "lang": i64([lang]),
                                                 "lengths": i64([len(ids)])})
        enc, logd = enc[0], logd[0]
        lmax = math.log1p(200)
        dur = []
        for i, tok in enumerate(ids):
            if tok <= 2:
                dur.append(0)
                continue
            d = js_round(math.expm1(min(max(float(logd[i]), 0.0), lmax)) / speed)
            d = min(max(d, 0), 200)
            dur.append(max(d, 1) if tok >= 10 else d)
        F = sum(dur)
        assert F <= 3000, "test sentences must fit one utterance"
        rows = []
        for i, n in enumerate(dur):
            for k in range(n):
                frac = (k + 0.5) / n
                rows.append(np.concatenate([enc[i], np.array([2 * frac - 1, math.sin(math.pi * frac),
                                                              math.cos(math.pi * frac), math.log1p(n) / lmax], np.float32)]))
        expanded = np.stack(rows).astype(np.float32)[None]
        cache = {c: np.zeros((1, self.kv_heads, 0, self.head_dim), np.float32) for c in CACHE}
        feats = []
        for p in range(0, F, 16):
            r = self.s["decoder"].run(None, {"expanded": expanded[:, p:p + 16], "position": np.array(p, np.int64), **cache})
            feats.append(r[0][0])
            cache = dict(zip(CACHE, r[1:]))
        return np.concatenate(feats).astype(np.float32)

    def vocode(self, feats):
        F = len(feats)
        state = {n: np.zeros((1, w), np.float32) for n, w in STATE}
        de, out = 0.0, []
        period = lambda v: min(255, max(32, js_round(256 / 2 ** (min(10, max(-10, float(v))) + 1.5))))
        for i in range(F):
            rows = [max(i - 1, 0), i, min(i + 1, F - 1)]
            ctx = np.stack([feats[r] for r in rows])[None].astype(np.float32)
            per = np.array([[period(feats[r][18]) for r in rows]], np.int64)
            cond = self.s["conditioning"].run(None, {"features": ctx, "periods": per})[0].reshape(-1)
            for sub in range(4):
                r = self.s["vocoder_step"].run(None, {"conditioning": cond[sub * 80:sub * 80 + 80][None],
                                                      "period": per[0, 1:2], **state})
                names = [o.name for o in self.s["vocoder_step"].get_outputs()]
                res = dict(zip(names, r))
                state = {n: res[n + "_out"] for n, _ in STATE}
                for x in res["excitation"].reshape(-1):
                    de = float(x) + 0.85 * de
                    out.append(de)
        return np.array(out, np.float32)

    def speak(self, chunks, voice_id, lang):
        pieces = []
        for i, ids in enumerate(chunks):
            if i:
                gap = 0.18 if chunks[i - 1][-2] in (4, 6, 7) else 0.08
                pieces.append(np.zeros(round(gap * 16000), np.float32))
            pieces.append(self.vocode(self.features(ids, self.voices[voice_id], self.langs[lang])))
        pcm = np.concatenate(pieces)
        return np.round(np.clip(pcm, -1, 1) * 32767).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(WEB.parent / "models/multi_w128_refine/onnx"))
    ap.add_argument("--dist", default=str(WEB / "dist"))
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--both-voices", action="store_true")
    ap.add_argument("--tol", type=int, default=8, help="per-frame max |difference| (16-bit) counted as equal")
    ap.add_argument("--max-frac", type=float, default=0.05,
                    help="share of 10 ms frames allowed to differ by more than --tol (the recurrent vocoder can "
                         "amplify WebAssembly-vs-native float rounding for a few frames)")
    ap.add_argument("--out", default=None, help="folder to keep the downloaded WAVs")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    dist = Path(a.dist)
    ref = Reference(a.model_dir)
    samples = dict(re.findall(r'^  (\w\w): \{ name: "\w+"[\s\S]*?text: "([^"]+)" \}',
                              (WEB / "app/main.js").read_text(encoding="utf-8"), re.M))
    port = free_port()
    serve(dist, port)
    failures = []

    def check(ok, message):
        print(("PASS " if ok else "FAIL ") + message, flush=True)
        if not ok:
            failures.append(message)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True).new_page()
        console = []
        # web fonts come from Google; offline / firewalled test machines cannot reach them, which is fine
        font = lambda m: "fonts.g" in (m.location or {}).get("url", "")
        page.on("console", lambda m: m.type in ("error", "warning") and not font(m) and console.append(m.text))
        page.on("pageerror", lambda e: console.append(str(e)))
        t0 = time.time()
        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_selector(".status-dot.ready", timeout=180000)
        check(True, f"model loaded in {time.time() - t0:.1f} s")
        n_opts = page.eval_on_selector("#lang-chip", "s => s.options.length")
        check(n_opts == 12, f"language drop-down has {n_opts} languages")

        def generate(lang, gender, text):
            page.select_option("#lang-chip", lang)
            page.select_option("#voice-chip", gender)
            page.fill("#text", text)
            page.click("#generate-btn")
            page.wait_for_selector("#download-btn:not([disabled])", timeout=300000)
            page.wait_for_function("!document.querySelector('#generate-btn').classList.contains('is-busy')", timeout=300000)
            with page.expect_download() as info:
                page.click("#download-btn")
            data = Path(info.value.path()).read_bytes()
            if a.out:
                Path(a.out).mkdir(parents=True, exist_ok=True)
                (Path(a.out) / info.value.suggested_filename).write_bytes(data)
            return read_wav(data), info.value.suggested_filename

        for lang in a.langs.split(","):
            for gender in (("f", "m") if a.both_voices else ("m" if LANGS.index(lang) % 2 else "f",)):
                text = samples[lang]
                t1 = time.time()
                audio, name = generate(lang, gender, text)
                took = time.time() - t1
                synced = page.evaluate("[document.querySelector('#lang-select').value, document.querySelector('#voice-select').value]")
                check(synced == [lang, gender], f"{lang}: settings drop-downs follow the chips {synced}")
                gname = {"f": "female", "m": "male"}[gender]
                check(name.startswith(f"voxal-{NAMES[lang]}-{gname}-"), f"{lang}_{gender}: download name {name}")
                out = subprocess.run(["node", str(WEB / "tests/chunks_multi.mjs"), str(dist / "lexicon/cmudict.txt"), lang, text],
                                     capture_output=True, text=True, check=True, encoding="utf-8")
                expected = ref.speak(json.loads(out.stdout)["chunks"], f"{lang}_{gender}", lang)
                same_len = len(audio) == len(expected)
                rms = float(np.sqrt(np.mean((audio / 32768.0) ** 2)))
                if same_len:
                    d = np.abs(audio - expected)
                    per_frame = d[: len(d) // 160 * 160].reshape(-1, 160).max(1)
                    frac = float((per_frame > a.tol).mean())
                    corr = float(np.corrcoef(audio, expected)[0, 1])
                    ok = frac <= a.max_frac and corr >= 0.9999 and rms > 0.01
                    info = f"max diff {int(d.max())}, frames differing {frac:.1%}, corr {corr:.6f}"
                else:
                    ok, info = False, f"length {len(audio)} vs {len(expected)}"
                check(ok, f"{lang}_{gender}: {len(audio) / 16000:.2f} s audio in {took:.1f} s; browser vs Python: {info}")
        for lang, text in EXTRA.items():
            if lang not in a.langs.split(","):
                continue
            audio, _ = generate(lang, "f", text)
            read_as = page.text_content("#read-as-text") or ""
            check(len(audio) > 16000 and read_as and not re.search(r"[0-9]", read_as), f"{lang} numbers + English words read as: {read_as}")
        from pivo_multi import frontend as py_frontend, numnorm
        for lang, text in NUMBERS.items():
            if lang not in a.langs.split(","):
                continue
            audio, _ = generate(lang, "m", text)
            read_as = page.text_content("#read-as-text") or ""
            want = py_frontend.normalize(numnorm.normalize_numbers(text, lang))
            check(len(audio) > 16000 and read_as == want, f"{lang} numbers read as: {read_as}"
                  + ("" if read_as == want else f"\n     Python: {want}"))
        page.select_option("#lang-chip", "en")
        page.fill("#text", "नमस्ते, आप कैसे हैं? यह हिन्दी में लिखा है।")
        page.dispatch_event("#text", "input")
        tip = page.inner_text("#lang-tip") if page.is_visible("#lang-tip") else ""
        check("Hindi" in tip, f"language hint for Hindi text with English selected: {tip!r}")
        check(not console, f"browser console clean ({console[:3]})")
        browser.close()
    print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
