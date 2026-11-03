"""End-to-end test of the Voxal Studio site in a real browser (Chromium via Playwright).

Serves web/dist, loads the page, generates speech, downloads the WAV and compares it with the
Python ONNX runtime on the same utterances. Also checks stop, empty input, emoji, code-mixed
text, speed, mobile layout and that the browser console stays clean.

    pip install playwright uvicorn && playwright install chromium
    python web/build.py
    python web/tests/e2e.py --model-dir models/PiVo-KN-1.4M/onnx
"""

import argparse
import io
import json
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WEB))

NUMBERS = "ಪರೀಕ್ಷೆ 03-11-2026ರಂದು ಬೆಳಿಗ್ಗೆ 10:30ಕ್ಕೆ ಆರಂಭವಾಗಲಿದೆ. ಶುಲ್ಕ ₹1,250 ಮಾತ್ರ, ಸಹಾಯಕ್ಕೆ 98450 12345 ಗೆ ಕರೆ ಮಾಡಿ."
MIXED = "ನನ್ನ phone ಬ್ಯಾಟರಿ 20% ಇದೆ, ಆದ್ದರಿಂದ meeting ಗೆ ಸ್ವಲ್ಪ late ಆಗಬಹುದು. Hubballi ಗೆ ಹೋಗುತ್ತೇನೆ 😀."
ENGLISH = ("The meeting in Bengaluru is at 10:30 AM on 03/11/2026, and the ticket costs ₹1,25,000. "
           "Call 98450 12345 if you need help.")
LONG = ("ಕನ್ನಡ ನಾಡು ಸಾಹಿತ್ಯ, ಸಂಗೀತ ಮತ್ತು ಕಲೆಗಳ ತವರು. ಇಲ್ಲಿನ ಜನರು ಸ್ನೇಹಪರರು ಮತ್ತು ಅತಿಥಿಗಳನ್ನು ಆದರದಿಂದ ಸ್ವಾಗತಿಸುತ್ತಾರೆ. " * 10)[:1000]


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


def python_reference(model_dir, dist, text, speed=1.0, voice="kn_m"):
    """Same utterances through the Python ONNX runtime, joined with the web app's pauses."""
    from pivo import kannada
    from pivo.runtime import Runtime
    out = subprocess.run(["node", str(WEB / "tests/chunks.mjs"), str(dist / "lexicon/cmudict.txt"), text],
                         capture_output=True, text=True, check=True, encoding="utf-8")
    chunks = json.loads(out.stdout)["chunks"]
    runtime = Runtime(model_dir)

    class Front:
        pass

    pieces = []
    for i, ids in enumerate(chunks):
        if i:
            gap = 0.18 if chunks[i - 1][-2] in (4, 6, 7) else 0.08
            pieces.append(np.zeros(round(gap * 16000), dtype=np.float32))
        front = Front()
        front.token_ids, front.symbols = tuple(ids), tuple(kannada.SYMBOLS[t] for t in ids)
        runtime.phonemize = lambda _text, overrides=None, f=front: f
        pieces.append(runtime.synthesize("-", voice=voice, speed=speed))
    audio = np.concatenate(pieces)
    return (audio.clip(-1, 1) * 32767).round().astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/PiVo-KN-1.4M/onnx")
    parser.add_argument("--dist", type=Path, default=WEB / "dist")
    parser.add_argument("--out", type=Path, default=WEB / "tests/out")
    parser.add_argument("--url", help="test an already running site instead of serving web/dist")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    url = args.url
    if not url:
        port = free_port()
        serve(args.dist, port)
        url = f"http://127.0.0.1:{port}/"
    problems, results = [], {}

    def check(ok, message):
        print(("  ok   " if ok else "  FAIL ") + message)
        if not ok:
            problems.append(message)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        context = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page = context.new_page()
        console = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        def generate(text, speed=None):
            page.fill("#text", text)
            if speed is not None:
                page.eval_on_selector("#speed", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", str(speed))
            t0 = time.time()
            page.click("#generate-btn")
            # (no wait_for_function: the site's CSP blocks the eval it uses, which is what we want)
            page.wait_for_selector("#generate-btn:not(.is-busy)", timeout=120000)
            page.wait_for_selector("#download-btn:not([disabled])", timeout=5000)
            return time.time() - t0

        def download():
            with page.expect_download() as info:
                page.click("#download-btn")
            path = args.out / info.value.suggested_filename
            info.value.save_as(path)
            return read_wav(path.read_bytes()), info.value.suggested_filename

        print(f"site: {url}")
        t0 = time.time()
        page.goto(url)
        page.wait_for_selector("#status-title:has-text('Ready on your device')", timeout=120000)
        results["load_seconds"] = round(time.time() - t0, 2)
        check(True, f"model loaded in {results['load_seconds']} s")
        page.screenshot(path=args.out / "desktop-ready.png", full_page=True)
        check(page.input_value("#voice-select") == "kn" and page.inner_text("#voice-name") == "Kannada",
              "Kannada is the default voice and is labelled 'Kannada'")

        # 1. numbers sentence: browser WAV vs Python runtime
        seconds = generate(NUMBERS)
        audio, name = download()
        numbers_audio = audio
        ref = python_reference(args.model_dir, args.dist, NUMBERS)
        same_len = len(audio) == len(ref)
        diff = int(np.abs(audio - ref).max()) if same_len else -1
        check(name.startswith("voxal-kannada-") and name.endswith(".wav"), f"download name {name}")
        check(same_len and diff <= 16, f"browser audio matches Python runtime ({len(audio)} vs {len(ref)} samples, max diff {diff} LSB)")
        results["numbers"] = {"audio_s": round(len(audio) / 16000, 2), "wall_s": round(seconds, 2), "max_diff_lsb": diff}
        read_as = page.text_content("#read-as-text")
        check("ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರರಂದು" in read_as and "ಒಂಬತ್ತು ಎಂಟು ನಾಲ್ಕು" in read_as, "numbers, date and phone read in Kannada")
        stats = page.inner_text("#stats")
        check("faster than real time" in stats, f"stats line: {stats}")
        page.screenshot(path=args.out / "desktop-generated.png", full_page=True)

        # 2. audio plays while it is generated; pause / play / seek work afterwards
        is_playing = lambda: page.eval_on_selector("#play-btn", "b => b.classList.contains('is-playing')")  # noqa: E731
        check(is_playing(), f"audio starts playing during generation (position {page.inner_text('#time-now')})")
        page.click("#play-btn")
        paused_at = page.inner_text("#time-now")
        page.wait_for_timeout(1200)
        check(not is_playing() and page.inner_text("#time-now") == paused_at, f"pause holds the position at {paused_at}")
        page.click("#play-btn")
        page.wait_for_timeout(1200)
        check(is_playing() and page.inner_text("#time-now") != paused_at, f"play resumes ({page.inner_text('#time-now')})")
        page.click("#play-btn")
        box = page.locator("#wave").bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
        page.wait_for_timeout(200)
        pos = page.inner_text("#time-now")
        check(pos != "0:00", f"seek by clicking the waveform -> {pos}")

        # 3. code-mixed text with an unknown name and an emoji
        generate(MIXED)
        audio, _ = download()
        ref = python_reference(args.model_dir, args.dist, MIXED)
        diff = int(np.abs(audio - ref).max()) if len(audio) == len(ref) else -1
        check(0 <= diff <= 16, f"code-mixed audio matches Python (max diff {diff} LSB)")
        note = page.inner_text("#read-as-note")
        check("😀" in note and "hubballi" in note, f"skipped / guessed words are reported: {note!r}")

        # 4. speed 1.5 is shorter
        generate(NUMBERS, speed=1.5)
        fast, _ = download()
        check(len(fast) < len(numbers_audio) * 0.8, f"speed 1.5x gives shorter audio ({len(fast) / 16000:.2f} s)")
        page.eval_on_selector("#speed", "el => { el.value = 1; el.dispatchEvent(new Event('input')); }")

        # 5. stop in the middle of a long text
        page.fill("#text", LONG)
        page.click("#generate-btn")
        page.wait_for_selector("#read-as:not([hidden])", timeout=30000)
        page.wait_for_timeout(1500)
        page.click("#generate-btn")
        page.wait_for_selector("#generate-btn:not(.is-busy)", timeout=10000)
        check("Stopped" in page.inner_text("#stats"), "stop button stops generation")
        # and the next generation still works after a stop
        generate("ನಮಸ್ಕಾರ.")
        check("of audio" in page.inner_text("#stats"), "generation works again after stopping")

        # 6. full 1,000-character text
        seconds = generate(LONG)
        audio, _ = download()
        results["long"] = {"chars": len(LONG), "audio_s": round(len(audio) / 16000, 2), "wall_s": round(seconds, 2)}
        check(len(audio) / 16000 > 40, f"1,000 characters -> {len(audio) / 16000:.1f} s of audio in {seconds:.1f} s")

        # 7. empty and unreadable input
        page.fill("#text", "   ")
        page.click("#generate-btn")
        check(page.is_visible("#error") and "Type some" in page.inner_text("#error"), "empty text shows a hint")
        page.fill("#text", "😀 ☺")
        page.click("#generate-btn")
        page.wait_for_selector("#error:has-text('nothing to read')", timeout=10000)
        check(True, "emoji-only text shows 'nothing to read'")

        # 8. English voice: mismatch tip, switch, generate, compare with Python (en_m)
        page.fill("#text", ENGLISH)
        page.dispatch_event("#text", "input")
        check(page.is_visible("#lang-tip") and "English" in page.inner_text("#lang-tip"), "tip suggests the English voice for English text")
        page.click("#lang-tip-btn")
        check(page.is_checked('input[name="lang"][value="en"]') and page.inner_text("#voice-name") == "English"
              and page.input_value("#voice-select") == "en" and page.is_hidden("#lang-tip"), "switched to the English voice")
        seconds = generate(ENGLISH)
        audio, name = download()
        ref = python_reference(args.model_dir, args.dist, ENGLISH, voice="en_m")
        diff = int(np.abs(audio - ref).max()) if len(audio) == len(ref) else -1
        check(name.startswith("voxal-english-"), f"English download name {name}")
        check(0 <= diff <= 16, f"English audio (en_m) matches Python runtime (max diff {diff} LSB, {len(audio) / 16000:.1f} s in {seconds:.1f} s)")
        read_as = page.text_content("#read-as-text")
        check("one lakh twenty five thousand rupees" in read_as and "third November" in read_as, f"Indian English reading: {read_as[:90]}…")
        page.screenshot(path=args.out / "desktop-english.png", full_page=True)
        page.reload()
        page.wait_for_selector("#status-title:has-text('Ready on your device')", timeout=120000)
        check(page.is_checked('input[name="lang"][value="en"]'), "the chosen language is remembered after a reload")
        page.select_option("#voice-select", "kn")
        check(page.is_checked('input[name="lang"][value="kn"]') and page.inner_text("#voice-name") == "Kannada", "chip menu switches back to Kannada")

        # 9. dialogs
        page.click("text=Writing tips")
        check(page.is_visible("#help-dialog"), "help dialog opens")
        page.keyboard.press("Escape")

        # 10. mobile layout
        mobile = browser.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
        mp = mobile.new_page()
        mp.on("pageerror", lambda e: console.append(f"pageerror (mobile): {e}"))
        mp.goto(url)
        mp.wait_for_selector("#status-title:has-text('Ready on your device')", timeout=120000)
        overflow = mp.evaluate("document.documentElement.scrollWidth")
        check(overflow <= 375, f"no horizontal scroll on a 375 px phone (scrollWidth {overflow})")
        mp.click("#generate-btn")
        mp.wait_for_selector("#download-btn:not([disabled])", timeout=120000)
        mp.screenshot(path=args.out / "mobile.png", full_page=True)
        check(True, "generation works on the mobile layout")

        # 11. console
        relevant = [c for c in console if "fonts.g" not in c and "ERR_TUNNEL" not in c and "net::ERR" not in c]
        check(not relevant, "no console errors" + ("" if not relevant else f": {relevant[:5]}"))
        browser.close()

    print(json.dumps(results, indent=2))
    print("PASSED" if not problems else f"FAILED: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
