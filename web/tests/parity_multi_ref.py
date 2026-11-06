"""Reference token ids for the JavaScript multilingual frontend (app/frontend_multi.js).

Generates random words and sentences in every supported script (letters, vowel signs,
virama, nukta, anusvara, ZWJ/ZWNJ, punctuation, English words, other-script words) and
records what the Python training frontend (training/data_pipeline_multi/pivo_multi) makes
of them. With --manifest it also adds real sentences (e.g. the dataset's val.jsonl) with the
token ids training used.

    python web/tests/parity_multi_ref.py out.json --cmudict src/pivo/resources/cmudict.dict \
        [--manifest training/data_pipeline_multi/val.jsonl] [--n 4000]
"""

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "training" / "data_pipeline_multi"))

from pivo_multi import brahmic, english, frontend  # noqa: E402
from pivo_multi.langs import LANG  # noqa: E402
from pivo_multi.scripts import BRAHMIC_BASE  # noqa: E402

LANGS = ["en", "kn", "hi", "te", "ta", "bn", "mr", "gu", "pa", "ml", "or", "as"]
WORD_LANGS = [c for c, v in LANG.items() if v["inherent"] is not None and v["script"] in BRAHMIC_BASE]


def block_chars(script):
    base = BRAHMIC_BASE[script]
    out = [chr(c) for c in range(base, base + 0x80) if unicodedata.name(chr(c), "")]
    letters = [c for c in out if "LETTER" in unicodedata.name(c)]
    signs = [c for c in out if "SIGN" in unicodedata.name(c)]
    return out, letters, signs


def random_word(rng, script):
    allc, letters, signs = block_chars(script)
    n = rng.randint(1, 7)
    w = []
    for _ in range(n):
        r = rng.random()
        if r < 0.55:
            w.append(rng.choice(letters))
        elif r < 0.85:
            w.append(rng.choice(signs))
        elif r < 0.95:
            w.append(rng.choice(allc))
        else:
            w.append(rng.choice(["‌", "‍"]))
    if rng.random() < 0.3:
        w.insert(0, rng.choice(letters))
    return "".join(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cmudict", required=True)
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--n", type=int, default=3000, help="random sentences per language")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    english.set_cmudict_path(a.cmudict)
    rng = random.Random(a.seed)
    cmu = [w for w in english.lexicon() if w.isalpha() and w.isascii()]
    punct = [".", ",", "?", "!", ";", ":", "।", "॥", "…", "—", "-", "'", '"', "(", ")", "“", "”", " ", "  "]
    scripts = {c: LANG[c]["script"] for c in LANGS if c != "en"}

    words = []
    for lang in WORD_LANGS:
        script = LANG[lang]["script"]
        for _ in range(a.n):
            w = random_word(rng, script)
            words.append({"lang": lang, "word": w, "phones": brahmic.word_to_phones(w, lang)})

    sentences, kept, failed = [], 0, 0
    for lang in LANGS:
        for _ in range(a.n):
            parts = []
            for _ in range(rng.randint(1, 9)):
                r = rng.random()
                if lang == "en" or r < 0.12:
                    w = rng.choice(cmu)
                    w = w.upper() if rng.random() < 0.05 else (w.capitalize() if rng.random() < 0.3 else w)
                elif r < 0.17:
                    w = random_word(rng, rng.choice(list(scripts.values())))
                else:
                    w = random_word(rng, scripts[lang])
                parts.append(w)
                if rng.random() < 0.35:
                    parts.append(rng.choice(punct))
            text = " ".join(parts) if rng.random() < 0.8 else "".join(parts)
            try:
                ids = list(frontend.phonemize(text, lang, strict=False).token_ids)
                kept += 1
            except Exception:  # noqa: BLE001 - the JS side is lenient; compare only clean cases
                ids = None
                failed += 1
            sentences.append({"lang": lang, "text": text, "ids": ids})

    real = []
    for path in a.manifest:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                real.append({"lang": r["lang"], "text": r["text"], "ids": r["token_ids"]})
    json.dump({"words": words, "sentences": sentences, "real": real}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"{len(words)} words, {len(sentences)} sentences ({kept} readable by Python, {failed} rejected), "
          f"{len(real)} real sentences -> {a.out}")


if __name__ == "__main__":
    main()
