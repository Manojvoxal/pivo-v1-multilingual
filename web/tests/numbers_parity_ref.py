"""Reference output for the JavaScript number normaliser (app/numbers_multi.js).

Writes number-heavy test sentences for the 10 languages with what the Python normaliser
(training/data_pipeline_multi/pivo_multi/numnorm.py) makes of them: hand-written cases for
every rule, every cardinal 0..10,000 plus random big numbers, and random sentences that mix
numbers, currencies, dates, times, units, suffixes, acronyms, symbols and native digits.

    python web/tests/numbers_parity_ref.py ref.json [--n 3000]
    node web/tests/numbers_parity.mjs ref.json
"""

import argparse
import ast
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "training" / "data_pipeline_multi"))

from pivo_multi import numnorm  # noqa: E402

CASES = [
    "0", "7", "10", "19", "99", "100", "101", "110", "199", "200", "999", "1000", "1001", "1100", "1999", "2000",
    "2026", "5000", "9999", "10000", "11000", "15000", "20000", "21000", "25000", "50000", "99999", "100000",
    "100001", "125000", "1,25,000", "1,250,000", "10,00,000", "1100000", "9999999", "1,00,00,000", "10000000",
    "1000000000", "99,99,99,99,999", "3.14", "0.5", ".5", "12.50", "-5", "−12.5", "(-3)", "x-5",
    "₹500", "₹1", "₹1,25,000.50", "Rs. 99.99", "Rs 5 lakh", "INR 2.5 crore", "₹5 million", "₹1.234567 lakh",
    "$3.20", "$1", "€5", "£2.5", "US$ 10", "₹ 10k", "₹2 cr",
    "12.5%", "100 %", "10-20", "10 – 20", "10-15 km", "5-10%", "2020-21",
    "5 km", "1 km", "5km", "2.5 kg", "100 m", "30°C", "-5 °C", "45°", "4 GB", "2 hrs", "1 min", "60 km/h", "5 mg",
    "1st", "2nd", "3rd", "4th", "21st", "100th", "10:30", "7:05 pm", "18:45", "00:15", "12:00 AM", "12 PM",
    "5 PM", "11.30 a.m.", "9:15:30", "15/08/1947", "15-08-2026", "15.08.2026", "31/12/1999", "32/13/2020",
    "2026-11-04", "9876543210", "+91 98765 43210", "+91-98765-43210", "080-23456789", "0471 2345678",
    "PIN 560001", "PIN: 560 001", "BJP", "UNESCO", "ISRO", "A & B", "5 + 3 = 8", "a@b.com", "100%", "°",
    "0123", "007", "1234567", "12345678901234567890", "Dr. Kalam", "Mr. Rao", "No. 5", "India vs. Pakistan",
    "August 1947", "15 August 1947", "May 2026", "1947", "1100", "1999", "2,026",
]

SCRIPT_BASE = numnorm.SCRIPT_BASE


def semiotic_sentences():
    """The one-per-class sentences of training/data_pipeline_multi/tests/test_semiotic.py."""
    tree = ast.parse((ROOT / "training" / "data_pipeline_multi" / "tests" / "test_semiotic.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SENTENCES")


def block_letters(lang):
    base = SCRIPT_BASE[lang]
    import unicodedata
    out = []
    for c in range(base, base + 0x80):
        ch = chr(c)
        name = unicodedata.name(ch, "")
        if name and "DIGIT" not in name and c not in (0x964, 0x965):
            out.append(ch)
    return out


def native_digits(s, lang, rng):
    base = SCRIPT_BASE[lang]
    if rng.random() < 0.5:
        return s
    return "".join(chr(base + 0x66 + int(c)) if c.isdigit() and c.isascii() else c for c in s)


def random_number(rng):
    r = rng.random()
    if r < 0.3:
        return str(rng.randint(0, 100))
    if r < 0.5:
        return str(rng.randint(0, 10 ** rng.randint(2, 9)))
    if r < 0.6:
        n = rng.randint(1000, 10 ** 12)
        s = str(n)
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        return ",".join(([head] if head else []) + groups + [tail])
    if r < 0.75:
        return f"{rng.randint(0, 999)}.{rng.randint(0, 99):0{rng.randint(1, 2)}d}"
    if r < 0.85:
        return str(rng.randint(1100, 2099))
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 18)))


def random_piece(lang, rng, letters):
    D = numnorm.DATA[lang]
    n = random_number(rng)
    kind = rng.randrange(18)
    if kind == 0:
        return rng.choice(["₹", "Rs ", "Rs.", "$", "€", "£", "INR "] + D["currency_abbr"]) + n + rng.choice(
            ["", "", " lakh", " crore", " k", " million", " " + rng.choice(D["scales"]["lakh"]["forms"])])
    if kind == 1:
        return n + rng.choice(["%", " %"])
    if kind == 2:
        return f"{rng.randint(0, 30)}{rng.choice(['-', '–', ' - '])}{rng.randint(0, 99)}" + rng.choice(
            ["", " km", "%", ""])
    if kind == 3:
        return n + rng.choice(["", " "]) + rng.choice(list(D["measures"]))
    if kind == 4:
        return f"{rng.randint(0, 25)}:{rng.randint(0, 61):02d}" + rng.choice(["", " am", " PM", " p.m.", ""])
    if kind == 5:
        return f"{rng.randint(0, 33)}{rng.choice('/.-')}{rng.randint(0, 13)}{rng.choice('/.-')}{rng.randint(1000, 2100)}"
    if kind == 6:
        return rng.choice(["+91 ", "", "+91-"]) + str(rng.randint(6000000000, 9999999999))
    if kind == 7:
        return str(rng.randint(1, 120)) + rng.choice(["st", "nd", "rd", "th"])
    if kind == 8:
        return n + rng.choice(["", "-"]) + rng.choice(list(D["ordinal"]["suffixes"]) + list(D.get("day_suffixes", {})))
    if kind == 9:
        suffix = "".join(rng.choice(letters) for _ in range(rng.randint(1, 4)))
        return n + rng.choice(["", "-"]) + suffix
    if kind == 10:
        return rng.choice(D["year_before"] + [""]) + " " + str(rng.randint(1000, 2100)) + " " + rng.choice(
            D["year_after"] + [""])
    if kind == 11:
        return rng.choice(D["months"]) + rng.choice([" ", ", "]) + str(rng.randint(1000, 2100))
    if kind == 12:
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(1, 7)))
    if kind == 13:
        return rng.choice(["&", "+", "=", "@", "%", "°", "-", "−", "#"])
    if kind == 14:
        return rng.choice(list(D["abbreviations"])) + rng.choice(["", " "]) + rng.choice([n, ""])
    if kind == 15:
        return rng.choice(D["letters"]) + ". " + rng.choice(D["letters"]) + "."
    if kind == 16:
        return "-" + n
    return n


def random_sentence(lang, rng, letters):
    parts = []
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.55:
            parts.append(random_piece(lang, rng, letters))
        elif r < 0.85:
            parts.append("".join(rng.choice(letters) for _ in range(rng.randint(1, 6))))
        elif r < 0.95:
            parts.append(rng.choice(["the", "India", "OK", "km", "Rs", "Dr.", "😀", "(", ")", ".", ",", "।", "?"]))
        else:
            parts.append(rng.choice([" ", "\t", "‌", "‍", "  "]))
    return native_digits(rng.choice([" ", "", " "]).join(parts), lang, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3000, help="random sentences per language")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for lang in numnorm.LANGS:
        letters = block_letters(lang)
        texts = list(CASES) + semiotic_sentences() + [str(i) for i in range(10001)]
        texts += [str(rng.randint(0, 10 ** 15 - 1)) for _ in range(2000)]
        texts += [native_digits(c, lang, rng) for c in CASES]
        texts += [random_sentence(lang, rng, letters) for _ in range(a.n)]
        for t in texts:
            rows.append({"lang": lang, "text": t, "out": numnorm.normalize_numbers(t, lang)})
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} cases to {a.out}")


if __name__ == "__main__":
    main()
