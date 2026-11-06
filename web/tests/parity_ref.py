"""Reference outputs for the JavaScript frontend parity test.

Builds a large, deterministic set of sentences (hand-written cases, templated numbers /
dates / times / money / phones / units in Kannada and English, and random fuzz strings),
runs them through the Python frontend (``pivo.kannada.phonemize``) and writes the results
to JSON. ``web/tests/parity.mjs`` then runs the same sentences through
``web/app/frontend.js`` and requires identical output.

    python web/tests/parity_ref.py web/tests/parity_ref.json
    node web/tests/parity.mjs web/tests/parity_ref.json web/dist/lexicon/cmudict.txt
"""

import ast
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pivo import kannada  # noqa: E402

HAND = [
    "ಈ ಶಾಲೆಯಲ್ಲಿ 250 ವಿದ್ಯಾರ್ಥಿಗಳಿದ್ದಾರೆ.", "ಜಿಲ್ಲೆಯ ಜನಸಂಖ್ಯೆ 12,34,567 ಇದೆ.", "ಒಟ್ಟು ೧೫೦೦ ಮನೆಗಳು ಹಾನಿಗೊಳಗಾಗಿವೆ.",
    "ಕನಿಷ್ಠ 1,00,000 ಜನ ಭಾಗವಹಿಸಿದ್ದರು.", "ಯೋಜನೆಗೆ 2,50,00,000 ಮಂಜೂರಾಗಿದೆ.", "2026ರಲ್ಲಿ ಚುನಾವಣೆ ನಡೆಯಲಿದೆ.",
    "ಬೆಳಿಗ್ಗೆ 10ಕ್ಕೆ ಸಭೆ ಆರಂಭವಾಗುತ್ತದೆ.", "ಮಾರ್ಚ್ 15ರಂದು ಫಲಿತಾಂಶ ಪ್ರಕಟವಾಗಲಿದೆ.",
    "1990ರಿಂದ 2000ದವರೆಗೆ ಅವರು ಇಲ್ಲಿ ಕೆಲಸ ಮಾಡಿದರು.", "ಬೆಲೆ 100ರಷ್ಟು ಏರಿಕೆಯಾಗಿದೆ.", "1000ಕ್ಕೂ ಹೆಚ್ಚು ಜನ ಬಂದಿದ್ದರು.",
    "ಅವನು 3ನೇ ತರಗತಿಯಲ್ಲಿ ಓದುತ್ತಿದ್ದಾನೆ.", "ಇದು ಅವರ 25 ನೇ ಹುಟ್ಟುಹಬ್ಬ.", "ಭಾರತ 2nd ಸ್ಥಾನ ಪಡೆಯಿತು.",
    "ಮಳೆ 3.5 ಸೆಂಟಿಮೀಟರ್ ದಾಖಲಾಗಿದೆ.", "ಜಿಡಿಪಿ ಬೆಳವಣಿಗೆ 7.25ರಷ್ಟು ಇದೆ.", "ಮತದಾನ 72% ಆಗಿದೆ.",
    "ಶೇ 50ರಷ್ಟು ಸೀಟುಗಳು ಖಾಲಿ ಇವೆ.", "ಬಡ್ಡಿ ದರ 8.5%ರಷ್ಟು ಇದೆ.", "ಟಿಕೆಟ್ ಬೆಲೆ ₹500.",
    "ಈ ಫೋನ್ ಬೆಲೆ Rs. 15,999 ಮಾತ್ರ.", "ಅವರಿಗೆ ₹1,25,000 ದಂಡ ವಿಧಿಸಲಾಗಿದೆ.", "ಚಹಾ ₹10.50ಕ್ಕೆ ಸಿಗುತ್ತದೆ.",
    "ರೂ. 2000 ನೋಟು ಹಿಂಪಡೆಯಲಾಗಿದೆ.", "ಸಭೆ 10:30ಕ್ಕೆ ಆರಂಭವಾಗುತ್ತದೆ.", "ರೈಲು 5:45 pm ಕ್ಕೆ ಹೊರಡುತ್ತದೆ.",
    "ಅಂಗಡಿ 9:00 AM ರಿಂದ ತೆರೆದಿರುತ್ತದೆ.", "ಕಾರ್ಯಕ್ರಮ 18:00ಕ್ಕೆ ಮುಗಿಯುತ್ತದೆ.", "ಪರೀಕ್ಷೆ 03-11-2026ರಂದು ನಡೆಯಲಿದೆ.",
    "ಅವರು 15/08/1947 ರಂದು ಜನಿಸಿದರು.", "ಗಡುವು 2026-12-31 ಆಗಿದೆ.", "ನವೆಂಬರ್ 1 ಕರ್ನಾಟಕ ರಾಜ್ಯೋತ್ಸವ.",
    "ನನ್ನ ಸಂಖ್ಯೆ 98450 12345 ಗೆ ಕರೆ ಮಾಡಿ.", "ಸಹಾಯವಾಣಿ +91 80123 45678 ಸಂಪರ್ಕಿಸಿ.", "ಪಿನ್ ಕೋಡ್ 560001 ಆಗಿದೆ.",
    "ತಾಪಮಾನ 25-30 ಡಿಗ್ರಿ ಇರಲಿದೆ.", "ಬೆಂಗಳೂರಿನಿಂದ ಮೈಸೂರಿಗೆ 145 km ದೂರ.", "ಇಂದು ತಾಪಮಾನ 32°C ದಾಖಲಾಗಿದೆ.",
    "ಅಕ್ಕಿ 5 kg ಬೆಲೆ ಏರಿದೆ.", "ಅವರು 10 ಕಿ.ಮೀ. ಓಡಿದರು.", "ಡಾ. ರಾಜ್‌ಕುಮಾರ್ ಕನ್ನಡದ ಮೇರು ನಟ.",
    "ಎಸ್.ಎಂ. ಕೃಷ್ಣ ಮುಖ್ಯಮಂತ್ರಿಯಾಗಿದ್ದರು.", "ಎಚ್.ಡಿ. ದೇವೇಗೌಡ ಪ್ರಧಾನಿಯಾಗಿದ್ದರು.", "BBMP ಚುನಾವಣೆ ಶೀಘ್ರದಲ್ಲೇ ನಡೆಯಲಿದೆ.",
    "ISRO ಹೊಸ ಉಪಗ್ರಹ ಉಡಾಯಿಸಿದೆ.", "KSRTC ಬಸ್ ದರ ಏರಿಕೆ.", "ನನ್ನ phone ಬ್ಯಾಟರಿ 20% ಇದೆ.", "ರಾಮ & ಶ್ಯಾಮ ಸ್ನೇಹಿತರು.",
    "The ticket costs ₹500.", "The project got Rs. 2,50,00,000 in funding.", "The population is 12,34,567.",
    "The meeting is at 10:30 AM on 03/11/2026.", "Call me at 98450 12345.",
    "Mysuru is 145 km from Bengaluru and it is 32°C today.", "He paid ₹10.50 for tea at 5:05 pm.",
    # unit tests and README examples
    "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?", "Hello, how are you?", "ನನ್ನ phone ಎಲ್ಲಿದೆ?", "ಇದು ಒಳ್ಳೆಯದು।",
    "NASA is ready for the next mission.", "Hello, my name is Vo.", "ನಾನು 5 ಗಂಟೆಗೆ ಬರುತ್ತೇನೆ", "ನನ್ನ ☺ ಎಲ್ಲಿ",
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.", "ಪರೀಕ್ಷೆ 03-11-2026ರಂದು 10:30ಕ್ಕೆ ನಡೆಯಲಿದೆ.",
    "ಬೆಲೆ ₹15,999 ಮಾತ್ರ, ಶೇ 50ರಷ್ಟು ರಿಯಾಯಿತಿ!", "The meeting in Bengaluru is at 5:05 pm.", "Mysuru is 145 km away.",
    "ನಾನು Hubballi ಗೆ ಹೋಗುತ್ತೇನೆ.", "Shetty and Byrappa met in Mysuru.", "ದುಃಖ ಕೃಷ್ಣ ಅಂಬಾ ಫ಼ೈಲ್ ಜ಼ೀರೋ",
    "ಕನ್ನಡ ನಾಡಿನ ಸಂಸ್ಕೃತಿ ಶ್ರೀಮಂತವಾಗಿದೆ. ಇಲ್ಲಿನ ಜನರು ಸ್ನೇಹಪರರು! ನೀವು ಬರುತ್ತೀರಾ?",
    "...ನಮಸ್ಕಾರ!!! ಹೇಗಿದ್ದೀರಿ??", "“ಉಲ್ಲೇಖ” — ‘ಒಂದು’ (ಎರಡು) [ಮೂರು] {ನಾಲ್ಕು} a/b c_d e*f",
    "I don't know, it's 5 o'clock.", "COVID-19 cases rose by 5%.", "$1,234.56 and £1.01 and €2", "$3.999",
    "2026-02-30 was not a date.", "3/4 of the cake", "The 21st and 22nd and 23rd and 101st.", "-5 and 3.14 and -2.5",
    "1234567890123456789 stars", "Dr. Rao and Mr. Gowda vs. Prof. Bhat, etc. at St. Mark's.", "CPU and GPU and NASA",
]

def semiotic_sentences():
    """The one-per-class sentences of training/data_pipeline_multi/tests/test_semiotic.py."""
    tree = ast.parse((ROOT / "training" / "data_pipeline_multi" / "tests" / "test_semiotic.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SENTENCES")


KN_WORDS = ["ಸಭೆ", "ಬೆಲೆ", "ಮನೆ", "ಜನ", "ಶಾಲೆ", "ರೈಲು", "ಬಸ್", "ಸಂಖ್ಯೆ", "ವರ್ಷ", "ಕೆಜಿ", "ತಾಪಮಾನ", "ಅವರು",
            "ಇದೆ", "ನಡೆಯಿತು", "ಆಗಿದೆ", "ಬಂದರು", "ಕರೆ", "ಮಾಡಿ", "ಒಟ್ಟು", "ದಿನ"]
SUFFIXES = ["", "", "", "ರಲ್ಲಿ", "ಕ್ಕೆ", "ಗೆ", "ನೇ", "ರಂದು", "ರಿಂದ", "ರಷ್ಟು", "ರ", "ದ", "ಕ್ಕೂ", "ರವರೆಗೆ", "ದಿಂದ",
            "ಗಿಂತ", "ರೊಳಗೆ", "ಅನ್ನು", "ಇಂದ", "ಯಲ್ಲಿ"]
EN_WORDS = ["the", "price", "is", "about", "on", "at", "call", "meeting", "costs", "and", "km", "today", "Bengaluru",
            "Hubballi", "Shetty", "Kempegowda", "o'clock", "don't", "isn't"]


def number(r):
    kind = r.random()
    n = r.choice([r.randint(0, 20), r.randint(0, 100), r.randint(0, 1000), r.randint(0, 99999),
                  r.randint(0, 10 ** 7), r.randint(0, 10 ** 10), r.randint(0, 10 ** 13)])
    if kind < 0.15:
        s = f"{n:,}"
    elif kind < 0.3:
        s = str(n)
        if len(s) > 3:                                  # Indian grouping
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            s = ",".join(([head] if head else []) + groups + [tail])
    elif kind < 0.42:
        s = f"{n}.{r.randint(0, 999):0{r.randint(1, 3)}d}"
    elif kind < 0.5:
        s = "0" + str(r.randint(0, 9999))
    elif kind < 0.58:
        s = str(n).translate(str.maketrans("0123456789", "೦೧೨೩೪೫೬೭೮೯"))
    else:
        s = str(n)
    return s


def kn_template(r):
    t = r.randrange(14)
    suf = r.choice(SUFFIXES)
    sp = r.choice(["", "", " "])
    w = lambda: r.choice(KN_WORDS)  # noqa: E731
    if t == 0:
        x = number(r)
    elif t == 1:
        d, m, y = r.randint(0, 32), r.randint(0, 13), r.randint(1000, 2099)
        x = r.choice([f"{d:02d}-{m:02d}-{y}", f"{d}/{m}/{y}", f"{d}.{m}.{y}", f"{y}-{m:02d}-{d:02d}"])
    elif t == 2:
        h, mi = r.randint(0, 25), r.randint(0, 61)
        x = f"{h}:{mi:02d}" + r.choice(["", "", " am", " pm", "AM", " P.M.", " a.m.", "pm"])
    elif t == 3:
        x = r.choice(["₹", "₹ ", "Rs", "Rs.", "Rs. ", "INR ", "ರೂ. ", "ರೂಪಾಯಿ "]) + number(r)
    elif t == 4:
        x = r.choice([f"{number(r)}%", f"{number(r)} %", f"ಶೇ {number(r)}", f"ಶೇ. {number(r)}"])
    elif t == 5:
        x = r.choice(["", "+91 ", "+91-", "+91"]) + str(r.randint(6, 9)) + str(r.randint(0, 9999)).zfill(4) \
            + r.choice(["", " ", "-"]) + str(r.randint(0, 99999)).zfill(5)
    elif t == 6:
        x = r.choice(["080-", "0821 ", "08182-"]) + str(r.randint(0, 99999999)).zfill(r.choice([6, 7, 8]))
    elif t == 7:
        x = r.choice(["ಪಿನ್ ಕೋಡ್ ", "ಪಿನ್: ", "PIN ", "Pin code ", "ಬೆಂಗಳೂರು-", "ಮೈಸೂರು - "]) + str(r.randint(100000, 999999))
    elif t == 8:
        x = f"{r.randint(0, 999)}{r.choice(['-', ' - ', '–'])}{r.randint(0, 9999)}"
    elif t == 9:
        x = number(r) + r.choice(["", " "]) + r.choice(["km", "kg", "cm", "mm", "ml", "g", "gm", "l", "ltr", "m",
                                                         "°C", "°", "GB", "MB", "hrs", "min", "km/h", "kmph", "kgs"])
    elif t == 10:
        x = f"{r.randint(0, 120)}{r.choice(['st', 'nd', 'rd', 'th'])}"
    elif t == 11:
        x = r.choice(["BBMP", "ISRO", "NASA", "KSRTC", "WiFi", "OK", "ATM", "abc", "Hello", "iPhone", "UNESCO", "COVID"])
    elif t == 12:
        x = r.choice(["ಡಾ.", "ಪ್ರೊ.", "ಶ್ರೀ.", "ಕ್ರಿ.ಶ.", "ಕ್ರಿ.ಪೂ.", "ನಂ.", "ಉದಾ.", "ಎಸ್.ಎಂ.", "ಎಚ್.ಡಿ.", "ಕೆ.ಜಿ"]) \
            + r.choice(["", " "]) + w()
    else:
        x = r.choice(EN_WORDS)
    sentence = [w() for _ in range(r.randint(0, 2))] + [x + (sp + suf if suf else "")] \
        + [w() for _ in range(r.randint(0, 2))]
    return " ".join(sentence) + r.choice([".", "", "?", "!", " ।", ","])


def en_template(r):
    t = r.randrange(10)
    if t == 0:
        x = number(r).translate(str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789"))
    elif t == 1:
        d, m, y = r.randint(0, 32), r.randint(0, 13), r.randint(1000, 2099)
        x = r.choice([f"{d:02d}-{m:02d}-{y}", f"{d}/{m}/{y}", f"{d}.{m}.{y}", f"{y}-{m:02d}-{d:02d}"])
    elif t == 2:
        x = f"{r.randint(0, 25)}:{r.randint(0, 61):02d}" + r.choice(["", " am", " PM", "a.m.", " p.m"])
    elif t == 3:
        x = r.choice(["₹", "Rs ", "Rs.", "INR ", "$", "£", "€"]) + r.choice(
            [str(r.randint(0, 10 ** 7)), f"{r.randint(0, 9999)}.{r.randint(0, 99):02d}", f"{r.randint(0, 10 ** 6):,}",
             f"{r.randint(0, 99)}.{r.randint(0, 9)}"])
    elif t == 4:
        x = r.choice(["", "+91 "]) + str(r.randint(6, 9)) + str(r.randint(0, 9999)).zfill(4) + r.choice(["", " ", "-"]) \
            + str(r.randint(0, 99999)).zfill(5)
    elif t == 5:
        x = f"{r.randint(0, 999)}{r.choice(['', ' '])}{r.choice(['km', 'kg', 'cm', 'mm', 'ml', '°C', '°', 'km/h'])}"
    elif t == 6:
        x = f"{r.randint(0, 2000)}{r.choice(['st', 'nd', 'rd', 'th', 'ST', 'Th'])}"
    elif t == 7:
        x = r.choice(["Dr.", "Mr.", "Mrs.", "Prof.", "St.", "vs.", "etc.", "NASA", "CPU", "GPU", "BBMP", "A", "I",
                      "e.g.", "U.S.", "x-ray", "rock-n-roll", "3-4", "-7", "1.5", "50%", "A&B", "1+1"])
    elif t == 8:
        x = r.choice(EN_WORDS)
    else:
        x = f"{r.randint(1, 99)}/{r.randint(1, 99)}"
    words = [r.choice(EN_WORDS) for _ in range(r.randint(0, 3))]
    words.insert(r.randint(0, len(words)), x)
    return " ".join(words) + r.choice([".", "", "?", "!", ","])


KN_ALPHABET = [chr(c) for c in range(0x0C80, 0x0D00)]
COMMON_KN = "ಅಆಇಈಉಊಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳಾಿೀುೂೃೆೇೈೊೋೌ್ಂಃ಼"
OTHER = list("0123456789೦೧೨೩೪೫೬೭೮೯abcxyzABCXYZ .,?!;:-–—/'\"()%₹$£€+&@#=~|•·^<>_*[]{}…।॥“”‘’°\t\n") \
    + ["‌", "‍", " ", " ", "﻿", "☺", "😀", "é", "ß", "½", "²"]


def fuzz(r, kannada_share):
    out = []
    for _ in range(r.randint(1, 40)):
        x = r.random()
        if x < kannada_share * 0.8:
            out.append(r.choice(COMMON_KN))
        elif x < kannada_share:
            out.append(r.choice(KN_ALPHABET))
        else:
            out.append(r.choice(OTHER))
    return "".join(out)


def category(error):
    m = str(error)
    for prefix, kind in [("could not read the number", "number"), ("unsupported character", "unsupported"),
                         ("unsupported text characters", "unsupported"), ("cannot read Kannada word", "kannada-word"),
                         ("out-of-vocabulary", "oov"), ("write slash dates", "slash"),
                         ("currency amounts", "currency"), ("unknown phone", "phone"),
                         ("unknown ARPAbet phone", "phone"), ("day is out of range", "date"),
                         ("month must be in", "date"), ("year ", "date")]:
        if m.startswith(prefix):
            return kind
    return f"python:{type(error).__name__}:{m}"


def main(out_path):
    r = random.Random(20261103)
    cases = list(HAND) + semiotic_sentences()
    cases += [kn_template(r) for _ in range(4000)]
    cases += [en_template(r) for _ in range(2500)]
    cases += [fuzz(r, 0.7) for _ in range(3000)]
    cases += [fuzz(r, 0.0) for _ in range(1500)]
    results = []
    for text in cases:
        try:
            front = kannada.phonemize(text)
            results.append({"text": text, "normalized": front.normalized_text, "ids": list(front.token_ids),
                            "unknown": list(front.unknown_words), "error": None})
        except Exception as error:  # noqa: BLE001 - every outcome is compared
            results.append({"text": text, "error": category(error)})
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    errors = sum(1 for x in results if x["error"])
    print(f"{len(results)} cases ({errors} raise in Python) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parity_ref.json")
