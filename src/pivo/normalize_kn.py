"""Text normalization for the Kannada + English model (formal / news reading style).

Two entry points, both run before phonemization in ``kannada.phonemize``:

``normalize_kannada_text(text)``
    For text containing Kannada script. Writes numbers and symbols out as Kannada
    words using the Indian system (ಸಾವಿರ, ಲಕ್ಷ, ಕೋಟಿ). Covers Kannada and ASCII
    digits, case suffixes on numbers (2026ರಲ್ಲಿ, 5ಕ್ಕೆ, 3ನೇ), decimals, percent,
    rupees and paise, times with AM/PM, DD-MM-YYYY dates, phone numbers, ranges,
    units, common abbreviations, Kannada initials (ಎಸ್.ಎಂ.) and English acronyms.

``normalize_indian_english(text)``
    For English text. Handles Indian conventions that PiVo's US-English frontend
    lacks: ₹/Rs. amounts, lakh/crore grouping (1,25,000), clock times, day-first
    dates, phone numbers and units. Everything else is left to ``text.normalize``.

Every rule is a small, readable table or function so a native speaker can adjust
the wording. Unit tests: ``tests/test_normalize_kn.py``.
"""

import re
import unicodedata

from .text import cardinal as en_cardinal, ordinal as en_ordinal

# ============================================================================ Kannada numbers
ONES = ["ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
        "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು",
        "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು"]
TENS = {2: "ಇಪ್ಪತ್ತು", 3: "ಮೂವತ್ತು", 4: "ನಲವತ್ತು", 5: "ಐವತ್ತು", 6: "ಅರವತ್ತು",
        7: "ಎಪ್ಪತ್ತು", 8: "ಎಂಬತ್ತು", 9: "ತೊಂಬತ್ತು"}
HUNDREDS = {1: "ನೂರು", 2: "ಇನ್ನೂರು", 3: "ಮುನ್ನೂರು", 4: "ನಾನೂರು", 5: "ಐನೂರು",
            6: "ಆರುನೂರು", 7: "ಏಳುನೂರು", 8: "ಎಂಟುನೂರು", 9: "ಒಂಬೈನೂರು"}
THOUSAND, LAKH, CRORE = "ಸಾವಿರ", "ಲಕ್ಷ", "ಕೋಟಿ"
POINT = "ಪಾಯಿಂಟ್"
PERCENT = "ಶೇಕಡಾ"
RUPEE, PAISE = "ರೂಪಾಯಿ", "ಪೈಸೆ"
HOUR, MINUTE = "ಗಂಟೆ", "ನಿಮಿಷ"
MONTHS = ["ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್",
          "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"]

_U = "ು"
_VOWEL_SIGN = {"ಅ": "", "ಆ": "ಾ", "ಇ": "ಿ", "ಈ": "ೀ", "ಉ": "ು", "ಊ": "ೂ", "ಎ": "ೆ", "ಏ": "ೇ",
               "ಐ": "ೈ", "ಒ": "ೊ", "ಓ": "ೋ", "ಔ": "ೌ"}
_FRONT_VOWEL_END = ("ಿ", "ೀ", "ೆ", "ೇ", "ೈ")


def _stem(word):
    """ಇಪ್ಪತ್ತು -> ಇಪ್ಪತ್ತ (drop the final -u so a suffix or another number can follow)."""
    return word[:-1] if word.endswith(_U) else word


def _join(left, right):
    """Glue two parts the way Kannada spells compound numbers: ಇಪ್ಪತ್ತು + ಒಂದು -> ಇಪ್ಪತ್ತೊಂದು."""
    stem = _stem(left)
    if right[0] in _VOWEL_SIGN:
        return stem + _VOWEL_SIGN[right[0]] + right[1:]
    return stem + right


def below_100(n):
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] if ones == 0 else _join(TENS[tens], ONES[ones])


def below_1000(n):
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return below_100(rest)
    return HUNDREDS[hundreds] if rest == 0 else _stem(HUNDREDS[hundreds]) + " " + below_100(rest)


def kannada_cardinal(n):
    """Integer -> Kannada words, Indian system. 2026 -> ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು."""
    if n < 0:
        return "ಮೈನಸ್ " + kannada_cardinal(-n)
    if n == 0:
        return ONES[0]
    crore, n = divmod(n, 10 ** 7)
    lakh, n = divmod(n, 10 ** 5)
    thousand, rest = divmod(n, 1000)
    parts = []
    if crore:
        parts.append([kannada_cardinal(crore), CRORE])
    if lakh:
        parts.append([below_100(lakh), LAKH])
    if thousand == 1 and rest and not parts:
        parts.append([THOUSAND])                          # 1,500 -> ಸಾವಿರದ ಐನೂರು
    elif thousand:
        parts.append([below_100(thousand), THOUSAND])     # ಒಂದು ಸಾವಿರ, ಎರಡು ಸಾವಿರ
    if rest:
        parts.append([below_1000(rest)])
    words = []
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if not last and part[-1] in (THOUSAND, LAKH):
            part = part[:-1] + [part[-1] + "ದ"]          # genitive link: ಸಾವಿರದ, ಲಕ್ಷದ
        words.extend(part)
    return " ".join(words)


def digits_kn(s):
    return " ".join(ONES[int(d)] for d in s)


def attach_suffix(words, suffix):
    """Add a case suffix written after a number: (ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು, ರಲ್ಲಿ) -> ...ಇಪ್ಪತ್ತಾರರಲ್ಲಿ."""
    if not suffix:
        return words
    head, _, last = words.rpartition(" ")
    if last.endswith(_U) and suffix.startswith("ಗ"):
        suffix = "ಕ್ಕ" + suffix[1:]            # numbers in -u take -ಕ್ಕೆ: ಐದಕ್ಕೆ, not ಐದಗೆ
    if last.endswith(_FRONT_VOWEL_END):
        # after i/e Kannada uses a y-glide and dative -ge: ಕೋಟಿಯಲ್ಲಿ, ಗಂಟೆಗೆ, ರೂಪಾಯಿಗೆ
        if suffix.startswith("ರ"):
            suffix = "ಯ" + suffix[1:]
        elif suffix.startswith("ಕ್ಕ"):
            suffix = "ಗ" + suffix[3:]
    if suffix[0] in _VOWEL_SIGN:
        last = _join(last, suffix)
    else:
        last = _stem(last) + suffix
    return (head + " " + last) if head else last


def kannada_number_text(raw):
    """'1,25,000' or '3.75' -> words (decimals read digit by digit after ಪಾಯಿಂಟ್)."""
    raw = raw.replace(",", "")
    whole, _, frac = raw.partition(".")
    words = kannada_cardinal(int(whole)) if whole else ONES[0]
    if frac:
        words += " " + POINT + " " + digits_kn(frac)
    return words


# ============================================================================ Kannada rules
KN = "ಀ-೿"
_KN_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")
_SUFFIX = f"([{KN}‌‍]*)"           # Kannada letters glued to a number, e.g. ರಲ್ಲಿ
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"

ABBREVIATIONS = {                          # written form -> spoken form (dots matter)
    "ಡಾ.": "ಡಾಕ್ಟರ್", "ಪ್ರೊ.": "ಪ್ರೊಫೆಸರ್", "ಶ್ರೀ.": "ಶ್ರೀ", "ಕ್ರಿ.ಶ.": "ಕ್ರಿಸ್ತ ಶಕ",
    "ಕ್ರಿ.ಪೂ.": "ಕ್ರಿಸ್ತ ಪೂರ್ವ", "ರೂ.": RUPEE, "ಶೇ.": PERCENT, "ನಂ.": "ನಂಬರ್",
    "ಉದಾ.": "ಉದಾಹರಣೆಗೆ", "ಕಿ.ಮೀ.": "ಕಿಲೋಮೀಟರ್", "ಕಿ.ಮೀ": "ಕಿಲೋಮೀಟರ್",
    "ಸೆಂ.ಮೀ.": "ಸೆಂಟಿಮೀಟರ್", "ಮಿ.ಮೀ.": "ಮಿಲಿಮೀಟರ್", "ಕೆ.ಜಿ.": "ಕೆಜಿ", "ಕೆ.ಜಿ": "ಕೆಜಿ",
}
UNITS = {                                  # after a number -> Kannada word
    "km/h": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಗಂಟೆ", "kmph": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಗಂಟೆ", "km": "ಕಿಲೋಮೀಟರ್",
    "kg": "ಕಿಲೋಗ್ರಾಂ", "cm": "ಸೆಂಟಿಮೀಟರ್", "mm": "ಮಿಲಿಮೀಟರ್", "ml": "ಮಿಲಿಲೀಟರ್",
    "gm": "ಗ್ರಾಂ", "g": "ಗ್ರಾಂ", "ltr": "ಲೀಟರ್", "l": "ಲೀಟರ್", "m": "ಮೀಟರ್",
    "°C": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್", "°c": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್", "°": "ಡಿಗ್ರಿ", "GB": "ಜಿಬಿ", "MB": "ಎಂಬಿ",
    "hrs": HOUR, "hr": HOUR, "mins": MINUTE, "min": MINUTE,
}
# Kannada letter names used as initials (ಎಸ್.ಎಂ. ಕೃಷ್ಣ): their dots are not full stops
INITIALS = {"ಎ", "ಬಿ", "ಸಿ", "ಡಿ", "ಇ", "ಎಫ್", "ಜಿ", "ಎಚ್", "ಹೆಚ್", "ಐ", "ಜೆ", "ಕೆ", "ಎಲ್", "ಎಂ",
            "ಎನ್", "ಒ", "ಪಿ", "ಕ್ಯೂ", "ಆರ್", "ಎಸ್", "ಟಿ", "ಯು", "ವಿ", "ಡಬ್ಲ್ಯು", "ಎಕ್ಸ್", "ವೈ", "ಜೆಡ್", "ಝಡ್"}
# English acronyms usually said as a word; other ALL-CAPS words are spelled letter by letter
ACRONYM_WORDS = {"NASA": "ನಾಸಾ", "ISRO": "ಇಸ್ರೋ", "COVID": "ಕೋವಿಡ್", "UNESCO": "ಯುನೆಸ್ಕೋ",
                 "NATO": "ನ್ಯಾಟೋ", "WiFi": "ವೈಫೈ", "OK": "ಓಕೆ"}
_SYMBOLS = {"&": " ಮತ್ತು ", "+": " ಪ್ಲಸ್ ", "=": " ಸಮ ", "@": " ಅಟ್ ", "#": " ", "~": " ",
            "|": " ", "•": " ", "·": " ", "^": " ", "<": " ", ">": " "}


def _period_word(hour, ampm):
    if ampm == "am":
        return "ಮಧ್ಯರಾತ್ರಿ" if hour == 12 else ("ರಾತ್ರಿ" if hour <= 3 else "ಬೆಳಿಗ್ಗೆ")
    if hour == 12 or hour <= 3:
        return "ಮಧ್ಯಾಹ್ನ"
    return "ಸಂಜೆ" if hour <= 6 else "ರಾತ್ರಿ"


def _time_kn(m):
    hour, minute = int(m["h"]), int(m["m"])
    ampm = (m["ampm"] or "").replace(".", "").lower() or None
    if ampm is None and hour > 12:                      # 17:30 -> ಸಂಜೆ ಐದು ಗಂಟೆ ಮೂವತ್ತು ನಿಮಿಷ
        ampm, hour = "pm", hour - 12
    words = []
    if ampm:
        words.append(_period_word(hour, ampm))
    words += [kannada_cardinal(hour), HOUR]
    if minute:
        words += [kannada_cardinal(minute), MINUTE]
    return " " + attach_suffix(" ".join(words), m["suf"]) + " "


def _date_kn(day, month, year, suffix):
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    words = f"{kannada_cardinal(day)} {MONTHS[month - 1]} {kannada_cardinal(year)}"
    return " " + attach_suffix(words, suffix) + " "


def _money_kn(m):
    amount = m["amt"].replace(",", "")
    rupees, _, paise = amount.partition(".")
    words = kannada_cardinal(int(rupees)) + " " + RUPEE
    if paise and int(paise.ljust(2, "0")[:2]):
        words += " " + kannada_cardinal(int(paise.ljust(2, "0")[:2])) + " " + PAISE
    return " " + attach_suffix(words, m["suf"]) + " "


# case suffixes that writers sometimes separate from the number with a space (5:45 pm ಕ್ಕೆ)
ORPHAN_SUFFIXES = ["ಕ್ಕೆ", "ಕ್ಕೂ", "ಕ್ಕಿಂತ", "ರಿಂದ", "ರಂದು", "ರಲ್ಲಿ", "ರವರೆಗೆ", "ರೊಳಗೆ", "ರಷ್ಟು",
                   "ರಂತೆ", "ದಿಂದ", "ದಲ್ಲಿ", "ದವರೆಗೆ", "ಗೆ", "ಗಿಂತ", "ನೇ", "ರ", "ದ"]
_SUF = f"(?P<suf>[{KN}\u200c\u200d]*)"
_END = r"(?!\d)"                       # end of a number (\b fails before a glued Kannada suffix)


def normalize_kannada_text(text):
    text = unicodedata.normalize("NFC", text).translate(_KN_DIGITS)
    for written, spoken in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![{KN}]){re.escape(written)}", " " + spoken + " ", text)
    # initials: ಎಸ್.ಎಂ. ಕೃಷ್ಣ -> ಎಸ್ ಎಂ ಕೃಷ್ಣ
    text = re.sub(rf"(?<![{KN}])([{KN}]+)\.(?=\s*[{KN}])",
                  lambda m: m[1] + " " if m[1] in INITIALS else m[0], text)
    # glue a separated case suffix back onto its number / time: "1947 ರಂದು" -> "1947ರಂದು"
    orphans = "|".join(sorted(ORPHAN_SUFFIXES, key=len, reverse=True))
    text = re.sub(rf"(\d|%|[AaPp]\.?[Mm]\.?)\s+({orphans})(?![{KN}])", r"\1\2", text)
    # phone numbers and PIN codes -> digit by digit (before any other number rule)
    text = re.sub(rf"(\+91[\s-]?)?(?<!\d)([6-9]\d{{4}})[\s-]?(\d{{5}}){_END}{_SUF}",
                  lambda m: (" ಪ್ಲಸ್ ಒಂಬತ್ತು ಒಂದು " if m[1] else " ")
                  + attach_suffix(digits_kn(m[2] + m[3]), m["suf"]) + " ", text)
    text = re.sub(rf"(?<!\d)0\d{{2,4}}[\s-]\d{{6,8}}{_END}{_SUF}",
                  lambda m: " " + attach_suffix(digits_kn(re.sub(r"\D", "", m[0][:len(m[0]) - len(m["suf"])])),
                                                m["suf"]) + " ", text)
    text = re.sub(rf"(ಪಿನ್\s?ಕೋಡ್|ಪಿನ್|PIN|[Pp]in\s?[Cc]ode|[{KN}]+\s?-)\s*:?\s*(\d{{6}}){_END}{_SUF}",
                  lambda m: m[1].rstrip("-") + " " + attach_suffix(digits_kn(m[2]), m["suf"]) + " ", text)
    # dates: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY (Indian, day first) and ISO YYYY-MM-DD
    text = re.sub(rf"(?<!\d)(\d{{4}})-(\d{{2}})-(\d{{2}}){_END}{_SUF}",
                  lambda m: _date_kn(int(m[3]), int(m[2]), int(m[1]), m["suf"]) or m[0], text)
    text = re.sub(rf"(?<![\d.])(\d{{1,2}})[/.-](\d{{1,2}})[/.-](\d{{4}}){_END}{_SUF}",
                  lambda m: _date_kn(int(m[1]), int(m[2]), int(m[3]), m["suf"]) or m[0], text)
    # times: 10:30, 5:00 pm, 18:00
    text = re.sub(rf"(?<!\d)(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\s*(?P<ampm>[AaPp]\.?[Mm]\.?)?(?![A-Za-z]){_SUF}",
                  _time_kn, text)
    # money: ₹500, Rs. 1,25,000.50, INR 20, ರೂ. 2000
    text = re.sub(rf"(?:₹|\bRs\.?|\bINR|{RUPEE})\s*(?P<amt>{_NUM}){_END}{_SUF}", _money_kn, text)
    # percent: 50% and ಶೇ 50 -> ಶೇಕಡಾ ಐವತ್ತು
    text = re.sub(rf"(?<![{KN}])ಶೇ(?![{KN}])", f" {PERCENT} ", text)
    text = re.sub(rf"({_NUM})\s*%{_SUF}",
                  lambda m: f" {PERCENT} " + attach_suffix(kannada_number_text(m[1]), m["suf"]) + " ", text)
    # ranges: 10-20 -> ಹತ್ತರಿಂದ ಇಪ್ಪತ್ತು
    text = re.sub(rf"(?<![\d.,])(\d+)\s?[-–]\s?(\d+){_END}{_SUF}",
                  lambda m: " " + attach_suffix(kannada_cardinal(int(m[1])), "ರಿಂದ") + " "
                  + attach_suffix(kannada_cardinal(int(m[2])), m["suf"]) + " ", text)
    # numbers with units: 5 km, 10kg, 30°C
    units = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    text = re.sub(rf"({_NUM})\s?({units})(?![A-Za-z])", lambda m: f" {kannada_number_text(m[1])} {UNITS[m[2]]} ", text)
    # ordinals in English style inside Kannada text: 3rd -> ಮೂರನೇ
    text = re.sub(r"(?<!\d)(\d+)(?:st|nd|rd|th)\b", lambda m: " " + attach_suffix(kannada_cardinal(int(m[1])), "ನೇ") + " ", text)
    # long digit strings and leading zeros are read digit by digit (IDs, account numbers)
    text = re.sub(rf"(?<![.,\d])(0\d+|\d{{7,}}){_END}{_SUF}",
                  lambda m: " " + attach_suffix(digits_kn(m[1]), m["suf"]) + " ", text)
    # remaining numbers, with any Kannada suffix glued on (2026ರಲ್ಲಿ, 3ನೇ, 5ಕ್ಕೆ)
    text = re.sub(rf"({_NUM}){_SUF}", lambda m: " " + attach_suffix(kannada_number_text(m[1]), m["suf"]) + " ", text)

    # English acronyms: said-as-word list, else spelled out (BBMP -> b b m p)
    def acronym(m):
        word = m[0]
        if word in ACRONYM_WORDS:
            return ACRONYM_WORDS[word]
        if word.isupper() and 2 <= len(word) <= 6:
            return " ".join(word.lower())
        return word
    text = re.sub(r"\b[A-Za-z]{2,6}\b", acronym, text)
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================ Indian English
# Pronunciations missing from (or US-flavoured in) the CMU dictionary
INDIAN_ENGLISH_LEXICON = {
    "lakh": ["L", "AA1", "K"], "lakhs": ["L", "AA1", "K", "S"],
    "crore": ["K", "R", "AO1", "R"], "crores": ["K", "R", "AO1", "R", "Z"],
    "rupee": ["R", "UW1", "P", "IY0"], "rupees": ["R", "UW1", "P", "IY0", "Z"],
    "paise": ["P", "AY1", "S", "EY0"], "paisa": ["P", "AY1", "S", "AA0"],
}
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
EN_UNITS = {"km/h": "kilometers per hour", "kmph": "kilometers per hour", "km": "kilometers",
            "kg": "kilograms", "cm": "centimeters", "mm": "millimeters", "ml": "milliliters",
            "°C": "degrees celsius", "°c": "degrees celsius", "°": "degrees"}
_DIGIT_WORDS = "zero one two three four five six seven eight nine".split()


def indian_english_cardinal(n):
    """1,25,000 -> one lakh twenty five thousand (lakh/crore above 99,999)."""
    if n < 100000:
        return en_cardinal(n)
    crore, n = divmod(n, 10 ** 7)
    lakh, rest = divmod(n, 10 ** 5)
    words = []
    if crore:
        words.append(indian_english_cardinal(crore) + " crore")
    if lakh:
        words.append(en_cardinal(lakh) + " lakh")
    if rest:
        words.append(en_cardinal(rest))
    return " ".join(words)


def _en_digits(s):
    return " ".join(_DIGIT_WORDS[int(d)] for d in s)


def normalize_indian_english(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\+91[\s-]?)?\b([6-9]\d{4})[\s-]?(\d{5})\b",
                  lambda m: (" plus nine one " if m[1] else " ") + _en_digits(m[2] + m[3]) + " ", text)

    def date(day, month, year, original):
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return original
        return f" {en_ordinal(day)} {EN_MONTHS[month - 1]}, {en_cardinal(year)} "
    text = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b",
                  lambda m: date(int(m[1]), int(m[2]), int(m[3]), m[0]), text)

    def clock(m):
        hour, minute = int(m[1]), int(m[2])
        words = en_cardinal(hour) + (" o'clock" if minute == 0 else
                                     " oh " + en_cardinal(minute) if minute < 10 else " " + en_cardinal(minute))
        ampm = (m[3] or "").replace(".", "").upper()
        return " " + words + (" " + " ".join(ampm) if ampm else "") + " "
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\s*([AaPp]\.?[Mm]\.?)?(?![A-Za-z])", clock, text)

    def money(m):
        amount = m[1].replace(",", "")
        rupees, _, paise = amount.partition(".")
        r = int(rupees)
        words = indian_english_cardinal(r) + (" rupee" if r == 1 else " rupees")
        if paise and int(paise.ljust(2, "0")[:2]):
            words += " and " + en_cardinal(int(paise.ljust(2, "0")[:2])) + " paise"
        return " " + words + " "
    text = re.sub(r"(?:₹|\bRs\.?|\bINR)\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", money, text)
    # Indian digit grouping (1,25,000 / 2,50,00,000) -> lakh / crore words
    text = re.sub(r"\b\d{1,2}(?:,\d{2})+,\d{3}\b",
                  lambda m: " " + indian_english_cardinal(int(m[0].replace(",", ""))) + " ", text)
    units = "|".join(re.escape(u) for u in sorted(EN_UNITS, key=len, reverse=True))
    text = re.sub(rf"(\d+(?:\.\d+)?)\s?({units})(?![A-Za-z])", lambda m: f"{m[1]} {EN_UNITS[m[2]]}", text)
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
