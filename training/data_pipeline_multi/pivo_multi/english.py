"""English frontend, copied from PiVo-1.4M `pivo/text.py` (Apache-2.0, Empero AI)
via the Kannada pipeline (pivo_kn_v3/english.py). Same normalisation and CMU lexicon,
so English pronunciations are identical to PiVo's. The CMU dictionary path is set with
`set_cmudict_path` (PiVo's pivo/resources/cmudict.dict).
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .symbols import PIVO_SPECIAL as SPECIAL, SYMBOL_TO_ID

_CMUDICT_PATH = None


def set_cmudict_path(path):
    global _CMUDICT_PATH
    _CMUDICT_PATH = str(path)
    lexicon.cache_clear()


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_MONTHS = "January February March April May June July August September October November December".split()


def cardinal(number: int) -> str:
    if number < 0:
        return "minus " + cardinal(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        return _TENS[number // 10] + (" " + cardinal(number % 10) if number % 10 else "")
    for scale, label in [
        (10**12, "trillion"), (10**9, "billion"), (10**6, "million"),
        (1000, "thousand"), (100, "hundred"),
    ]:
        if number >= scale:
            if number >= 10**15:
                return " ".join(_ONES[int(d)] for d in str(number))
            return (
                cardinal(number // scale) + " " + label
                + (" " + cardinal(number % scale) if number % scale else "")
            )
    raise AssertionError("unreachable")


def ordinal(number: int) -> str:
    words = cardinal(number).split()
    last = words[-1]
    words[-1] = _ORDINALS.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(words)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": ",", "–": ",", "…": "..."})
    )

    def date(match):
        from datetime import date as calendar_date

        year, month, day = map(int, match.groups())
        calendar_date(year, month, day)
        return f"{_MONTHS[month - 1]} {ordinal(day)}, {cardinal(year)}"

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date, text)
    if re.search(r"\d/\d", text):
        raise ValueError("write slash dates/fractions in words, or use an ISO date")
    for abbreviation, expansion in {
        "Mr": "Mister", "Mrs": "Missus", "Dr": "Doctor", "Prof": "Professor",
        "St": "Saint", "vs": "versus", "etc": "et cetera",
    }.items():
        text = re.sub(rf"\b{abbreviation}\.(?=\s|$)", expansion, text, flags=re.I)

    def money(match):
        symbol, raw = match.groups()
        raw = raw.replace(",", "")
        whole, _, fraction = raw.partition(".")
        unit = {"$": "dollar", "£": "pound", "€": "euro"}[symbol]
        result = cardinal(int(whole)) + " " + unit + ("s" if int(whole) != 1 else "")
        cents = int(fraction.ljust(2, "0")) if fraction else 0
        if cents:
            result += (
                " and " + cardinal(cents)
                + (
                    " penny" if cents == 1 and symbol == "£"
                    else " pence" if symbol == "£"
                    else " cent" if cents == 1
                    else " cents"
                )
            )
        return result

    if re.search(r"[$£€]\d[\d,]*\.\d{3,}", text):
        raise ValueError("currency amounts must have at most two decimal places")
    text = re.sub(r"([$£€])(\d[\d,]*(?:\.\d{1,2})?)(?!\d)", money, text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(int(m[1])), text, flags=re.I)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m[0].replace(",", ""), text)
    text = re.sub(
        r"-?(?:\d+\.\d+|\d+)",
        lambda m: (
            ("minus " if m[0].startswith("-") else "")
            + " point ".join(
                [cardinal(int(m[0].lstrip("-").split(".")[0]))]
                + [" ".join(_ONES[int(c)] for c in m[0].split(".")[1])]
                if "." in m[0]
                else [cardinal(int(m[0].lstrip("-")))]
            )
        ),
        text,
    )
    text = text.replace("%", " percent ").replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"\bNASA\b", "nasa", text)
    text = re.sub(r"\b[A-Z]{2,5}\b", lambda m: " ".join(m[0]), text)
    text = text.replace("-", " ")
    text = re.sub(r'[()\[\]{}"/]', " ", text)
    unsupported = re.findall(r"[^a-zA-Z\s'.,?!;:]", text)
    if unsupported:
        raise ValueError(
            f"unsupported text characters after normalization: {sorted(set(unsupported))}"
        )
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def lexicon():
    if _CMUDICT_PATH is None:
        raise RuntimeError("call set_cmudict_path() with PiVo's pivo/resources/cmudict.dict")
    dictionary = {}
    with open(_CMUDICT_PATH, encoding="utf-8") as fh:
        raw = fh.read()
    for line in raw.splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        dictionary.setdefault(word, []).append(parts[1:])
    return dictionary


@dataclass(frozen=True)
class FrontendResult:
    normalized_text: str
    symbols: tuple
    token_ids: tuple
    unknown_words: tuple


def word_to_symbols(word: str):
    """First CMU pronunciation of a lowercase word, or None if out of vocabulary."""
    entry = lexicon().get(word.lower())
    return list(entry[0]) if entry else None


def phonemize(text: str, *, strict: bool = True, overrides=None) -> FrontendResult:
    normalized = normalize(text)
    dictionary = lexicon()
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    tokens = ["<bos>"]
    unknown = []
    for item in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]", normalized):
        if item in SPECIAL:
            tokens.append(item)
            continue
        word = item.lower()
        phones = overrides.get(word)
        if phones is None and word in dictionary:
            phones = dictionary[word][0]
        if phones is None:
            unknown.append(word)
            phones = [phone for c in word if c.isalpha() for phone in dictionary[c][0]]
        if tokens[-1] not in ["<bos>", "<space>"]:
            tokens.append("<space>")
        for phone in phones:
            if phone not in SYMBOL_TO_ID:
                raise ValueError(f"unknown ARPAbet phone {phone!r} in {word!r}")
            tokens.append(phone)
    if unknown and strict:
        raise ValueError(
            f"out-of-vocabulary words require pronunciation overrides: {', '.join(sorted(set(unknown)))}"
        )
    tokens.append("<eos>")
    return FrontendResult(
        normalized, tuple(tokens), tuple(SYMBOL_TO_ID[s] for s in tokens), tuple(unknown)
    )
