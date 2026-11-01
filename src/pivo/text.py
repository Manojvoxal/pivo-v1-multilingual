"""Deterministic English frontend. OOV words are reported, never silently guessed."""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"]
SYMBOLS = tuple(SPECIAL + CONSONANTS + [v + s for v in VOWELS for s in "012"])
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = 128  # Reserved entries are included in the parameter count.
VOICE_IDS = {"Pi": 0, "Vo": 1}
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_MONTHS = (
    "January February March April May June July August September October November December".split()
)


def cardinal(number: int) -> str:
    if number < 0:
        return "minus " + cardinal(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        return _TENS[number // 10] + (" " + cardinal(number % 10) if number % 10 else "")
    for scale, label in [
        (10**12, "trillion"),
        (10**9, "billion"),
        (10**6, "million"),
        (1000, "thousand"),
        (100, "hundred"),
    ]:
        if number >= scale:
            if number >= 10**15:
                return " ".join(_ONES[int(d)] for d in str(number))
            return (
                cardinal(number // scale)
                + " "
                + label
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

    # An unambiguous ISO date; ambiguous slash dates are rejected below.
    def date(match):
        from datetime import date as calendar_date

        year, month, day = map(int, match.groups())
        calendar_date(year, month, day)
        return f"{_MONTHS[month - 1]} {ordinal(day)}, {cardinal(year)}"

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date, text)
    if re.search(r"\d/\d", text):
        raise ValueError("write slash dates/fractions in words, or use an ISO date")
    for abbreviation, expansion in {
        "Mr": "Mister",
        "Mrs": "Missus",
        "Dr": "Doctor",
        "Prof": "Professor",
        "St": "Saint",
        "vs": "versus",
        "etc": "et cetera",
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
                " and "
                + cardinal(cents)
                + (
                    " penny"
                    if cents == 1 and symbol == "£"
                    else " pence"
                    if symbol == "£"
                    else " cent"
                    if cents == 1
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
    # Known spoken acronym; preserve its dictionary pronunciation before the
    # generic initialism rule separates uppercase letters (CPU, GPU, ...).
    text = re.sub(r"\bNASA\b", "nasa", text)
    text = re.sub(r"\b[A-Z]{2,5}\b", lambda m: " ".join(m[0]), text)
    text = text.replace("-", " ")
    # Parentheses are ordinary prose in v1. No event semantics.
    text = re.sub(r'[()\[\]{}"/]', " ", text)
    unsupported = re.findall(r"[^a-zA-Z\s'.,?!;:]", text)
    if unsupported:
        raise ValueError(
            f"unsupported text characters after normalization: {sorted(set(unsupported))}"
        )
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def lexicon():
    from importlib.resources import files

    dictionary = {}
    raw = files("pivo").joinpath("resources/cmudict.dict").read_text(encoding="utf-8")
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
    symbols: tuple[str, ...]
    token_ids: tuple[int, ...]
    unknown_words: tuple[str, ...]


def phonemize(
    text: str, *, strict: bool = True, overrides: dict[str, list[str]] | None = None
) -> FrontendResult:
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
            # Explicit opt-in fallback spells the unknown word; production callers should use overrides.
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
