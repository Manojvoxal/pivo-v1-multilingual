"""Multilingual frontend: text in any of the supported languages -> PiVo token sequence.

Token layout is PiVo's:
    <bos> w1-phones <space> w2-phones . <space> w3-phones ? <eos>

* English sentences (lang="en") use PiVo's English normaliser + CMU lexicon (ARPAbet).
* Indic sentences: each word is read by the G2P of its script. A word in the sentence
  language's own script uses that language's rules; a word in another Indic script uses
  that script's default language; Latin-script words are read with the CMU lexicon
  (code-mixing), giving ARPAbet tokens.
* Training mode (strict=True, the default) rejects text with digits (numbers are not
  expanded yet for most languages), unknown characters, or English words missing from
  the CMU lexicon, so those clips are dropped instead of mis-labelled.
"""

import re
import unicodedata
from dataclasses import dataclass

from . import alphabetic, brahmic, english
from .langs import LANG, SCRIPT_DEFAULT_LANG, lang_scripts
from .scripts import BRAHMIC_BASE, ZWJ, script_of_word
from .symbols import PIVO_SPECIAL, SYMBOL_TO_ID


class FrontendError(ValueError):
    """Message starts with a short reason code (used as the drop reason)."""


@dataclass(frozen=True)
class Phonemized:
    lang: str
    normalized_text: str
    symbols: tuple            # full token sequence incl. <bos>/<space>/punct/<eos>
    token_ids: tuple
    words: tuple              # ((word, (symbol, ...)), ...) in spoken order


_PUNCT_MAP = str.maketrans({
    "।": ".", "॥": ".", "|": ".", "…": ".", "۔": ".", "؟": "?", "،": ",", "؛": ";",
    "᱾": ".", "᱿": ".", "꯫": ".", "꫰": ".", "꫱": ".",
    "—": ",", "–": ",", "‘": " ", "’": " ", "“": " ", "”": " ", '"': " ", "'": " ",
    "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ", "/": " ", "\\": " ",
    "-": " ", "_": " ", "*": " ", "«": " ", "»": " ", "<": " ", ">": " ", "~": " ",
    "‌": "", " ": " ", "​": "", "﻿": "",
})
_INDIC_CLASS = (r"ऀ-ൿ᱐-᱿᳐-᳿꣠-ꣿꫠ-꫿ꯀ-꯿"
                r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿‍")
_TOKEN_RE = re.compile(rf"[{_INDIC_CLASS}]+|[A-Za-z]+|[.,?!;:]|\S")
_DIGIT_RE = re.compile(r"\d")


# Old-style Malayalam chillus (consonant + virama + ZWJ) -> atomic chillu letters, so
# no ZWJ survives into words (MFA might strip it from transcripts but not the lexicon).
_ML_CHILLU = {"ണ്\u200d": "ൺ", "ന്\u200d": "ൻ", "ര്\u200d": "ർ", "ല്\u200d": "ൽ",
              "ള്\u200d": "ൾ", "ക്\u200d": "ൿ"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for old, new in _ML_CHILLU.items():
        text = text.replace(old, new)
    text = text.replace("\u200d", "").translate(_PUNCT_MAP)
    text = re.sub(r"\s*([.,?!;:])", r"\1", text)          # no space before punctuation
    text = re.sub(r"([.,?!;:])(?:\s*[.,?!;:])+", r"\1", text)  # collapse runs, keep the first
    return re.sub(r"\s+", " ", text).strip()


def _assemble(lang, normalized, items):
    """items: list of ("word", word, symbols) or ("punct", p, None)."""
    tokens, words = ["<bos>"], []
    for kind, value, symbols in items:
        if kind == "punct":
            if tokens[-1] == "<bos>" or tokens[-1] in PIVO_SPECIAL[4:]:
                continue  # leading punctuation carries no sound; no double punctuation
            if tokens[-1] == "<space>":
                tokens.pop()
            tokens.append(value)
            continue
        if tokens[-1] not in ("<bos>", "<space>"):
            tokens.append("<space>")
        tokens.extend(symbols)
        words.append((value, tuple(symbols)))
    tokens.append("<eos>")
    if not words:
        raise FrontendError("empty: no speakable words")
    return Phonemized(lang, normalized, tuple(tokens),
                      tuple(SYMBOL_TO_ID[t] for t in tokens), tuple(words))


def indic_word(word: str, lang: str):
    """Symbols ("in_*") for one Indic word, or raise FrontendError."""
    script = script_of_word(word)
    if script is None:
        raise FrontendError(f"mixed_script: {word!r}")
    if script == "arabic":
        raise FrontendError(f"unsupported_script: Perso-Arabic G2P not written yet ({word!r})")
    rules = lang if script in lang_scripts(lang) else SCRIPT_DEFAULT_LANG.get(script)
    if script in BRAHMIC_BASE:
        if rules not in LANG or LANG[rules]["inherent"] is None:
            rules = SCRIPT_DEFAULT_LANG[script]
        phones = brahmic.word_to_phones(word, rules)
    elif script == "olchiki":
        phones = alphabetic.olchiki_to_phones(word)
    elif script == "meetei":
        phones = alphabetic.meetei_to_phones(word)
    else:
        phones = None
    if not phones:
        raise FrontendError(f"g2p_failed: {word!r}")
    return ["in_" + p for p in phones]


def english_word(word: str):
    phones = english.word_to_symbols(word.lower())
    if phones is None:
        raise FrontendError(f"oov_english: {word!r}")
    return phones


def phonemize_indic(text: str, lang: str, strict: bool = True) -> Phonemized:
    if strict and _DIGIT_RE.search(text):
        raise FrontendError("digits: numbers are not expanded in training data")
    normalized = normalize(text)
    items = []
    for tok in _TOKEN_RE.findall(normalized):
        if tok in PIVO_SPECIAL:
            items.append(("punct", tok, None))
        elif re.fullmatch(r"[A-Za-z]+", tok):
            items.append(("word", tok.lower(), english_word(tok)))
        elif re.fullmatch(rf"[{_INDIC_CLASS}]+", tok):
            tok = tok.strip(ZWJ)
            if not tok:
                continue
            items.append(("word", tok, indic_word(tok, lang)))
        else:
            raise FrontendError(f"unsupported_char: {tok!r} (U+{ord(tok[0]):04X})")
    return _assemble(lang, normalized, items)


def phonemize_en(text: str, strict: bool = True) -> Phonemized:
    if strict and _DIGIT_RE.search(text):
        raise FrontendError("digits: numbers are not expanded in training data")
    try:
        normalized = english.normalize(text)
    except ValueError as exc:
        raise FrontendError(f"en_normalize: {exc}") from exc
    items = []
    for tok in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]", normalized):
        if tok in PIVO_SPECIAL:
            items.append(("punct", tok, None))
            continue
        items.append(("word", tok.lower(), english_word(tok)))
    return _assemble("en", normalized, items)


def phonemize(text: str, lang: str, strict: bool = True) -> Phonemized:
    if lang == "en":
        return phonemize_en(text, strict)
    if lang not in LANG:
        raise ValueError(f"unknown language {lang!r}")
    return phonemize_indic(text, lang, strict)


def drop_reason(exc: Exception) -> str:
    return str(exc).split(":", 1)[0].strip() or "frontend"
