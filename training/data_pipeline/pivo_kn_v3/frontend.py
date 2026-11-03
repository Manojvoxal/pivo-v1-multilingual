"""Bilingual (Kannada + English) frontend producing PiVo-style token sequences.

Token layout follows PiVo exactly:
    <bos> w1-phones <space> w2-phones . <space> w3-phones ? <eos>
i.e. <space> goes before every word except the first one after <bos>/<space>,
and punctuation tokens are appended where they occur.

English utterances (lang="en") go through PiVo's own English frontend
unchanged. Kannada utterances (lang="kn") use the Kannada G2P for Kannada-script
words and PiVo's CMU lexicon for any Latin-script words (code-mix).
"""

import re
import unicodedata
from dataclasses import dataclass

from . import english, kannada
from .symbols import PIVO_SPECIAL, SYMBOL_TO_ID


class FrontendError(ValueError):
    pass


@dataclass(frozen=True)
class Phonemized:
    lang: str
    normalized_text: str
    symbols: tuple            # full token sequence incl. <bos>/<space>/punct/<eos>
    token_ids: tuple
    words: tuple              # ((word, (phone, ...)), ...) in spoken order


_PUNCT_MAP = str.maketrans({
    "।": ".", "॥": ".", "…": ".", "—": ",", "–": ",", "‘": "'", "’": "'",
    "“": " ", "”": " ", '"': " ", "(": " ", ")": " ", "[": " ", "]": " ",
    "{": " ", "}": " ", "/": " ", "-": " ", "_": " ", "*": " ", "'": " ",
})
_KN_TOKEN_RE = re.compile(r"[ಀ-೿‌‍]+|[A-Za-z]+|[.,?!;:]|\S")


def normalize_kn(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_PUNCT_MAP)
    text = text.replace("‌", "").replace("‍", "")
    text = re.sub(r"([.?!])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _assemble(lang, normalized, items):
    """items: list of ("word", word, phones) or ("punct", p, None)."""
    tokens, words = ["<bos>"], []
    for kind, value, phones in items:
        if kind == "punct":
            if tokens[-1] == "<bos>":
                continue  # leading punctuation carries no sound
            tokens.append(value)
            continue
        if tokens[-1] not in ("<bos>", "<space>"):
            tokens.append("<space>")
        tokens.extend(phones)
        words.append((value, tuple(phones)))
    tokens.append("<eos>")
    if not words:
        raise FrontendError("no speakable words")
    return Phonemized(lang, normalized, tuple(tokens),
                      tuple(SYMBOL_TO_ID[t] for t in tokens), tuple(words))


def phonemize_kn(text: str) -> Phonemized:
    normalized = normalize_kn(text)
    items = []
    for tok in _KN_TOKEN_RE.findall(normalized):
        if tok in PIVO_SPECIAL:
            items.append(("punct", tok, None))
        elif re.fullmatch(r"[A-Za-z]+", tok):
            phones = english.word_to_symbols(tok)
            if phones is None:
                raise FrontendError(f"English word not in CMU lexicon: {tok!r}")
            items.append(("word", tok.lower(), phones))
        elif re.fullmatch(r"[ಀ-೿‌‍]+", tok):
            phones = kannada.word_to_symbols(tok)
            if phones is None:
                raise FrontendError(f"cannot phonemize Kannada word {tok!r} (digits or unknown signs)")
            items.append(("word", tok, phones))
        else:
            raise FrontendError(f"unsupported character {tok!r}")
    return _assemble("kn", normalized, items)


def phonemize_en(text: str) -> Phonemized:
    try:
        normalized = english.normalize(text)
    except ValueError as exc:
        raise FrontendError(str(exc)) from exc
    items = []
    for tok in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]", normalized):
        if tok in PIVO_SPECIAL:
            items.append(("punct", tok, None))
            continue
        phones = english.word_to_symbols(tok)
        if phones is None:
            raise FrontendError(f"English word not in CMU lexicon: {tok!r}")
        items.append(("word", tok.lower(), phones))
    result = _assemble("en", normalized, items)
    # Parity check against PiVo's own phonemize for in-vocabulary text.
    # (Only possible token difference: PiVo keeps leading punctuation; we drop it.)
    reference = english.phonemize(text, strict=True).symbols
    if [t for t in reference if t not in PIVO_SPECIAL] != [
        t for t in result.symbols if t not in PIVO_SPECIAL
    ]:
        raise FrontendError("English frontend parity mismatch")
    return result


def phonemize(text: str, lang: str) -> Phonemized:
    if lang == "kn":
        return phonemize_kn(text)
    if lang == "en":
        return phonemize_en(text)
    raise ValueError(f"unknown language {lang!r}")
