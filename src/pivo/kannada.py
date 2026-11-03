"""Kannada + English frontend for bilingual exports (metadata ``"frontend": "pivo-kn-en-v1"``).

Text without Kannada script goes through the English frontend in ``text.py``
unchanged, so English token ids (0-78) and pronunciations are identical to
PiVo-1.4M. Text that contains Kannada script is read word by word:

* Kannada words use a rule-based grapheme-to-phoneme converter (below) and map
  to 48 Kannada phones, token ids 79-126.
* Latin-script words inside Kannada text (code-mixing, e.g. "ನನ್ನ phone")
  use the CMU dictionary like English, plus any pronunciation overrides.
* Punctuation becomes PiVo's pause tokens. Danda (।) is read as a full stop.

The tokens follow the same layout as English:
``<bos> word <space> word , <space> word ? <eos>``.
This is the frontend the Kannada training data was built with
(training/data_pipeline/pivo_kn_v3).

Numbers, dates, times, money, units and abbreviations are written out as words
first (``normalize_kn.py``), in formal / news reading style.
"""

import re
import unicodedata

from .normalize_kn import INDIAN_ENGLISH_LEXICON, normalize_indian_english, normalize_kannada_text
from .text import SPECIAL, SYMBOLS as ENGLISH_SYMBOLS, FrontendResult, lexicon
from .text import normalize as normalize_english
from .text import phonemize as phonemize_english

KN_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"]
KN_CONSONANTS = [
    "k", "kh", "g", "gh", "ng",       # velar
    "c", "ch", "j", "jh", "ny",       # palatal
    "tt", "tth", "dd", "ddh", "nn",   # retroflex
    "t", "th", "d", "dh", "n",        # dental
    "p", "ph", "b", "bh", "m",        # labial
    "y", "r", "l", "v", "ll",         # approximants (ll = retroflex lateral)
    "sh", "ssh", "s", "h",            # sibilants and glottal
    "f", "z",                         # nukta loans
]
KN_PHONES = tuple("kn_" + p for p in KN_VOWELS + KN_CONSONANTS)
SYMBOLS = ENGLISH_SYMBOLS + KN_PHONES          # 79 + 48 = 127 of the 128 embedding rows
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
FRONTEND_NAME = "pivo-kn-en-v1"

# ---------------------------------------------------------------- Kannada grapheme-to-phoneme
_V = {  # independent vowels
    "ಅ": ["a"], "ಆ": ["aa"], "ಇ": ["i"], "ಈ": ["ii"], "ಉ": ["u"], "ಊ": ["uu"],
    "ಋ": ["r", "u"], "ೠ": ["r", "uu"], "ಌ": ["l", "u"], "ೡ": ["l", "uu"],
    "ಎ": ["e"], "ಏ": ["ee"], "ಐ": ["ai"], "ಒ": ["o"], "ಓ": ["oo"], "ಔ": ["au"],
}
_M = {  # dependent vowel signs
    "ಾ": ["aa"], "ಿ": ["i"], "ೀ": ["ii"], "ು": ["u"], "ೂ": ["uu"],
    "ೃ": ["r", "u"], "ೄ": ["r", "uu"], "ೢ": ["l", "u"], "ೣ": ["l", "uu"],
    "ೆ": ["e"], "ೇ": ["ee"], "ೈ": ["ai"], "ೊ": ["o"], "ೋ": ["oo"], "ೌ": ["au"],
}
_C = {
    "ಕ": "k", "ಖ": "kh", "ಗ": "g", "ಘ": "gh", "ಙ": "ng",
    "ಚ": "c", "ಛ": "ch", "ಜ": "j", "ಝ": "jh", "ಞ": "ny",
    "ಟ": "tt", "ಠ": "tth", "ಡ": "dd", "ಢ": "ddh", "ಣ": "nn",
    "ತ": "t", "ಥ": "th", "ದ": "d", "ಧ": "dh", "ನ": "n",
    "ಪ": "p", "ಫ": "ph", "ಬ": "b", "ಭ": "bh", "ಮ": "m",
    "ಯ": "y", "ರ": "r", "ಱ": "r", "ಲ": "l", "ವ": "v",
    "ಶ": "sh", "ಷ": "ssh", "ಸ": "s", "ಹ": "h", "ಳ": "ll", "ೞ": "ll",
    "ೝ": "n",
}
_NUKTA_MAP = {"ph": "f", "j": "z"}
_VIRAMA, _ANUSVARA, _VISARGA, _CANDRABINDU, _NUKTA = "್", "ಂ", "ಃ", "ಁ", "಼"
_IGNORED = {"ಽ", "ೕ", "ೖ", "‌", "‍"}
_PLACE = {}
for _nasal, _group in (("ng", "k kh g gh"), ("ny", "c ch j jh"), ("nn", "tt tth dd ddh"),
                       ("n", "t th d dh"), ("m", "p ph b bh")):
    for _p in _group.split():
        _PLACE[_p] = _nasal

KANNADA_CHARS = "ಀ-೿‌‍"
_KANNADA_WORD = re.compile(f"[{KANNADA_CHARS}]+")
_HAS_KANNADA = re.compile("[ಀ-೿]")


def kannada_word_phones(word):
    """Kannada word -> list of phones without the ``kn_`` prefix, or None if unsupported.

    * A consonant without a vowel sign or virama keeps the inherent /a/ (also word-finally).
    * Virama removes the inherent vowel (clusters, gemination).
    * Anusvara becomes the nasal of the next consonant's place, else /m/.
    * Visarga is /h/; vocalic r is /ru/; nukta letters ಫ಼ and ಜ಼ are /f/ and /z/.
    """
    word = unicodedata.normalize("NFC", word)
    if not word or not _KANNADA_WORD.fullmatch(word) or re.search("[೦-೯]", word):
        return None
    out, pending, i, n = [], [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _IGNORED:
            i += 1
        elif ch in _C:
            cons, j = _C[ch], i + 1
            if j < n and word[j] == _NUKTA:
                cons, j = _NUKTA_MAP.get(cons, cons), j + 1
            while j < n and word[j] in _IGNORED:
                j += 1
            out.append(cons)
            if j < n and word[j] == _VIRAMA:
                j += 1
            elif j < n and word[j] in _M:
                out.extend(_M[word[j]])
                j += 1
            else:
                out.append("a")
            i = j
        elif ch in _V:
            out.extend(_V[ch])
            i += 1
        elif ch in (_ANUSVARA, _CANDRABINDU):
            pending.append(len(out))
            out.append("m")
            i += 1
        elif ch == _VISARGA:
            out.append("h")
            i += 1
        elif ch in _M or ch in (_VIRAMA, _NUKTA):
            i += 1  # stray sign without a base letter
        else:
            return None
    for pos in pending:
        nxt = out[pos + 1] if pos + 1 < len(out) else None
        out[pos] = _PLACE.get(nxt, "m")
    return out or None


# ---------------------------------------------------------------- romanized words (names)
# Latin-script words missing from the CMU dictionary are mostly Indian names and places
# (Bengaluru, Hubballi, Shetty). They are read with Indian romanization rules instead of
# failing: th -> ತ, dh -> ಧ, sh -> ಶ, ee -> ಈ, oo -> ಊ, single t/d -> ಟ/ಡ, doubled letters
# geminate. Pass guess_unknown=False to get PiVo's strict error instead.
_ROMAN = [("chh", ["ch"]), ("sh", ["sh"]), ("ch", ["c"]), ("th", ["t"]), ("dh", ["dh"]),
          ("bh", ["bh"]), ("kh", ["kh"]), ("gh", ["gh"]), ("ph", ["ph"]), ("aa", ["aa"]),
          ("ee", ["ii"]), ("ii", ["ii"]), ("oo", ["uu"]), ("uu", ["uu"]), ("ai", ["ai"]),
          ("au", ["au"]), ("ou", ["au"]), ("ow", ["au"]), ("ei", ["ee"]), ("a", ["a"]), ("e", ["e"]), ("i", ["i"]),
          ("o", ["o"]), ("u", ["u"]), ("b", ["b"]), ("c", ["k"]), ("d", ["dd"]), ("f", ["f"]),
          ("g", ["g"]), ("h", ["h"]), ("j", ["j"]), ("k", ["k"]), ("l", ["l"]), ("m", ["m"]),
          ("n", ["n"]), ("p", ["p"]), ("q", ["k"]), ("r", ["r"]), ("s", ["s"]), ("t", ["tt"]),
          ("v", ["v"]), ("w", ["v"]), ("x", ["k", "s"]), ("y", ["y"]), ("z", ["z"])]
_KN_VOWELS = set(KN_VOWELS)
_KN_TO_ARPABET = {"a": "AH", "aa": "AA", "i": "IH", "ii": "IY", "u": "UH", "uu": "UW", "e": "EH",
                  "ee": "EY", "o": "OW", "oo": "OW", "ai": "AY", "au": "AW", "k": "K", "kh": "K",
                  "g": "G", "gh": "G", "ng": "NG", "c": "CH", "ch": "CH", "j": "JH", "jh": "JH",
                  "ny": "N", "tt": "T", "tth": "T", "dd": "D", "ddh": "D", "nn": "N", "t": "T",
                  "th": "T", "d": "D", "dh": "D", "n": "N", "p": "P", "ph": "P", "b": "B", "bh": "B",
                  "m": "M", "y": "Y", "r": "R", "l": "L", "v": "V", "ll": "L", "sh": "SH",
                  "ssh": "SH", "s": "S", "h": "HH", "f": "F", "z": "Z"}


def romanized_word_phones(word):
    """'Bengaluru' -> ['b', 'e', 'n', 'g', 'a', 'l', 'u', 'r', 'u'] (Kannada phones, no prefix)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    out, i = [], 0
    while i < len(word):
        if word[i] == "y" and (i + 1 == len(word) or word[i + 1] not in "aeiou") and out \
                and out[-1] not in _KN_VOWELS:
            out.append("i" if i + 1 == len(word) else "ai")   # Shetty -> i; Mysuru, Byrappa -> ai
            i += 1
            continue
        for letters, phones in _ROMAN:
            if word.startswith(letters, i):
                out.extend(phones)
                i += len(letters)
                break
    return out


def romanized_arpabet(word):
    """The same reading in ARPAbet, for English sentences (first vowel stressed)."""
    out, stressed = [], False
    for phone in romanized_word_phones(word):
        symbol = _KN_TO_ARPABET[phone]
        if phone in _KN_VOWELS:
            symbol += "0" if stressed else "1"
            stressed = True
        out.append(symbol)
    return out


# ---------------------------------------------------------------- bilingual frontend
_PUNCT = str.maketrans({
    "।": ".", "॥": ".", "…": ".", "—": ",", "–": ",", "‘": "'", "’": "'",
    "“": " ", "”": " ", '"': " ", "(": " ", ")": " ", "[": " ", "]": " ",
    "{": " ", "}": " ", "/": " ", "-": " ", "_": " ", "*": " ",
})
_TOKEN = re.compile(f"[{KANNADA_CHARS}]+|[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]|\\S")


def normalize(text):
    text = unicodedata.normalize("NFC", text).translate(_PUNCT)
    text = text.replace("‌", "").replace("‍", "")
    text = re.sub(r"([.?!])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def phonemize(text, *, strict=True, overrides=None, guess_unknown=True):
    """Text in Kannada, English, or both -> FrontendResult with ids from ``SYMBOLS``.

    Numbers, dates, times, money and abbreviations are first written out as words
    (``normalize_kn``): in Kannada when the text contains Kannada script, otherwise in
    Indian English before PiVo's English frontend runs.
    """
    merged = dict(INDIAN_ENGLISH_LEXICON)
    merged.update(overrides or {})
    if not _HAS_KANNADA.search(text):
        english = normalize_indian_english(text)
        if guess_unknown:
            try:
                words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", normalize_english(english))
            except ValueError:
                words = []                            # let the English frontend report it
            dictionary, lowered = lexicon(), {k.lower() for k in merged}
            for word in {w.lower() for w in words}:
                if word not in dictionary and word not in lowered and re.fullmatch("[a-z]+", word):
                    merged[word] = romanized_arpabet(word)
        return phonemize_english(english, strict=strict, overrides=merged)
    normalized = normalize(normalize_kannada_text(text))
    over = {}
    for word, phones in merged.items():
        word = unicodedata.normalize("NFC", word)
        over[word if _HAS_KANNADA.search(word) else word.lower()] = phones
    dictionary = lexicon()
    tokens, unknown = ["<bos>"], []
    for item in _TOKEN.findall(normalized):
        if item in SPECIAL:
            if tokens[-1] != "<bos>":            # leading punctuation carries no sound
                tokens.append(item)
            continue
        if _KANNADA_WORD.fullmatch(item):
            phones = over.get(item)
            if phones is None:
                bare = kannada_word_phones(item)
                if bare is None:
                    raise ValueError(f"cannot read Kannada word {item!r}; write numbers in words "
                                     "or add a pronunciation override")
                phones = ["kn_" + p for p in bare]
        elif re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)*", item):
            word = item.lower()
            phones = over.get(word)
            if phones is None and word in dictionary:
                phones = dictionary[word][0]
            if phones is None:
                unknown.append(word)
                if guess_unknown:
                    phones = ["kn_" + p for p in romanized_word_phones(word)]
                else:
                    phones = [p for c in word if c.isalpha() for p in dictionary[c][0]]
        elif item.isdigit():
            raise ValueError(f"could not read the number {item!r}")
        else:
            raise ValueError(f"unsupported character in Kannada text: {item!r}")
        if tokens[-1] not in ("<bos>", "<space>"):
            tokens.append("<space>")
        for phone in phones:
            if phone not in SYMBOL_TO_ID:
                raise ValueError(f"unknown phone {phone!r} in {item!r}")
            tokens.append(phone)
    if unknown and strict and not guess_unknown:
        raise ValueError("out-of-vocabulary words require pronunciation overrides: "
                         + ", ".join(sorted(set(unknown))))
    tokens.append("<eos>")
    return FrontendResult(normalized, tuple(tokens), tuple(SYMBOL_TO_ID[s] for s in tokens), tuple(unknown))
