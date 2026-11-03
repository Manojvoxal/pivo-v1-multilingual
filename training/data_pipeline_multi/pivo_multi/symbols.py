"""Token table for the multilingual PiVo model (trained from scratch).

Layout (ids are fixed; append new symbols only at the end):
  0-9     PiVo specials: <pad> <bos> <eos> <space> . , ? ! ; :
  10-78   ARPAbet with stress (English words, also English words inside Indic text)
  79-...  shared Indic phones, prefix "in_". One inventory for all 22 Indian
          languages: the same sound gets the same token in every language, and the
          language embedding in the model handles accent differences.

The Indic inventory is "phonemic, close to the Brahmic letters": the rule-based G2P
(brahmic.py, alphabetic.py) does schwa deletion, nasalisation, Tamil voicing etc., so
the network only has to learn phones -> sound.
"""

PIVO_SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"]
PIVO_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
PIVO_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
ARPABET = tuple(PIVO_CONSONANTS + [v + s for v in PIVO_VOWELS for s in "012"])

IN_VOWELS = [
    "a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au",
    "ae",   # æ   Marathi/Hindi ॲ ऍ, Gujarati ઍ (English loans)
    "aw",   # ɔ   Bengali/Assamese/Odia/Santali inherent vowel, Hindi ऑ
    "ix",   # ɨ   Malayalam samvruthokaram, Kashmiri
]
IN_NASAL_VOWELS = [v + "~" for v in
                   ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au", "ae", "aw"]]
IN_CONSONANTS = [
    "k", "kh", "g", "gh", "ng",          # velar
    "c", "ch", "j", "jh", "ny",          # palatal
    "tt", "tth", "dd", "ddh", "nn",      # retroflex
    "t", "th", "d", "dh", "n",           # dental
    "p", "ph", "b", "bh", "m",           # labial
    "y", "r", "l", "v", "ll",            # approximants (ll = retroflex lateral ळ ಳ ள)
    "sh", "ssh", "s", "h",               # sibilants / glottal
    "f", "z",                            # nukta loans फ़ ज़
    "q", "x", "gx",                      # क़ ख़ ग़ (Urdu/Hindi loans; Assamese x)
    "rr", "rrh",                         # ड़ ढ़ retroflex flap
    "zh",                                # ழ ഴ retroflex approximant
    "rx", "tx", "nx",                    # ற / റ, ற்ற / റ്റ, ன / alveolar ന
    "bq", "dq", "jq", "gq",              # Sindhi implosives ॿ ॾ ॼ ॻ (phase 3)
]
IN_PHONES = tuple("in_" + p for p in IN_VOWELS + IN_NASAL_VOWELS + IN_CONSONANTS)

SYMBOLS = tuple(PIVO_SPECIAL) + ARPABET + IN_PHONES
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = 160  # embedding rows; a few spare rows for later additions
NUM_SPECIAL = len(PIVO_SPECIAL)
ENGLISH_ID_RANGE = (NUM_SPECIAL, NUM_SPECIAL + len(ARPABET) - 1)
INDIC_ID_RANGE = (NUM_SPECIAL + len(ARPABET), len(SYMBOLS) - 1)

IN_VOWEL_SET = set(IN_VOWELS) | set(IN_NASAL_VOWELS)
IN_STOPS_BY_PLACE = {}
for _nasal, _group in (("ng", "k kh g gh q x gx"), ("ny", "c ch j jh"),
                       ("nn", "tt tth dd ddh"), ("n", "t th d dh"), ("m", "p ph b bh f")):
    for _p in _group.split():
        IN_STOPS_BY_PLACE[_p] = _nasal

assert len(ARPABET) == 69
assert len(SYMBOLS) <= VOCAB_SIZE, len(SYMBOLS)
assert len(set(SYMBOLS)) == len(SYMBOLS)


def is_phone(symbol: str) -> bool:
    return SYMBOL_TO_ID[symbol] >= NUM_SPECIAL


def mfa_phone(symbol: str) -> str:
    """Phone name as written in MFA lexicons. MFA drops everything after '_' when it
    loads a saved model, so '_' is removed; '~' (nasal vowel) becomes 'N'.
    The mapping is injective: Indic names are lowercase with an 'in' prefix,
    ARPAbet names are uppercase."""
    return symbol.replace("_", "").replace("~", "N")


def nasalize(phone: str) -> str:
    """'aa' -> 'aa~' (bare names, no prefix). Non-vowels are returned unchanged."""
    if phone in IN_VOWELS and phone != "ix":
        return phone + "~"
    return phone
