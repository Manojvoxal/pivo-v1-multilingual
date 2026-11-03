"""Shared symbol table: PiVo's original 79 English symbols (ids 0-78, unchanged)
plus 48 Kannada phones (ids 79-126). Total 127 <= PiVo's VOCAB_SIZE of 128,
so the pretrained token-embedding table keeps its shape and every English id
keeps its pretrained row.
"""

PIVO_SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"]
PIVO_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
PIVO_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
PIVO_SYMBOLS = tuple(
    PIVO_SPECIAL + PIVO_CONSONANTS + [v + s for v in PIVO_VOWELS for s in "012"]
)

KN_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"]
KN_CONSONANTS = [
    "k", "kh", "g", "gh", "ng",          # velar
    "c", "ch", "j", "jh", "ny",          # palatal
    "tt", "tth", "dd", "ddh", "nn",      # retroflex
    "t", "th", "d", "dh", "n",           # dental
    "p", "ph", "b", "bh", "m",           # labial
    "y", "r", "l", "v", "ll",            # approximants (ll = retroflex lateral)
    "sh", "ssh", "s", "h",               # sibilants / glottal
    "f", "z",                            # nukta loans
]
KN_PHONES = tuple("kn_" + p for p in KN_VOWELS + KN_CONSONANTS)

SYMBOLS = PIVO_SYMBOLS + KN_PHONES
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = 128
NUM_SPECIAL = len(PIVO_SPECIAL)  # ids < 10 are non-phone tokens

assert len(PIVO_SYMBOLS) == 79
assert len(KN_PHONES) == 48
assert len(SYMBOLS) <= VOCAB_SIZE
assert len(set(SYMBOLS)) == len(SYMBOLS)


def is_phone(symbol: str) -> bool:
    return SYMBOL_TO_ID[symbol] >= NUM_SPECIAL
