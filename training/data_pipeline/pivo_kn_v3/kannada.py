"""Rule-based Kannada grapheme-to-phoneme for the PiVo Kannada phone set.

Kannada spelling is close to phonemic, so the rules are short:
  * a consonant with no vowel sign and no virama carries the inherent vowel /a/
    (Kannada keeps it word-finally, so there is no schwa deletion);
  * virama (್) removes the inherent vowel, so a cluster or gemination follows;
  * anusvara (ಂ) becomes the nasal that matches the following consonant's place
    of articulation, and /m/ before non-stops or at the end of a word;
  * visarga (ಃ) becomes /h/;  ಋ / ೃ are read /ru/ as in modern Kannada;
  * ಫ಼ -> /f/, ಜ಼ -> /z/ (nukta loans); other nukta letters keep their base sound.

Words containing digits or characters outside Kannada script return None, so
the caller can drop or override them instead of guessing.
"""

import re
import unicodedata

V = {  # independent vowels
    "ಅ": ["a"], "ಆ": ["aa"], "ಇ": ["i"], "ಈ": ["ii"], "ಉ": ["u"], "ಊ": ["uu"],
    "ಋ": ["r", "u"], "ೠ": ["r", "uu"], "ಌ": ["l", "u"], "ೡ": ["l", "uu"],
    "ಎ": ["e"], "ಏ": ["ee"], "ಐ": ["ai"], "ಒ": ["o"], "ಓ": ["oo"], "ಔ": ["au"],
}
M = {  # dependent vowel signs (matras)
    "ಾ": ["aa"], "ಿ": ["i"], "ೀ": ["ii"], "ು": ["u"], "ೂ": ["uu"],
    "ೃ": ["r", "u"], "ೄ": ["r", "uu"], "ೢ": ["l", "u"], "ೣ": ["l", "uu"],
    "ೆ": ["e"], "ೇ": ["ee"], "ೈ": ["ai"], "ೊ": ["o"], "ೋ": ["oo"], "ೌ": ["au"],
}
C = {
    "ಕ": "k", "ಖ": "kh", "ಗ": "g", "ಘ": "gh", "ಙ": "ng",
    "ಚ": "c", "ಛ": "ch", "ಜ": "j", "ಝ": "jh", "ಞ": "ny",
    "ಟ": "tt", "ಠ": "tth", "ಡ": "dd", "ಢ": "ddh", "ಣ": "nn",
    "ತ": "t", "ಥ": "th", "ದ": "d", "ಧ": "dh", "ನ": "n",
    "ಪ": "p", "ಫ": "ph", "ಬ": "b", "ಭ": "bh", "ಮ": "m",
    "ಯ": "y", "ರ": "r", "ಱ": "r", "ಲ": "l", "ವ": "v",
    "ಶ": "sh", "ಷ": "ssh", "ಸ": "s", "ಹ": "h", "ಳ": "ll", "ೞ": "ll",
    "ೝ": "n",  # nakaara pollu: a bare /n/
}
NUKTA_MAP = {"ph": "f", "j": "z"}
VIRAMA, ANUSVARA, VISARGA, CANDRABINDU, NUKTA = "್", "ಂ", "ಃ", "ಁ", "಼"
IGNORED = {"ಽ", "ೕ", "ೖ", "‌", "‍"}  # avagraha, length marks, ZWNJ, ZWJ

PLACE = {}
for nasal, group in (
    ("ng", "k kh g gh"), ("ny", "c ch j jh"), ("nn", "tt tth dd ddh"),
    ("n", "t th d dh"), ("m", "p ph b bh"),
):
    for p in group.split():
        PLACE[p] = nasal

KANNADA_RE = re.compile(r"[ಀ-೿‌‍]+")


def word_to_phones(word: str):
    """Return a list of bare phone names (no kn_ prefix), or None if unsupported."""
    word = unicodedata.normalize("NFC", word)
    if not word or not KANNADA_RE.fullmatch(word):
        return None
    if re.search(r"[೦-೯]", word):  # Kannada digits need number expansion first
        return None
    out = []           # phones so far
    pending = []       # anusvara positions to resolve once the next phone is known
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in IGNORED:
            i += 1
            continue
        if ch in C:
            cons = C[ch]
            j = i + 1
            if j < n and word[j] == NUKTA:
                cons = NUKTA_MAP.get(cons, cons)
                j += 1
            while j < n and word[j] in IGNORED:
                j += 1
            out.append(cons)
            if j < n and word[j] == VIRAMA:
                j += 1
            elif j < n and word[j] in M:
                out.extend(M[word[j]])
                j += 1
            else:
                out.append("a")
            i = j
            continue
        if ch in V:
            out.extend(V[ch])
            i += 1
            continue
        if ch in (ANUSVARA, CANDRABINDU):
            pending.append(len(out))
            out.append("m")  # placeholder, resolved below
            i += 1
            continue
        if ch == VISARGA:
            out.append("h")
            i += 1
            continue
        if ch in M or ch == VIRAMA or ch == NUKTA:
            i += 1  # stray sign with no base letter: skip it
            continue
        return None  # anything else (unknown code point)
    for pos in pending:
        nxt = out[pos + 1] if pos + 1 < len(out) else None
        out[pos] = PLACE.get(nxt, "m")
    return out or None


def word_to_symbols(word: str):
    phones = word_to_phones(word)
    return None if phones is None else ["kn_" + p for p in phones]
