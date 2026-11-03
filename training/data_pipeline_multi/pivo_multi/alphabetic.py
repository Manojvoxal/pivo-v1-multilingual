"""G2P for the two non-Brahmic Indian scripts used in Rasa: Ol Chiki (Santali) and
Meetei Mayek (Manipuri). Both map almost letter-for-letter to phones.

Ol Chiki is a true alphabet (vowels are written as letters, no inherent vowel).
Meetei Mayek is an abugida with an inherent /ə/, vowel signs, final "lonsum" letters
and apun iyek (virama).
"""

import unicodedata

from .symbols import IN_VOWELS, nasalize

# ------------------------------------------------------------------ Ol Chiki (Santali)
OL_LETTERS = {
    "ᱚ": ("V", "aw"), "ᱛ": ("C", "t"), "ᱜ": ("C", "g"), "ᱝ": ("C", "ng"), "ᱞ": ("C", "l"),
    "ᱟ": ("V", "a"), "ᱠ": ("C", "k"), "ᱡ": ("C", "j"), "ᱢ": ("C", "m"), "ᱣ": ("C", "v"),
    "ᱤ": ("V", "i"), "ᱥ": ("C", "s"), "ᱦ": ("C", "h"), "ᱧ": ("C", "ny"), "ᱨ": ("C", "r"),
    "ᱩ": ("V", "u"), "ᱪ": ("C", "c"), "ᱫ": ("C", "d"), "ᱬ": ("C", "nn"), "ᱭ": ("C", "y"),
    "ᱮ": ("V", "e"), "ᱯ": ("C", "p"), "ᱰ": ("C", "dd"), "ᱱ": ("C", "n"), "ᱲ": ("C", "rr"),
    "ᱳ": ("V", "o"), "ᱴ": ("C", "tt"), "ᱵ": ("C", "b"), "ᱶ": ("C", "v"),
}
OL_ASPIRATE = "ᱷ"        # makes the previous stop aspirated
OL_NASAL = {"ᱸ", "ᱺ"}    # mu ttuddag / mu-gaahlaa ttuddaag: nasal vowel
OL_MODIFIER = {"ᱹ", "ᱺ"}  # gaahlaa ttuddaag: central vowel /ə/
OL_RELAA = "ᱻ"           # vowel length
OL_PHAARKA, OL_AHAD = "ᱼ", "ᱽ"
OL_PUNCT = {"᱾", "᱿"}
OL_CHECKED = {"g": "k", "j": "c", "d": "t", "b": "p"}  # word-final ᱜ ᱡ ᱫ ᱵ are checked stops
ASPIRATE = {"k": "kh", "g": "gh", "c": "ch", "j": "jh", "t": "th", "d": "dh", "tt": "tth",
            "dd": "ddh", "p": "ph", "b": "bh", "rr": "rrh"}
LONG = {"a": "aa", "i": "ii", "u": "uu", "e": "ee", "o": "oo"}


def olchiki_to_phones(word):
    word = unicodedata.normalize("NFC", word)
    out = []
    ahad_after = set()
    for idx, ch in enumerate(word):
        if ch in OL_LETTERS:
            out.append(OL_LETTERS[ch][1])
        elif ch == OL_ASPIRATE:
            if out and out[-1] in ASPIRATE:
                out[-1] = ASPIRATE[out[-1]]
            else:
                out.append("h")
        elif ch in OL_MODIFIER or ch in OL_NASAL:
            if out and out[-1] in IN_VOWELS:
                if ch in OL_MODIFIER:
                    out[-1] = "a"                    # ᱚᱹ ᱮᱹ -> /ə/
                if ch in OL_NASAL:
                    out[-1] = nasalize(out[-1])
        elif ch == OL_RELAA:
            if out and out[-1] in LONG:
                out[-1] = LONG[out[-1]]
        elif ch == OL_AHAD:
            ahad_after.add(len(out) - 1)
        elif ch == OL_PHAARKA or ch in ("‌", "‍"):
            continue
        elif 0x1C50 <= ord(ch) <= 0x1C59:
            return None                              # Ol Chiki digits: needs number expansion
        else:
            return None
    if not out:
        return None
    last = len(out) - 1
    if out[last] in OL_CHECKED and last not in ahad_after:
        out[last] = OL_CHECKED[out[last]]
    return out


# ------------------------------------------------------------------ Meetei Mayek (Manipuri)
MM_CONS = {
    "ꯀ": "k", "ꯁ": "s", "ꯂ": "l", "ꯃ": "m", "ꯄ": "p", "ꯅ": "n", "ꯆ": "c", "ꯇ": "t",
    "ꯈ": "kh", "ꯉ": "ng", "ꯊ": "th", "ꯋ": "v", "ꯌ": "y", "ꯍ": "h", "ꯐ": "ph",
    "ꯒ": "g", "ꯓ": "jh", "ꯔ": "r", "ꯕ": "b", "ꯖ": "j", "ꯗ": "d", "ꯘ": "gh",
    "ꯙ": "dh", "ꯚ": "bh",
    # Meetei Mayek extensions (Sanskrit/Bengali loans)
    "ꫢ": "c", "ꫣ": "nn", "ꫤ": "tt", "ꫥ": "tth", "ꫦ": "dd", "ꫧ": "ddh", "ꫨ": "nn",
    "ꫩ": "sh", "ꫪ": "ssh",
}
MM_VOWEL_LETTERS = {"ꯑ": None, "ꯎ": "u", "ꯏ": "i"}   # ꯑ = vowel carrier (inherent /ə/)
MM_LONSUM = {"ꯛ": "k", "ꯜ": "l", "ꯝ": "m", "ꯞ": "p", "ꯟ": "n", "ꯠ": "t", "ꯡ": "ng", "ꯢ": "i"}
MM_SIGNS = {"ꯣ": "o", "ꯤ": "i", "ꯥ": "aa", "ꯦ": "e", "ꯧ": "au", "ꯨ": "u", "ꯩ": "ai",
            "ꫫ": "ii", "ꫬ": "uu", "ꫭ": "ai", "ꫮ": "au", "ꫯ": "au"}
MM_NUNG = "ꯪ"     # anusvara -> /ŋ/
MM_APUN = "꯭"     # virama
MM_LUM = "꯬"      # heavy tone mark
MM_EXT_VIRAMA = "꫶"


def meetei_to_phones(word):
    word = unicodedata.normalize("NFC", word)
    out = []
    chars = list(word)
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        if ch in MM_CONS or ch in MM_VOWEL_LETTERS:
            if ch in MM_CONS:
                out.append(MM_CONS[ch])
            j = i + 1
            while j < n and chars[j] == MM_LUM:
                j += 1
            if j < n and chars[j] in (MM_APUN, MM_EXT_VIRAMA):
                i = j + 1
                continue
            if j < n and chars[j] in MM_SIGNS:
                out.append(MM_SIGNS[chars[j]])
                i = j + 1
                continue
            if ch in MM_VOWEL_LETTERS and MM_VOWEL_LETTERS[ch] is not None:
                out.append(MM_VOWEL_LETTERS[ch])
            else:
                out.append("a")                       # inherent /ə/
            i = j
            continue
        if ch in MM_LONSUM:
            out.append(MM_LONSUM[ch])
        elif ch == MM_NUNG:
            out.append("ng")
        elif ch in (MM_LUM, "‌", "‍", "ꫵ"):
            pass
        elif 0xABF0 <= ord(ch) <= 0xABF9:
            return None                               # Meetei digits
        elif ch in MM_SIGNS:
            out.append(MM_SIGNS[ch])                  # stray sign: read the vowel
        else:
            return None
        i += 1
    if not out:
        return None
    m = {"ii": "i", "uu": "u", "ee": "e", "oo": "o"}  # no phonemic vowel length
    return [m.get(p, p) for p in out]
