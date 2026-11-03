"""Rule-based G2P for the nine Brahmic scripts (Devanagari, Bengali-Assamese, Gurmukhi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam).

All nine Unicode blocks share the ISCII layout, so one parser works on the offset of a
character inside its block (0x15 = KA in every script). Language rules on top:

  * inherent vowel: /a/ for most, /ɔ/ ("aw") for Bengali, Assamese, Odia
  * schwa deletion: none (Dravidian, Odia, Sanskrit), final only, or final + the
    VC_CV medial rule (Hindi, Marathi, Gujarati, Punjabi, Maithili, Dogri, Bengali)
  * anusvara / candrabindu / tippi: homorganic nasal before a stop; otherwise a
    nasalised vowel (Indo-Aryan) or /m/ (Dravidian, Sanskrit)
  * Tamil: voicing of k c ṭ t p by position (initial, geminate, after nasal, between vowels)
  * Malayalam: chillus, word-final virama = samvruthokaram /ɨ/, alveolar ന, ന്റ, റ്റ
  * Bengali / Assamese: ya-phala and ba-phala gemination, sibilant merger,
    Assamese c->s, j->z, ś/ṣ/s->x, no retroflex series
  * Gurmukhi: addak (gemination), tippi/bindi

word_to_phones() returns bare phone names (no "in_" prefix) or None when the word
cannot be read (digits, unknown code points, mixed scripts).
"""

import unicodedata

from .langs import LANG
from .scripts import BRAHMIC_BASE, ZWJ, ZWNJ, script_of_word
from .symbols import IN_STOPS_BY_PLACE, IN_VOWELS, nasalize

# ------------------------------------------------------------------ offset tables
IND_VOWEL = {0x05: "@IND", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
             0x0B: "VR", 0x0C: "VL", 0x0D: "ae", 0x0E: "e", 0x0F: "ee", 0x10: "ai",
             0x11: "aw", 0x12: "o", 0x13: "oo", 0x14: "au", 0x60: "VRR", 0x61: "VLL"}
MATRA = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "VR", 0x44: "VRR",
         0x45: "ae", 0x46: "e", 0x47: "ee", 0x48: "ai", 0x49: "aw", 0x4A: "o", 0x4B: "oo",
         0x4C: "au", 0x62: "VL", 0x63: "VLL"}
CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "c", 0x1B: "ch",
        0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "tt", 0x20: "tth", 0x21: "dd", 0x22: "ddh",
        0x23: "nn", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nx",
        0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
        0x31: "rx", 0x32: "l", 0x33: "ll", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "ssh",
        0x38: "s", 0x39: "h"}
CANDRABINDU, ANUSVARA, VISARGA, NUKTA, AVAGRAHA, VIRAMA, OM = 0x01, 0x02, 0x03, 0x3C, 0x3D, 0x4D, 0x50
DANDA_CPS = {0x0964, 0x0965}
NUKTA_MAP = {"k": "q", "kh": "x", "g": "gx", "j": "z", "jh": "z", "dd": "rr", "ddh": "rrh",
             "ph": "f"}

# per-script extras: offset -> ("C", phone) consonant, ("C0", phone) consonant without
# vowel (chillu-like), ("V", phone) independent vowel, ("M", phone) vowel sign,
# ("NAS", kind), ("GEM",), ("IGN",)
EXTRA = {
    "devanagari": {0x58: ("C", "q"), 0x59: ("C", "x"), 0x5A: ("C", "gx"), 0x5B: ("C", "z"),
                   0x5C: ("C", "rr"), 0x5D: ("C", "rrh"), 0x5E: ("C", "f"), 0x5F: ("C", "y"),
                   0x3A: ("M", "o"), 0x3B: ("M", "oo"), 0x4E: ("IGN",), 0x4F: ("M", "aw"),
                   0x55: ("IGN",), 0x56: ("M", "u"), 0x57: ("M", "uu"),
                   0x70: ("IGN",), 0x71: ("IGN",), 0x72: ("V", "ae"), 0x73: ("V", "o"),
                   0x74: ("V", "oo"), 0x75: ("V", "aw"), 0x76: ("V", "u"), 0x77: ("V", "uu"),
                   0x79: ("C", "z"), 0x7A: ("C", "y"), 0x7B: ("C", "gq"), 0x7C: ("C", "jq"),
                   0x7D: ("IGN",), 0x7E: ("C", "dq"), 0x7F: ("C", "bq"), 0x00: ("NAS", "candrabindu")},
    "bengali": {0x5C: ("C", "rr"), 0x5D: ("C", "rrh"), 0x5F: ("C", "y"), 0x4E: ("C0", "t"),
                0x70: ("C", "r"), 0x71: ("C", "v"), 0x57: ("IGN",), 0x72: ("IGN",), 0x73: ("IGN",),
                0x7A: ("IGN",), 0x7C: ("IGN",), 0x7D: ("IGN",), 0x7E: ("IGN",)},
    "gurmukhi": {0x59: ("C", "x"), 0x5A: ("C", "gx"), 0x5B: ("C", "z"), 0x5C: ("C", "rr"),
                 0x5E: ("C", "f"), 0x70: ("NAS", "anusvara"), 0x71: ("GEM",), 0x72: ("BEARER", "i"),
                 0x73: ("BEARER", "u"), 0x74: ("IGN",), 0x75: ("C", "y"), 0x51: ("IGN",)},
    "gujarati": {0x7A: ("IGN",), 0x7B: ("IGN",), 0x7C: ("IGN",), 0x7D: ("IGN",), 0x7E: ("IGN",),
                 0x7F: ("IGN",)},
    "odia": {0x5C: ("C", "rr"), 0x5D: ("C", "rrh"), 0x5F: ("C", "y"), 0x71: ("C", "v"),
             0x56: ("IGN",), 0x57: ("IGN",), 0x70: ("IGN",)},
    "tamil": {0x57: ("IGN",)},
    "telugu": {0x58: ("C", "c"), 0x59: ("C", "j"), 0x5A: ("C", "r"), 0x5D: ("C0", "n"),
               0x00: ("IGN",), 0x01: ("IGN",), 0x04: ("IGN",), 0x55: ("IGN",), 0x56: ("IGN",)},
    "kannada": {0x5D: ("C0", "n"), 0x5E: ("C", "ll"), 0x55: ("IGN",), 0x56: ("IGN",),
                0x71: ("IGN",), 0x72: ("IGN",), 0x00: ("IGN",), 0x04: ("IGN",)},
    "malayalam": {0x54: ("C0", "m"), 0x55: ("C0", "y"), 0x56: ("C0", "zh"), 0x7A: ("C0", "nn"),
                  0x7B: ("C0", "nx"), 0x7C: ("C0", "r"), 0x7D: ("C0", "l"), 0x7E: ("C0", "ll"),
                  0x7F: ("C0", "k"), 0x4E: ("C0", "r"), 0x57: ("M", "au"), 0x00: ("IGN",),
                  0x3B: ("IGN",), 0x3C: ("IGN",), 0x4F: ("IGN",)},
}
IGNORE_OFFSETS = {AVAGRAHA, 0x51, 0x52, 0x53, 0x54}
DIGIT_OFFSETS = set(range(0x66, 0x70))

ASPIRATE_TO_PLAIN = {"kh": "k", "gh": "g", "ch": "c", "jh": "j", "tth": "tt", "ddh": "dd",
                     "th": "t", "dh": "d", "ph": "p", "bh": "b", "rrh": "rr"}
NASALS = {"ng", "ny", "nn", "n", "m", "nx"}
VOICELESS_OBSTRUENTS = {"k", "c", "tt", "t", "p", "s", "sh", "ssh", "tx", "kh", "ch", "tth",
                        "th", "ph", "f", "x", "q", "h"}
SIBILANT_CLUSTER_NEXT = {"t", "th", "tt", "tth", "n", "nn", "l", "r", "k", "kh", "p", "ph", "m"}

# language-specific letter overrides (phone -> phone), applied to consonants
OVERRIDES = {
    "hi": {"ssh": "sh"}, "mai": {"ssh": "sh"}, "doi": {"ssh": "sh"}, "gu": {"ssh": "sh"},
    "pa": {"ssh": "sh"}, "kok": {"ssh": "sh"}, "brx": {"ssh": "sh"},
    "ne": {"sh": "s", "ssh": "s"},
    "or": {"sh": "s", "ssh": "s"},
    "bn": {"ssh": "sh", "s": "sh", "v": "b"},
    "as": {"c": "s", "ch": "s", "j": "z", "jh": "z", "tt": "t", "tth": "th", "dd": "d",
           "ddh": "dh", "nn": "n", "sh": "x", "ssh": "x", "s": "x", "rr": "r", "rrh": "r"},
    "mni": {"ssh": "s", "sh": "s", "v": "b"},
}
# ज्ञ-type cluster (j + ny)
JNY = {"hi": ["g", "y"], "mai": ["g", "y"], "doi": ["g", "y"], "ne": ["g", "y"], "kok": ["g", "y"],
       "brx": ["g", "y"], "pa": ["g", "y"], "or": ["g", "y"], "mr": ["d", "ny"], "gu": ["g", "n"],
       "bn": ["g", "g"], "as": ["g", "g"], "mni": ["g", "y"]}


class _Item:
    __slots__ = ("kind", "phone", "off", "deleted", "out")

    def __init__(self, kind, phone=None, off=None):
        self.kind, self.phone, self.off = kind, phone, off
        self.deleted = False
        self.out = None  # final phones (list) for this item

    def __repr__(self):
        return f"{self.kind}:{self.phone}"


def _vowel_phones(v, cfg):
    r_vowel = cfg["vocalic_r"][1]
    long_v = {"i": "ii", "u": "uu"}[r_vowel]
    if v == "VR":
        return ["r", r_vowel]
    if v == "VRR":
        return ["r", long_v]
    if v == "VL":
        return ["l", r_vowel]
    if v == "VLL":
        return ["l", long_v]
    if v == "@IND":
        return [cfg["inherent"]]
    return [v]


def _parse(word, script, lang):
    """Characters -> items. Returns None on unsupported input."""
    base = BRAHMIC_BASE[script]
    extra = EXTRA.get(script, {})
    items = []
    chars = list(word)
    n = len(chars)
    i = 0

    def attach_vowel(j):
        """After a consonant at index < j: read virama / vowel sign / nothing.
        Returns new index."""
        while j < n and chars[j] == ZWNJ:
            j += 1
        if j < n and ord(chars[j]) - base == VIRAMA and BRAHMIC_BASE.get(script) is not None:
            j += 1
            if j < n and chars[j] == ZWJ:          # old-style Malayalam chillu / explicit half form
                return j + 1
            if script == "malayalam":
                k = j
                while k < n and chars[k] in (ZWNJ, ZWJ):
                    k += 1
                if k >= n:                           # word-final virama = samvruthokaram
                    items.append(_Item("HALEND"))
            return j
        if j < n and base <= ord(chars[j]) < base + 0x80:
            o = ord(chars[j]) - base
            if o in MATRA:
                items.append(_Item("V", MATRA[o], o))
                return j + 1
            ex = extra.get(o)
            if ex and ex[0] == "M":
                items.append(_Item("V", ex[1], o))
                return j + 1
        items.append(_Item("@"))
        return j

    while i < n:
        ch = chars[i]
        if ch in (ZWJ, ZWNJ):
            i += 1
            continue
        cp = ord(ch)
        if cp in DANDA_CPS:
            i += 1
            continue
        if not (base <= cp < base + 0x80):
            return None
        o = cp - base
        if o in DIGIT_OFFSETS:
            return None
        ex = extra.get(o)
        if ex is not None:
            kind = ex[0]
            if kind == "IGN":
                i += 1
            elif kind == "C":
                items.append(_Item("C", ex[1], o))
                i = attach_vowel(i + 1)
            elif kind == "C0":
                items.append(_Item("C", ex[1], o))
                i += 1
            elif kind == "V":
                items.append(_Item("V", ex[1], o))
                i += 1
            elif kind == "M":
                i += 1  # vowel sign with no consonant: skip
            elif kind == "NAS":
                items.append(_Item("NAS", ex[1], o))
                i += 1
            elif kind == "GEM":
                items.append(_Item("GEM", None, o))
                i += 1
            elif kind == "BEARER":
                j = i + 1
                if j < n and base <= ord(chars[j]) < base + 0x80 and (ord(chars[j]) - base) in MATRA:
                    items.append(_Item("V", MATRA[ord(chars[j]) - base], o))
                    j += 1
                else:
                    items.append(_Item("V", ex[1], o))
                i = j
            continue
        if o in CONS:
            phone = CONS[o]
            if o == 0x29:
                phone = "nx" if lang in ("ta", "ml") else "n"
            elif o == 0x31:
                phone = "rx" if lang in ("ta", "ml") else "r"
            elif o == 0x34:
                phone = "zh" if lang in ("ta", "ml") else "ll"
            j = i + 1
            if j < n and ord(chars[j]) - base == NUKTA:
                phone = NUKTA_MAP.get(phone, phone)
                if o == 0x2F:
                    o = 0x5F          # য় / य़ (decomposed by NFC) = /y/, not য
                j += 1
            items.append(_Item("C", phone, o))
            i = attach_vowel(j)
            continue
        if o in IND_VOWEL:
            items.append(_Item("V", IND_VOWEL[o], o))
            i += 1
            continue
        if o in (ANUSVARA, CANDRABINDU):
            items.append(_Item("NAS", "anusvara" if o == ANUSVARA else "candrabindu", o))
            i += 1
            continue
        if o == VISARGA:
            items.append(_Item("AYTHAM" if script == "tamil" else "VIS", None, o))
            i += 1
            continue
        if o == OM:
            items.append(_Item("V", "oo", o))
            items.append(_Item("C", "m", o))
            i += 1
            continue
        if o in IGNORE_OFFSETS or o in (NUKTA, VIRAMA) or o in MATRA:
            i += 1  # stray sign
            continue
        return None
    return items


def _slots(items):
    return [k for k, it in enumerate(items) if it.kind in ("V", "@")]


def _onset_len(items, slots, s_idx):
    lo = slots[s_idx - 1] + 1 if s_idx > 0 else 0
    return sum(1 for it in items[lo:slots[s_idx]] if it.kind in ("C", "NAS", "GEM", "VIS"))


def _onset_phones(items, slots, s_idx):
    lo = slots[s_idx - 1] + 1 if s_idx > 0 else 0
    return [it.phone for it in items[lo:slots[s_idx]] if it.kind == "C"]


def _delete_schwas(items, mode, keep_after_cluster=False):
    if mode in (None, "none"):
        return
    slots = _slots(items)
    if len(slots) < 2:
        return
    last = slots[-1]
    it = items[last]
    if it.kind == "@" and last == len(items) - 1:
        onset = _onset_phones(items, slots, len(slots) - 1)
        prev = items[slots[-2]]
        keep = False
        if onset == ["y"] and prev.kind == "V" and prev.phone in ("i", "ii"):
            keep = True                        # प्रिय priya
        elif len(onset) >= 2 and (onset[-1] in ("r", "y", "v") or keep_after_cluster):
            keep = True                        # मित्र mitra, सत्य satya (but धर्म dharm); bn ধর্ম dhɔrmo
        elif not onset:
            keep = True
        if not keep:
            it.deleted = True
    if mode != "medial":
        return
    for s in range(len(slots) - 2, 0, -1):
        it = items[slots[s]]
        if it.kind != "@" or it.deleted:
            continue
        nxt = items[slots[s + 1]]
        if nxt.deleted:
            continue
        if _onset_len(items, slots, s) == 1 and _onset_len(items, slots, s + 1) == 1:
            # the next vowel slot must start right after one consonant (no NAS after @)
            if items[slots[s] + 1].kind == "C":
                it.deleted = True


def _cluster_rules(items, lang):
    """Bengali/Assamese ya- and ba-phala, ज्ञ, क्ष, standalone য / ଯ."""
    out = []
    for it in items:
        prev = out[-1] if out else None
        after_cons = it.kind == "C" and prev is not None and prev.kind == "C"
        if after_cons:
            word_initial = all(x.kind == "C" for x in out)
            phala = (it.off == 0x2F and it.phone == "y" and prev.phone != "r") or \
                    (it.off == 0x2C and it.phone == "b" and prev.phone not in ("r", "m"))
            if lang in ("bn", "as") and phala:
                if not word_initial:   # বাক্য bakko, বিশ্বাস biʃʃash; word-initial স্বামী, ব্যাগ: dropped
                    out.append(_Item("C", ASPIRATE_TO_PLAIN.get(prev.phone, prev.phone), it.off))
                continue
            if prev.phone == "j" and prev.off == 0x1C and it.phone == "ny" and lang in JNY:
                out.pop()
                for p in JNY[lang]:
                    out.append(_Item("C", p, 0x1C))
                continue
            if prev.phone == "k" and it.off == 0x37 and lang in ("bn", "as"):
                out.pop()                      # ক্ষ -> kh (khy)
                if not word_initial:
                    out.append(_Item("C", "k", 0x15))
                out.append(_Item("C", "kh", 0x16))
                continue
        if it.kind == "C" and it.off == 0x2F and it.phone == "y":
            if lang in ("bn", "as") and (not after_cons or prev.phone == "r"):
                it = _Item("C", "j" if lang == "bn" else "z", it.off)   # য = /dʒ/ (bn), /z/ (as)
            elif lang == "or" and not after_cons:
                it = _Item("C", "j", it.off)                              # ଯ = /dʒ/; ଯ-phala = /y/
        out.append(it)
    return out


def _tamil_voicing(items):
    """Tamil script has one letter per stop place; voicing is positional."""
    NATIVE = {0x15: ("k", "g"), 0x1A: ("s", "j"), 0x1F: ("tt", "dd"), 0x24: ("t", "d"),
              0x2A: ("p", "b")}
    seq = [it for it in items if it.kind in ("C", "V", "@") and not it.deleted]
    for idx, it in enumerate(seq):
        if it.kind != "C":
            continue
        prev = seq[idx - 1] if idx > 0 else None
        nxt = seq[idx + 1] if idx + 1 < len(seq) else None
        if it.off == 0x31:  # ற: geminate ற்ற is an alveolar stop
            if (nxt is not None and nxt.kind == "C" and nxt.off == 0x31) or \
               (prev is not None and prev.kind == "C" and prev.off == 0x31):
                it.phone = "tx"
            continue
        if it.off not in NATIVE:
            continue
        voiceless, voiced = NATIVE[it.off]
        gem = (prev is not None and prev.kind == "C" and prev.off == it.off) or \
              (nxt is not None and nxt.kind == "C" and nxt.off == it.off)
        if prev is None:
            it.phone = voiceless                      # word-initial (ச -> s)
        elif gem:
            it.phone = "c" if it.off == 0x1A else voiceless
        elif nxt is not None and nxt.kind == "C":
            it.phone = "c" if it.off == 0x1A else ("k" if it.off == 0x15 else voiceless)
        elif prev.kind == "C" and prev.phone in NASALS:
            it.phone = voiced                         # ங்க ஞ்ச ண்ட ந்த ம்ப
        elif prev.kind == "C" and prev.phone in VOICELESS_OBSTRUENTS:
            it.phone = voiceless
        else:
            it.phone = "s" if it.off == 0x1A else voiced  # between vowels / after r l y v zh


def _malayalam_rules(items):
    seq = [it for it in items if it.kind in ("C", "V", "@") and not it.deleted]
    for idx, it in enumerate(seq):
        if it.kind != "C":
            continue
        prev = seq[idx - 1] if idx > 0 else None
        nxt = seq[idx + 1] if idx + 1 < len(seq) else None
        if it.off == 0x28:  # ന: dental word-initially and before dentals, else alveolar
            if prev is None or (nxt is not None and nxt.kind == "C" and nxt.phone in ("t", "th", "d", "dh")):
                it.phone = "n"
            else:
                it.phone = "nx"
        elif it.off == 0x31:  # റ
            if prev is not None and prev.kind == "C" and prev.off in (0x28, 0x7B) and prev.phone in ("n", "nx"):
                it.phone = "d"                         # ന്റ -> /nd/ (alveolar)
                prev.phone = "nx"
            elif (nxt is not None and nxt.kind == "C" and nxt.off == 0x31) or \
                 (prev is not None and prev.kind == "C" and prev.off == 0x31):
                it.phone = "tx"                        # റ്റ -> alveolar stop
            else:
                it.phone = "rx"


def _resolve(items, lang):
    cfg = LANG[lang]
    inherent = cfg["inherent"]
    flat = []          # list of (phone, item_index) for consonants/vowels, to look ahead for NAS
    pending = []       # (position in flat, kind) nasal placeholders
    k = 0
    while k < len(items):
        it = items[k]
        if it.kind == "C":
            flat.append(it.phone)
        elif it.kind == "V":
            flat.extend(_vowel_phones(it.phone, cfg))
        elif it.kind == "@":
            if not it.deleted:
                flat.append(inherent)
        elif it.kind == "NAS":
            pending.append((len(flat), it.phone))
            flat.append("<NAS>")
        elif it.kind == "GEM":
            flat.append("<GEM>")
        elif it.kind == "VIS":
            flat.append("<VIS>")
        elif it.kind == "AYTHAM":
            flat.append("<AYT>")
        elif it.kind == "HALEND":
            flat.append("ix")
        k += 1
    out = []
    for pos, ph in enumerate(flat):
        nxt = next((p for p in flat[pos + 1:] if not p.startswith("<")), None)
        if ph == "<NAS>":
            kind = dict(pending)[pos]
            vowel_before = bool(out) and out[-1] in IN_VOWELS and out[-1] != "ix"
            mode = cfg["anusvara"]
            if kind == "candrabindu" and vowel_before:
                out[-1] = nasalize(out[-1])            # always a nasal vowel
            elif mode == "velar" and kind == "anusvara":
                out.append("ng")                       # Bengali/Assamese/Odia ং = /ŋ/
            elif nxt in IN_STOPS_BY_PLACE:
                out.append(IN_STOPS_BY_PLACE[nxt])     # homorganic nasal before a stop
            elif nxt in NASALS:
                out.append("n" if nxt == "nx" else nxt)
            elif mode == "dravidian":
                out.append("m")
            elif vowel_before:
                out[-1] = nasalize(out[-1])            # Indo-Aryan: हंस hãs, में mẽ
            elif kind == "anusvara":
                out.append("m")
        elif ph == "<GEM>":
            if nxt is not None and nxt not in IN_VOWELS:
                out.append(ASPIRATE_TO_PLAIN.get(nxt, nxt))
        elif ph == "<VIS>":
            out.append("h")
            if lang == "sa" and nxt is None and len(out) >= 2 and out[-2] in IN_VOWELS:
                echo = {"aa": "a", "ii": "i", "uu": "u", "ee": "e", "oo": "o", "ai": "i", "au": "u"}
                out.append(echo.get(out[-2], out[-2]))
        elif ph == "<AYT>":
            if nxt == "p":
                out.append("<F>")
            else:
                out.append("h")
        else:
            if out and out[-1] == "<F>" and ph == "p":
                out[-1] = "f"
                continue
            out.append(ph)
    return [p for p in out if p is not None and p != "<F>"]


def _post(phones, lang):
    cfg = LANG[lang]
    ov = OVERRIDES.get(lang, {})
    res = []
    for idx, p in enumerate(phones):
        nxt = phones[idx + 1] if idx + 1 < len(phones) else None
        if lang == "bn" and p in ("sh", "ssh", "s"):
            p = "s" if nxt in SIBILANT_CLUSTER_NEXT else "sh"   # স্ত st, but সময় ʃɔmɔy
        elif lang == "as" and p in ("sh", "ssh", "s"):
            p = "s" if nxt in SIBILANT_CLUSTER_NEXT else "x"    # অসম ɔxɔm, but স্থান st
        else:
            p = ov.get(p, p)
        res.append(p)
    ml = cfg["merge_len"]
    if ml:
        m = {"ii": "i", "uu": "u", "ii~": "i~", "uu~": "u~"}
        if ml == "all":
            m.update({"ee": "e", "oo": "o", "ee~": "e~", "oo~": "o~"})
        res = [m.get(p, p) for p in res]
    return res


def word_to_phones(word, lang):
    """Bare phone names for one word in a Brahmic script, or None if unsupported.

    `lang` selects the rules; it must be a language whose rules fit the word's script
    (the frontend picks the script's default language otherwise)."""
    word = unicodedata.normalize("NFC", word)
    script = script_of_word(word)
    if script not in BRAHMIC_BASE:
        return None
    items = _parse(word, script, lang)
    if not items:
        return None
    items = _cluster_rules(items, lang)
    _delete_schwas(items, LANG[lang]["schwa"], keep_after_cluster=lang in ("bn", "as"))
    if lang == "ta":
        _tamil_voicing(items)
    elif lang == "ml":
        _malayalam_rules(items)
    phones = _resolve(items, lang)
    phones = _post(phones, lang)
    return phones or None
