"""Unicode script ranges used by the frontend."""

# The nine Brahmic blocks share one ISCII-derived layout: the same offset inside each
# 128-code-point block is the same letter (0x15 = KA, 0x3E = AA sign, 0x4D = virama...).
BRAHMIC_BASE = {
    "devanagari": 0x0900, "bengali": 0x0980, "gurmukhi": 0x0A00, "gujarati": 0x0A80,
    "odia": 0x0B00, "tamil": 0x0B80, "telugu": 0x0C00, "kannada": 0x0C80,
    "malayalam": 0x0D00,
}
OTHER_RANGES = {
    "olchiki": [(0x1C50, 0x1C7F)],
    "meetei": [(0xABC0, 0xABFF), (0xAAE0, 0xAAFF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A)],
}
ZWNJ, ZWJ = "‌", "‍"


def script_of_char(ch):
    cp = ord(ch)
    for name, base in BRAHMIC_BASE.items():
        if base <= cp < base + 0x80:
            return name
    for name, ranges in OTHER_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    if 0x1CD0 <= cp <= 0x1CFF or 0xA8E0 <= cp <= 0xA8FF:
        return "devanagari"  # Vedic extensions / Devanagari extended
    return None


def script_of_word(word):
    """Script of a word, ignoring ZWJ/ZWNJ and the shared danda signs; None if mixed/unknown."""
    found = set()
    for ch in word:
        if ch in (ZWJ, ZWNJ):
            continue
        s = script_of_char(ch)
        if s is None:
            return None
        found.add(s)
    return found.pop() if len(found) == 1 else None


def is_word_char(ch):
    return ch in (ZWJ, ZWNJ) or script_of_char(ch) is not None
