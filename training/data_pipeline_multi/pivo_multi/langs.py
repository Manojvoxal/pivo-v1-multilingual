"""Language table: one row per language the multilingual model speaks.

Fields
  code      ISO 639 code used everywhere (manifests, speaker names, lang ids)
  name      Rasa folder / config name (None if not in Rasa)
  script    primary script (key of scripts.SCRIPTS)
  phase     1 = probe set, 2 = rest of the Brahmic/alphabetic languages,
            3 = Perso-Arabic (G2P not written yet)
  inherent  inherent vowel of the script: "a" (ə/a) or "aw" (ɔ: Bengali, Assamese, Odia)
  schwa     inherent-vowel deletion: "none" | "final" | "medial" (final + VC_CV rule)
  anusvara  "indo": homorganic nasal before a stop, else nasalise the vowel
            "dravidian": homorganic nasal before a stop, else /m/
            "velar": anusvara is always /ŋ/ (Bengali, Assamese, Odia)
  merge_len no phonemic vowel length -> ii->i, uu->u (and ee->e, oo->o if "all")
  vocalic_r reading of ऋ-type letters
"""

LANGS = [
    # code   name          script        phase inherent schwa     anusvara     merge_len vocalic_r
    ("en",  None,         "latin",       1,  None,  None,     None,        None,  None),
    ("kn",  "Kannada",    "kannada",     1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("hi",  "Hindi",      "devanagari",  1,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("te",  "Telugu",     "telugu",      1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("ta",  "Tamil",      "tamil",       1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("bn",  "Bengali",    "bengali",     1,  "aw",  "medial", "velar",      "all", ("r", "i")),
    ("mr",  "Marathi",    "devanagari",  2,  "a",   "medial", "indo",      None,  ("r", "u")),
    ("ne",  "Nepali",     "devanagari",  2,  "a",   "final",  "indo",      "iu",  ("r", "i")),
    ("kok", "Konkani",    "devanagari",  2,  "a",   "final",  "indo",      None,  ("r", "u")),
    ("mai", "Maithili",   "devanagari",  2,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("doi", "Dogri",      "devanagari",  2,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("brx", "Bodo",       "devanagari",  2,  "a",   "final",  "indo",      None,  ("r", "i")),
    ("sa",  "Sanskrit",   "devanagari",  2,  "a",   "none",   "dravidian", None,  ("r", "i")),
    ("as",  "Assamese",   "bengali",     2,  "aw",  "final",  "velar",      "all", ("r", "i")),
    ("gu",  "Gujarati",   "gujarati",    2,  "a",   "medial", "indo",      None,  ("r", "u")),
    ("pa",  "Punjabi",    "gurmukhi",    2,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("or",  "Odia",       "odia",        2,  "aw",  "none",   "velar",      "all", ("r", "u")),
    ("ml",  "Malayalam",  "malayalam",   2,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("mni", "Manipuri",   "meetei",      2,  "a",   "none",   "indo",      "all", ("r", "i")),
    ("sat", "Santali",    "olchiki",     2,  None,  None,     None,        None,  None),
    ("ur",  "Urdu",       "arabic",      3,  None,  None,     None,        None,  None),
    ("ks",  "Kashmiri",   "arabic",      3,  None,  None,     None,        None,  None),
    ("sd",  "Sindhi",     "arabic",      3,  None,  None,     None,        None,  None),
]

FIELDS = ("code", "name", "script", "phase", "inherent", "schwa", "anusvara", "merge_len", "vocalic_r")
LANG = {row[0]: dict(zip(FIELDS, row)) for row in LANGS}
LANG_CODES = tuple(row[0] for row in LANGS)
LANG_TO_ID = {code: i for i, code in enumerate(LANG_CODES)}

# Which language's rules to use for a word written in a script that is not the
# sentence language's own script (e.g. a Devanagari word inside a Marathi sentence
# uses Marathi rules; inside a Bengali sentence it falls back to Hindi rules).
SCRIPT_DEFAULT_LANG = {
    "devanagari": "hi", "bengali": "bn", "gurmukhi": "pa", "gujarati": "gu",
    "odia": "or", "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml",
    "olchiki": "sat", "meetei": "mni",
}

# Scripts each language may legitimately be written in (Rasa text may use either
# for Manipuri / Santali / Sindhi / Kashmiri; the scan report shows which).
LANG_SCRIPTS = {
    "mni": ("meetei", "bengali"),
    "sat": ("olchiki", "devanagari"),
    "ks": ("arabic", "devanagari"),
    "sd": ("arabic", "devanagari"),
}


def lang_scripts(code):
    return LANG_SCRIPTS.get(code, (LANG[code]["script"],))
