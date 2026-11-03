"""Language table: one row per language the multilingual model can speak.

The ORDER of this table fixes lang_id (= row index) and voice_id (= 2 * lang_id + 0 F / 1 M).
It is append-only: languages are ordered by tier, so a model trained on tiers 1-2 uses
ids 0..11 and later tiers only add higher ids (grow the embeddings, nothing renumbers).

Tiers (`phase`), decided 3 Nov 2026 before the first data run:
  1  probe set: English, Kannada, Hindi, Telugu, Tamil, Bengali
  2  rest of the core: Marathi, Gujarati, Punjabi, Malayalam, Odia, Assamese
     -> tiers 1 + 2 = the 11 major Indian languages + English (default run)
  3  easy to read but niche (Devanagari, rules already work): Nepali, Konkani,
     Maithili, Dogri, Sanskrit. Optional: add with --phases 3 once tiers 1-2 sound good.
  4  dropped for now, too hard to get good output from rules + a tiny model:
       Urdu, Kashmiri, Sindhi  Perso-Arabic script leaves short vowels unwritten, so
                               rules cannot produce phones (needs a lexicon / neural G2P)
       Manipuri, Bodo          tonal languages whose tones are not written
       Santali                 checked consonants + vowel marks; G2P unverified on real text
     Their G2P code stays in place; nothing in tier 4 is prepared by default.

Fields
  code      ISO 639 code used everywhere (manifests, speaker names, lang ids)
  name      Rasa folder / config name (None if not in Rasa)
  script    primary script (key of scripts.SCRIPTS)
  phase     tier, see above
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
    # tier 1: probe set
    ("en",  None,         "latin",       1,  None,  None,     None,        None,  None),
    ("kn",  "Kannada",    "kannada",     1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("hi",  "Hindi",      "devanagari",  1,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("te",  "Telugu",     "telugu",      1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("ta",  "Tamil",      "tamil",       1,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("bn",  "Bengali",    "bengali",     1,  "aw",  "medial", "velar",     "all", ("r", "i")),
    # tier 2: rest of the core
    ("mr",  "Marathi",    "devanagari",  2,  "a",   "medial", "indo",      None,  ("r", "u")),
    ("gu",  "Gujarati",   "gujarati",    2,  "a",   "medial", "indo",      None,  ("r", "u")),
    ("pa",  "Punjabi",    "gurmukhi",    2,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("ml",  "Malayalam",  "malayalam",   2,  "a",   "none",   "dravidian", None,  ("r", "u")),
    ("or",  "Odia",       "odia",        2,  "aw",  "none",   "velar",     "all", ("r", "u")),
    ("as",  "Assamese",   "bengali",     2,  "aw",  "final",  "velar",     "all", ("r", "i")),
    # tier 3: easy but niche (optional)
    ("ne",  "Nepali",     "devanagari",  3,  "a",   "final",  "indo",      "iu",  ("r", "i")),
    ("kok", "Konkani",    "devanagari",  3,  "a",   "final",  "indo",      None,  ("r", "u")),
    ("mai", "Maithili",   "devanagari",  3,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("doi", "Dogri",      "devanagari",  3,  "a",   "medial", "indo",      None,  ("r", "i")),
    ("sa",  "Sanskrit",   "devanagari",  3,  "a",   "none",   "dravidian", None,  ("r", "i")),
    # tier 4: dropped for now (see above)
    ("brx", "Bodo",       "devanagari",  4,  "a",   "final",  "indo",      None,  ("r", "i")),
    ("mni", "Manipuri",   "meetei",      4,  "a",   "none",   "indo",      "all", ("r", "i")),
    ("sat", "Santali",    "olchiki",     4,  None,  None,     None,        None,  None),
    ("ur",  "Urdu",       "arabic",      4,  None,  None,     None,        None,  None),
    ("ks",  "Kashmiri",   "arabic",      4,  None,  None,     None,        None,  None),
    ("sd",  "Sindhi",     "arabic",      4,  None,  None,     None,        None,  None),
]
DEFAULT_PHASES = (1, 2)

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
