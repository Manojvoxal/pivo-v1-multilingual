"""What goes into the multilingual dataset: voices, sources, hours, style filter.

Voice ids are FIXED by the order of `all_speakers()` (= the order of langs.LANGS, which is
append-only and sorted by tier), so they never change when a tier is added later.
Default run (tiers 1 + 2): 12 languages x 2 voices = 24 voices, ids 0..23.
All tiers listed: 23 languages, 46 voice ids.
"""

from .langs import DEFAULT_PHASES, LANG, LANGS

# ------------------------------------------------------------------ hours
RASA_HOURS_PER_VOICE = 12.0       # target after filtering (Rasa has ~22-30 h per voice)
SYSPIN_KN_HOURS = 14.5            # same as PiVo-KN
ENGLISH_HOURS_PER_VOICE = 13.0    # IndicTTS English has ~11-13 h per voice: take all
OVERCOLLECT = 1.25                # prepare a bit more; alignment/build drop some clips

# ------------------------------------------------------------------ Rasa
RASA_REPO = "ai4bharat/Rasa"
RASA_SPLIT = "train"              # Rasa's own test split is kept aside for evaluation
# Rasa `style` labels seen in the 3 Nov 2026 smoke scan (Hindi, Tamil):
#   read speech : WIKI, BOOK, NEWS, INDIC (IndicTTS sentences)          -> kept
#   emotions    : ANGER, DISGUST, FEAR, HAPPY, SAD, SURPRISE             -> dropped
#   other       : CONV (conversation), PROPER NOUN (lists of names, ~2 s clips),
#                 ALEXA, BB (BigBasket), DIGI (DigiPay), UMANG (commands)  -> dropped
# Plain read speech gives a tiny model one consistent speaking style per voice.
# Override with --styles "A,B,C"; the scan report lists every label with its hours.
DEFAULT_STYLES = ("WIKI", "BOOK", "NEWS", "INDIC")


def style_allowed(style, explicit=None):
    s = (style or "").strip().lower()
    allowed = explicit if explicit is not None else {x.lower() for x in DEFAULT_STYLES}
    return bool(s) and s in allowed


# ------------------------------------------------------------------ English
# IndicTTS (IIT Madras) English, recorded by the same voice talents as the Indian
# languages. Upload the unzipped folders to the raw volume (see README):
#   modal volume put tts1-raw-datasets <local folder> /english/indictts/tamil_female
INDICTTS_EN = [
    {"gender": "f", "dir": "/raw/english/indictts/tamil_female", "tag": "tamil"},
    {"gender": "m", "dir": "/raw/english/indictts/tamil_male", "tag": "tamil"},
]
HIFI_EN = [  # fallback / smoke test: American English (already used for PiVo-KN)
    {"gender": "f", "key": "92"},
    {"gender": "m", "key": "6097"},
]

# ------------------------------------------------------------------ SYSPIN Kannada
SYSPIN_DIR = "/raw/kannada/syspin_hf"


def all_speakers(english="indictts"):
    """Every voice of the full model, in voice-id order."""
    out = []
    for row in LANGS:
        code = row[0]
        for g in ("f", "m"):
            if code == "en":
                if english == "hifi":
                    src = next(s for s in HIFI_EN if s["gender"] == g)
                    spk = {"name": f"en_hifi{src['key']}_{g}", "source": "hifi", "key": src["key"],
                           "hours": ENGLISH_HOURS_PER_VOICE}
                else:
                    src = next(s for s in INDICTTS_EN if s["gender"] == g)
                    spk = {"name": f"en_iitm_{src['tag']}_{g}", "source": "indictts", "dir": src["dir"],
                           "hours": ENGLISH_HOURS_PER_VOICE}
            elif code == "kn":
                spk = {"name": f"kn_syspin_{g}", "source": "syspin",
                       "key": "Female" if g == "f" else "Male", "hours": SYSPIN_KN_HOURS}
            else:
                spk = {"name": f"{code}_rasa_{g}", "source": "rasa", "config": LANG[code]["name"],
                       "hours": RASA_HOURS_PER_VOICE}
            spk.update(lang=code, gender=g, phase=LANG[code]["phase"])
            out.append(spk)
    for i, spk in enumerate(out):
        spk["voice_id"] = i
    return out


def active_speakers(phases=DEFAULT_PHASES, english="indictts", langs=None):
    """Speakers prepared in this run (voice ids stay those of all_speakers)."""
    return [s for s in all_speakers(english)
            if s["phase"] in phases and (langs is None or s["lang"] in langs)]
