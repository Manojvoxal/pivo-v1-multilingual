"""G2P and frontend checks for every supported script.

Run:  python -m pytest tests/test_g2p.py -q      (from the pivo_multi_data folder)
Set PIVO_CMUDICT=/path/to/cmudict.dict for the code-mix / English tests
(PiVo's file: pivo-v1/src/pivo/resources/cmudict.dict).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivo_multi import english, frontend  # noqa: E402
from pivo_multi.brahmic import word_to_phones  # noqa: E402
from pivo_multi.alphabetic import meetei_to_phones, olchiki_to_phones  # noqa: E402
from pivo_multi.symbols import SYMBOL_TO_ID  # noqa: E402

CASES = [
    # lang, word, expected phones (space-separated)
    # Hindi: final + medial schwa deletion, nasalisation, nukta
    ("hi", "कमल", "k a m a l"),
    ("hi", "कमला", "k a m l aa"),
    ("hi", "समझना", "s a m a jh n aa"),
    ("hi", "बचपन", "b a c p a n"),
    ("hi", "सरकार", "s a r k aa r"),
    ("hi", "भारत", "bh aa r a t"),
    ("hi", "मित्र", "m i t r a"),
    ("hi", "धर्म", "dh a r m"),
    ("hi", "हिंदी", "h i n d ii"),
    ("hi", "में", "m ee~"),
    ("hi", "हंस", "h a~ s"),
    ("hi", "ज़मीन", "z a m ii n"),
    ("hi", "पढ़ना", "p a rrh n aa"),
    ("hi", "क्षमा", "k sh a m aa"),
    ("hi", "ज्ञान", "g y aa n"),
    ("hi", "डॉक्टर", "dd aw k tt a r"),
    ("hi", "प्रिय", "p r i y a"),
    ("hi", "न", "n a"),
    # Marathi keeps ष, ळ; ज्ञ = dny
    ("mr", "महाराष्ट्र", "m a h aa r aa ssh tt r a"),
    ("mr", "ज्ञान", "d ny aa n"),
    ("mr", "मुळे", "m u ll ee"),
    # Sanskrit: no deletion, visarga echo
    ("sa", "रामः", "r aa m a h a"),
    ("sa", "संस्कृतम्", "s a m s k r i t a m"),
    # Nepali: sibilants -> s, no length
    ("ne", "नेपाल", "n ee p aa l"),
    ("ne", "शिक्षा", "s i k s aa"),
    # Bengali: inherent ɔ, ং = ŋ, ya-phala gemination, sibilants
    ("bn", "কলকাতা", "k aw l k aa t aa"),
    ("bn", "আমার", "aa m aa r"),
    ("bn", "বাংলা", "b aa ng l aa"),
    ("bn", "বাক্য", "b aa k k aw"),
    ("bn", "সময়", "sh aw m aw y"),
    ("bn", "স্টেশন", "s tt e sh aw n"),
    ("bn", "যদি", "j aw d i"),
    ("bn", "ধর্ম", "dh aw r m aw"),
    ("bn", "সূর্য", "sh u r j aw"),
    # Assamese
    ("as", "অসম", "aw x aw m"),
    ("as", "চাহ", "s aa h"),
    ("as", "জল", "z aw l"),
    # Odia: inherent ɔ kept, sibilants -> s
    ("or", "ଘର", "gh aw r aw"),
    ("or", "ଓଡ଼ିଆ", "o rr i aa"),
    ("or", "ସତ୍ୟ", "s aw t y aw"),
    # Gujarati / Punjabi
    ("gu", "ગુજરાતી", "g u j r aa t ii"),
    ("gu", "ઑફિસ", "aw ph i s"),
    ("pa", "ਪੰਜਾਬ", "p a ny j aa b"),
    ("pa", "ਸੱਚ", "s a c c"),
    ("pa", "ਖ਼ਬਰ", "x a b a r"),
    # Dravidian
    ("kn", "ಕನ್ನಡ", "k a n n a dd a"),
    ("kn", "ಸಂಸಾರ", "s a m s aa r a"),
    ("kn", "ಹಂಪಿ", "h a m p i"),
    ("te", "తెలుగు", "t e l u g u"),
    ("te", "సంగీతం", "s a ng g ii t a m"),
    ("ta", "தமிழ்", "t a m i zh"),
    ("ta", "மகன்", "m a g a nx"),
    ("ta", "அப்பா", "a p p aa"),
    ("ta", "சென்னை", "s e nx nx ai"),
    ("ta", "பச்சை", "p a c c ai"),
    ("ta", "பஞ்சு", "p a ny j u"),
    ("ta", "வெற்றி", "v e tx tx i"),
    ("ta", "பாடம்", "p aa dd a m"),
    ("ta", "சக்தி", "s a k t i"),
    ("ml", "മലയാളം", "m a l a y aa ll a m"),
    ("ml", "എന്റെ", "e nx d e"),
    ("ml", "അവന്", "a v a nx ix"),
    ("ml", "അവൻ", "a v a nx"),
    ("ml", "നന്ദി", "n a n d i"),
    ("ml", "കുറ്റം", "k u tx tx a m"),
    ("ml", "പഴം", "p a zh a m"),
]


@pytest.mark.parametrize("lang,word,expected", CASES)
def test_brahmic(lang, word, expected):
    assert " ".join(word_to_phones(word, lang)) == expected


def test_olchiki():
    assert " ".join(olchiki_to_phones("ᱥᱟᱱᱛᱟᱲᱤ")) == "s a n t a rr i"
    assert " ".join(olchiki_to_phones("ᱚᱞ")) == "aw l"
    assert " ".join(olchiki_to_phones("ᱫᱟᱜ")) == "d a k"          # final ᱜ checked
    assert " ".join(olchiki_to_phones("ᱠᱷᱟᱹ")) == "kh a"


def test_meetei():
    assert " ".join(meetei_to_phones("ꯃꯤꯇꯩ")) == "m i t ai"
    assert " ".join(meetei_to_phones("ꯃꯅꯤꯄꯨꯔ")) == "m a n i p u r a"
    assert " ".join(meetei_to_phones("ꯂꯩꯕꯥꯛ")) == "l ai b aa k"


def test_old_style_chillu_and_zwj():
    ph = frontend.normalize("അവന\u0d4d\u200d വന്നു")
    assert "\u200d" not in ph and ph.startswith("അവൻ")
    assert " ".join(word_to_phones("क्\u200dष", "hi")) == "k sh a"


def test_rejects():
    assert word_to_phones("१२३", "hi") is None
    assert word_to_phones("abc", "hi") is None
    with pytest.raises(frontend.FrontendError):
        frontend.phonemize("2026 ರಲ್ಲಿ", "kn")


def test_every_phone_is_a_symbol():
    for lang, word, _ in CASES:
        for p in word_to_phones(word, lang):
            assert "in_" + p in SYMBOL_TO_ID, (lang, word, p)


CMU = os.environ.get("PIVO_CMUDICT")


@pytest.mark.skipif(not CMU, reason="set PIVO_CMUDICT to run code-mix tests")
def test_frontend_codemix_and_layout():
    english.set_cmudict_path(CMU)
    ph = frontend.phonemize("ನಾನು office ಗೆ ಹೋಗುತ್ತೇನೆ.", "kn")
    assert ph.symbols[0] == "<bos>" and ph.symbols[-1] == "<eos>" and ph.symbols[-2] == "."
    assert "AO1" in ph.symbols and "in_k" not in ph.symbols[:3]
    ph = frontend.phonemize("मैं ठीक हूँ, धन्यवाद।", "hi")
    assert [w for w, _ in ph.words] == ["मैं", "ठीक", "हूँ", "धन्यवाद"]
    assert "," in ph.symbols and ph.symbols[-2] == "."
    ph = frontend.phonemize("Good morning, how are you?", "en")
    assert ph.symbols[-2] == "?"
    # a Devanagari word inside a Bengali sentence falls back to Hindi rules
    ph = frontend.phonemize("আমি नमस्ते বলি", "bn")
    assert len(ph.words) == 3
