"""Semiotic pre-pass (pivo_multi/semiotic.py): telephones, times, dates, money, units, fractions,
math, e-mail / URLs, serial codes, roman numerals and abbreviations, read in all ten languages.

The same pass runs in front of the Kannada and Indian-English readers (src/pivo/normalize_kn.py) and in
the browser (web/app/semiotic.js); web/tests/numbers_parity.mjs checks that the browser copy agrees.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivo_multi import brahmic, semiotic  # noqa: E402
from pivo_multi.numnorm import LANGS, normalize_numbers  # noqa: E402

# One sentence per semiotic class (English carrier words stay English: the code-mix G2P reads them).
SENTENCES = [
    # telephone
    "Call 020 7946 0018 before 6:15.", "Call 080-2345-6789 now.", "Call (080) 2345 6789 now.",
    "Call +91 98765 43210 now.", "Call 98765-43210 now.", "Call 1800 123 4567 toll free.",
    "Call 1800-11-4000 now.", "Call +1 (555) 123-4567 now.", "Call +44 20 7946 0018 now.",
    "Dial 100 or 112.", "Call 044-2811 2233 ext. 245.",
    # time
    "Meet at 06:05.", "Meet at 18:45.", "Meet at 6:15PM.", "Meet at 6 pm.", "Meet at 6.15 pm.",
    "Meet at 10:30:45.", "Open 9am-5pm.", "Open 9:00 am - 5:30 pm.", "Open 6:15-7:30.", "Meet at 00:00.",
    "Meet at 12:00 noon.",
    # date
    "On 15/08/2024 we met.", "On 15-08-2024 we met.", "On 15.08.2024 we met.", "On 2024-08-15 we met.",
    "On 2024/08/15 we met.", "On 15/08/24 we met.", "On 15 Aug 2024 we met.", "On Aug 15, 2024 we met.",
    "On August 15th, 2024 we met.", "In August 2024 we met.", "On 1st Jan we met.", "In the 1990s we met.",
    "In FY 2023-24 we grew.", "From 2019 to 2024 we grew.",
    # cardinal, decimal, fraction, ordinal
    "We have 1,234 books.", "We have 12,34,567 books.", "We have 1234567 books.", "We have 10K followers.",
    "We have 1.5 lakh followers.", "It is -5 today.", "It is +5 today.", "The answer is 007.",
    "It is .5 of it.", "It is 0.05 of it.", "Add 1/2 cup.", "Add 2 1/2 cups.", "Add ½ cup.", "Add 1½ cups.",
    "He came 22nd.", "On the 21st floor.",
    # money
    "Pay ₹1,250.50.", "Pay Rs 500/-.", "Pay ₹ 2,000/-.", "Pay INR 300.", "Pay $5.99.", "Pay €20 now.",
    "Pay £3.50 now.", "Pay ₹5 crore.", "Pay USD 10.", "Pay $2.5 million.", "Pay ₹10K.",
    # measure, percent, range
    "Walk 5km.", "Drive 60 km/h.", "Drive 60 mph.", "It is 25°C.", "It is 98.6°F.", "Drink 500 ml water.",
    "It is 5'11\" tall.", "Area 100 m².", "Area 1200 sq ft.", "Store 1 TB.", "A 48 MP camera.",
    "A 60 Hz screen.", "Use 5 kWh.", "Wait 2.5 hrs.", "Wait 45 sec.", "Take 250 mg daily.",
    "Up 12.5 %.", "Up 5-10%.", "Pages 10–20.", "Temperature 10-15°C.", "Ratio is 2:1.",
    # electronic, serial codes, roman numerals, abbreviations
    "Mail info@example.com today.", "Visit www.example.com today.", "Visit https://example.com/help today.",
    "Follow @voxal today.", "Use #TTS today.", "COVID-19 spread.", "Use 4G or 5G.", "Print on A4 paper.",
    "Drink H2O.", "Flight AI-202 left.", "Car KA-01-AB-1234 parked.", "Version v2.0.1 shipped.",
    "Server 192.168.1.1 is down.", "Chapter IV begins.", "World War II ended.", "Dr. Rao met Mr. Shah.",
    "Apples, oranges etc. are fruits.", "Room No. 5 please.", "e.g. this one.",
    # math and symbols
    "So 2+2=4.", "So 5×3=15.", "So 10÷2=5.", "So 3^2=9.", "So x² is big.", "So 5 > 3 and 2 < 4.", "It is ±5.",
    "Use 50*2.", "Tom & Jerry.", "Price ~100.", "Copyright © 2024.", "Brand™ and Brand®.", "Yes/No.",
    "Path a\\b.", "Rating 4.5/5.", "Item #5.", "Item № 5.",
]

LEFTOVER = re.compile(r"[0-9₹$€£%°/\\@#&×÷=<>^±~©№²³½¼¾+*™®]")


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_every_class_is_read(lang):
    """No digit or symbol survives, and every native word has a pronunciation."""
    for text in SENTENCES:
        out = normalize_numbers(text, lang)
        assert not LEFTOVER.search(out), (lang, text, out)
        for word in re.findall(r"[^\sA-Za-z.,;:!?'\"()\-]+", out):
            assert brahmic.word_to_phones(word, lang), (lang, word, text, out)


@pytest.mark.parametrize("lang", sorted(LANGS) + ["kn"])
def test_every_semiotic_word_is_pronounceable(lang):
    W = semiotic.WORDS[lang]
    words = []
    for value in W.values():
        if isinstance(value, str):
            words.append(value)
        elif isinstance(value, list):
            words += [v for v in value if isinstance(v, str)]
        elif isinstance(value, dict):                                         # units
            for form in value.values():
                words += [form] if isinstance(form, str) else list(form)
    for phrase in words:
        phrase = re.sub(r"\{[a-z]+\}", " ", phrase)
        for word in re.findall(r"[^\sA-Za-z.,;:!?'\"()\-]+", phrase):
            assert brahmic.word_to_phones(word, lang), (lang, word, phrase)


GOLDEN = {
    "hi": [
        ("Call 020 7946 0018 before 6:15.", "Call शून्य दो शून्य, सात नौ चार छह, शून्य शून्य एक आठ before छह बजकर पंद्रह मिनट."),
        ("Call +1 (555) 123-4567 now.", "Call प्लस एक, पाँच पाँच पाँच, एक दो तीन, चार पाँच छह सात now."),
        ("Meet at 6:15 pm.", "Meet at शाम छह बजकर पंद्रह मिनट."),
        ("Meet at 10:30:45.", "Meet at दस बजकर तीस मिनट और पैंतालीस सेकंड."),
        ("Open 9am-5pm.", "Open सुबह नौ बजे से शाम पाँच बजे."),
        ("On Aug 15, 2024 we met.", "On पंद्रह अगस्त दो हज़ार चौबीस we met."),
        ("On 2024-08-15 we met.", "On पंद्रह अगस्त दो हज़ार चौबीस we met."),
        ("Pay $5.99.", "Pay पाँच डॉलर और निन्यानवे सेंट."),
        ("Area 100 m².", "Area एक सौ वर्ग मीटर."),
        ("Up 5-10%.", "Up पाँच से दस प्रतिशत."),
        ("It is .5 of it.", "It is शून्य दशमलव पाँच of it."),
        ("Add 2 1/2 cups.", "Add दो और आधा cups."),
        ("So 2+2=4.", "So दो प्लस दो बराबर चार."),
        ("Chapter IV begins.", "Chapter चार begins."),
        ("Mail info@example.com today.", "Mail info एट example डॉट com today."),
        ("Dr. Rao met Mr. Shah.", "डॉक्टर Rao met मिस्टर Shah."),
        ("We have 10K followers.", "We have दस हज़ार followers."),
    ],
    "ta": [
        ("Call 020 7946 0018 before 6:15.",
         "Call பூஜ்ஜியம் இரண்டு பூஜ்ஜியம், ஏழு ஒன்பது நான்கு ஆறு, பூஜ்ஜியம் பூஜ்ஜியம் ஒன்று எட்டு before ஆறு மணி பதினைந்து நிமிடம்."),
        ("Open 9am-5pm.", "Open காலை ஒன்பது மணி முதல் மாலை ஐந்து மணி."),
        ("Up 5-10%.", "Up ஐந்து முதல் பத்து சதவீதம்."),
        ("Rating 4.5/5.", "Rating ஐந்தில் நான்கு புள்ளி ஐந்து."),
    ],
    "bn": [
        ("Call 020 7946 0018 before 6:15.",
         "Call শূন্য দুই শূন্য, সাত নয় চার ছয়, শূন্য শূন্য এক আট before ছটা বেজে পনেরো মিনিট."),
        ("Area 100 m².", "Area একশো বর্গমিটার."),
    ],
}


@pytest.mark.parametrize("lang,text,expected", [(lang, t, e) for lang, rows in GOLDEN.items() for t, e in rows])
def test_golden(lang, text, expected):
    assert normalize_numbers(text, lang) == expected


@pytest.mark.parametrize("text,expected", [
    ("Prof.2", "ਪ੍ਰੋਫੈਸਰ ਦੋ"),                       # an abbreviation glued to a number keeps its dot
    ("੧੭–੨੦%", "ਸਤਾਰਾਂ ਤੋਂ ਵੀਹ ਪ੍ਰਤੀਸ਼ਤ"),             # range with a percent sign
    ("+1567", "ਪਲੱਸ ਇੱਕ ਹਜ਼ਾਰ ਪੰਜ ਸੌ ਸਤਾਹਠ"),          # not a telephone number
])
def test_edge_cases_pa(text, expected):
    assert normalize_numbers(text, "pa") == expected
