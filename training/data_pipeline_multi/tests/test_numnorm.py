"""Number normaliser (pivo_multi/numnorm.py): golden readings and invariants.

The golden readings were checked by hand against CLDR / AI4Bharat / standard usage; a change
here is a change in what the web app says, so update NUMBER_REVIEW.xlsx with it.
"""

import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivo_multi import brahmic, frontend  # noqa: E402
from pivo_multi.numnorm import DATA, LANGS, cardinal, normalize_numbers, ordinal, year_words  # noqa: E402

GOLDEN = {
    "hi": [
        ('सन् 1947 में', 'सन् उन्नीस सौ सैंतालीस में'),
        ('₹1,25,000.50', 'एक लाख पच्चीस हज़ार रुपये और पचास पैसे'),
        ('Rs 5 lakh', 'पाँच लाख रुपये'),
        ('21वीं सदी', 'इक्कीसवीं सदी'),
        ('7:05 pm', 'रात सात बजकर पाँच मिनट'),
        ('15/08/1947', 'पंद्रह अगस्त उन्नीस सौ सैंतालीस'),
        ('-5°C', 'माइनस पाँच डिग्री सेल्सियस'),
        ('BJP', 'बी जे पी'),
        ('10-15 km', 'दस से पंद्रह किलोमीटर'),
        ('1998 ई. में', 'उन्नीस सौ अट्ठानवे ईस्वी में'),
        ('10:30 बजे तक', 'दस बजकर तीस मिनट तक'),
    ],
    "mr": [
        ('१९९८ साली', 'एकोणीसशे अठ्ठ्याण्णव साली'),
        ('५वा', 'पाचवा'),
        ('२०२६मध्ये', 'दोन हजार सव्वीसमध्ये'),
        ('रु. ५००', 'पाचशे रुपये'),
        ('१२.५%', 'बारा पूर्णांक पाच टक्के'),
        ('१००', 'शंभर'),
        ('११००', 'एक हजार शंभर'),
    ],
    "gu": [
        ('૨૦૨૬માં', 'બે હજાર છવ્વીસમાં'),
        ('૪થી તારીખ', 'ચોથી તારીખ'),
        ('૫થી ૧૦', 'પાંચથી દસ'),
        ('૨૦૦', 'બસો'),
        ('૧૫ ઓગસ્ટ ૧૯૪૭ના રોજ', 'પંદર ઓગસ્ટ ઓગણીસ સો સુડતાલીસના રોજ'),
        ('૧૦:૩૦ વાગ્યે', 'દસ વાગીને ત્રીસ મિનિટ'),
    ],
    "pa": [
        ('੫ਵਾਂ', 'ਪੰਜਵਾਂ'),
        ('ਸੰਨ ੧੯੪੭', 'ਸੰਨ ਉੱਨੀ ਸੌ ਸੰਤਾਲੀ'),
        ('₹੧,੨੫੦', 'ਇੱਕ ਹਜ਼ਾਰ ਦੋ ਸੌ ਪੰਜਾਹ ਰੁਪਏ'),
        ('੧੦:੩੦ ਵਜੇ', 'ਦਸ ਵੱਜ ਕੇ ਤੀਹ ਮਿੰਟ'),
        ('੯੯', 'ਨੜ੍ਹਿੰਨਵੇ'),
    ],
    "bn": [
        ('১লা বৈশাখ', 'পয়লা বৈশাখ'),
        ('২৫শে ডিসেম্বর', 'পঁচিশে ডিসেম্বর'),
        ('১৫ই আগস্ট', 'পনেরোই আগস্ট'),
        ('২০২৬-এ', 'দুই হাজার ছাব্বিশে'),
        ('১৯৭১ সালে', 'উনিশশো একাত্তর সালে'),
        ('২য়', 'দ্বিতীয়'),
        ('১০০-র বেশি', 'একশোর বেশি'),
        ('১০:৩০-এ', 'দশটা বেজে ত্রিশ মিনিটে'),
        ('৬০০', 'ছশো'),
    ],
    "or": [
        ('୨୦୨୬ରେ', 'ଦୁଇ ହଜାର ଛବିଶରେ'),
        ('୧୯୪୭ ମସିହା', 'ଊଣେଇଶ ଶହ ସତଚାଳିଶ ମସିହା'),
        ('୫ମ', 'ପଞ୍ଚମ'),
        ('୧୦:୩୦ରେ', 'ଦଶଟା ତିରିଶ ମିନିଟରେ'),
        ('୧୦-୨୦', 'ଦଶରୁ କୋଡ଼ିଏ'),
        ('୯୯', 'ଅନେଶତ'),
    ],
    "as": [
        ('২০২৬ত', 'দুই হাজাৰ ছাব্বিশত'),
        ('১৯৪৭ চনত', 'ঊনৈশ শ সাতচল্লিশ চনত'),
        ('১০-২০', 'দহৰ পৰা বিশ'),
        ('১০:৩০ বজাত', 'দহ বাজি ত্ৰিশ মিনিট'),
        ('৪৫', 'পঞ্চল্লিশ'),
    ],
    "ta": [
        ('2026இல்', 'இரண்டாயிரத்து இருபத்து ஆறில்'),
        ('2026-ம் ஆண்டு', 'இரண்டாயிரத்து இருபத்து ஆறாம் ஆண்டு'),
        ('5வது', 'ஐந்தாவது'),
        ('1000இல்', 'ஆயிரத்தில்'),
        ('1 நாள்', 'ஒரு நாள்'),
        ('5000', 'ஐயாயிரம்'),
        ('21000', 'இருபத்து ஓராயிரம்'),
        ('1,25,000', 'ஒரு லட்சத்து இருபத்து ஐந்தாயிரம்'),
        ('10:30 மணிக்கு', 'பத்து மணி முப்பது நிமிடத்துக்கு'),
        ('800', 'எண்ணூறு'),
    ],
    "te": [
        ('2026లో', 'రెండు వేల ఇరవై ఆరులో'),
        ('5వ తేదీ', 'ఐదవ తేదీ'),
        ('2000లో', 'రెండు వేలలో'),
        ('500 మంది', 'ఐదు వందల మంది'),
        ('500.', 'ఐదు వందలు.'),
        ('1:30', 'ఒంటి గంట ముప్పై నిమిషాలు'),
        ('1998 సంవత్సరంలో', 'పంతొమ్మిది వందల తొంభై ఎనిమిది సంవత్సరంలో'),
        ('₹2.5 crore', 'రెండు కోట్ల యాభై లక్షల రూపాయలు'),
        ('1 km', 'ఒక కిలోమీటరు'),
        ('21000', 'ఇరవై ఒక వేలు'),
        ('1947 ఆగస్టు 15న', 'పంతొమ్మిది వందల నలభై ఏడు ఆగస్టు పదిహేనున'),
        ('10:30 గంటలకు', 'పది గంటల ముప్పై నిమిషాలకు'),
    ],
    "ml": [
        ('2026-ൽ', 'രണ്ടായിരത്തി ഇരുപത്തിയാറിൽ'),
        ('5-ാം', 'അഞ്ചാം'),
        ('1000-ൽ', 'ആയിരത്തിൽ'),
        ('1 ദിവസം', 'ഒരു ദിവസം'),
        ('10-ന്', 'പത്തിന്'),
        ('20000', 'ഇരുപതിനായിരം'),
        ('9000', 'ഒമ്പതിനായിരം'),
        ('5000', 'അയ്യായിരം'),
        ('1,25,000', 'ഒരു ലക്ഷത്തി ഇരുപത്തിയഞ്ചായിരം'),
        ('29', 'ഇരുപത്തൊമ്പത്'),
    ],
}


@pytest.mark.parametrize("lang,text,expected", [(lang, t, e) for lang, rows in GOLDEN.items() for t, e in rows])
def test_golden(lang, text, expected):
    assert normalize_numbers(text, lang) == expected


def test_all_ten_languages():
    assert set(LANGS) == {"hi", "mr", "gu", "pa", "bn", "or", "as", "ta", "te", "ml"}


@pytest.mark.parametrize("lang", sorted(DATA))
def test_every_word_has_a_pronunciation(lang):
    rng = random.Random(1)
    texts = [cardinal(n, lang, attr) for n in list(range(0, 3000)) + [rng.randint(0, 10 ** 15) for _ in range(300)]
             for attr in (False, True)]
    texts += [year_words(y, lang) for y in range(1100, 2100)]
    texts += [ordinal(n, lang, e) for n in range(1, 120) for e in DATA[lang]["ordinal"]["endings"]]
    texts += [normalize_numbers(f"{h}:{m:02d}", lang) for h in range(24) for m in (0, 5, 30)]
    D = DATA[lang]
    texts += D["months"] + D["letters"] + [D["point"], D["minus"], D["percent"]]
    texts += [w if isinstance(w, str) else " ".join(w) for w in D["measures"].values()]
    for text in texts:
        for word in text.split():
            assert brahmic.word_to_phones(word, lang), (lang, word, text)


@pytest.mark.parametrize("lang", sorted(DATA))
def test_no_digits_left(lang):
    rng = random.Random(2)
    pieces = ["₹", "Rs ", "$", "%", " km", "-", ":", "/", ".", ",", " ", "st", "th", "BJP", "lakh "]
    for _ in range(400):
        text = "".join(rng.choice(pieces) + str(rng.randint(0, 10 ** rng.randint(1, 13))) for _ in range(rng.randint(1, 4)))
        out = normalize_numbers(text, lang)
        assert not re.search(r"[0-9]", out), (text, out)


def test_frontend_reads_numbers_only_when_asked():
    with pytest.raises(frontend.FrontendError):
        frontend.phonemize("कीमत ₹1,250 है।", "hi")                     # training mode drops digits
    p = frontend.phonemize("कीमत ₹1,250 है।", "hi", strict=False, numbers=True)
    assert p.normalized_text == "कीमत एक हज़ार दो सौ पचास रुपये है."
