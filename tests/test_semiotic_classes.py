"""Semiotic classes (telephone, time, date, money, measure, math, electronic, ...) in the Kannada and
Indian-English readers, and the copies of the shared pre-pass that must stay identical."""

import json
from pathlib import Path

import pytest

from pivo import kannada, text
from pivo.normalize_kn import normalize_indian_english, normalize_kannada_text

ROOT = Path(__file__).resolve().parents[1]


def english(sentence):
    return text.normalize(normalize_indian_english(sentence))


@pytest.mark.parametrize("sentence, spoken", [
    ("Call 020 7946 0018 before 6:15.",
     "Call zero two zero, seven nine four six, zero zero one eight before six fifteen."),
    ("Call +1 (555) 123-4567 now.", "Call plus one, five five five, one two three, four five six seven now."),
    ("Call 1800 123 4567 toll free.", "Call one eight zero zero, one two three, four five six seven toll free."),
    ("Meet at 6:15PM.", "Meet at six fifteen P M."),
    ("Meet at 10:30:45.", "Meet at ten thirty and forty five seconds."),
    ("Meet at 00:00.", "Meet at midnight."),
    ("Open 9am-5pm.", "Open nine A M to five P M."),
    ("On 2024-08-15 we met.", "On fifteenth August, twenty twenty four we met."),
    ("On Aug 15, 2024 we met.", "On fifteenth August, twenty twenty four we met."),
    ("In the 1990s we met.", "In the nineteen nineties we met."),
    ("We have 12,34,567 books.", "We have twelve lakh thirty four thousand five hundred sixty seven books."),
    ("We have 10K followers.", "We have ten thousand followers."),
    ("It is .5 of it.", "It is zero point five of it."),
    ("Add 2 1/2 cups.", "Add two and a half cups."),
    ("Pay ₹1,250.50.", "Pay one thousand two hundred fifty rupees and fifty paise."),
    ("Pay $5.99.", "Pay five dollars and ninety nine cents."),
    ("Pay Rs 500/-.", "Pay five hundred rupees."),
    ("It is 25°C.", "It is twenty five degrees celsius."),
    ("Area 100 m².", "Area one hundred square meters."),
    ("It is 5'11\" tall.", "It is five feet eleven inches tall."),
    ("Up 5-10%.", "Up five to ten percent."),
    ("Mail info@example.com today.", "Mail info at example dot com today."),
    ("Chapter IV begins.", "Chapter four begins."),
    ("So 2+2=4.", "So two plus two equals four."),
    ("So x² is big.", "So x squared is big."),
    ("Rating 4.5/5.", "Rating four point five out of five."),
    ("Open 24/7.", "Open twenty four by seven."),
    ("Copyright © 2024.", "copyright twenty twenty four."),
    ("Dr. Rao met Mr. Shah.", "Doctor Rao met Mister Shah."),
])
def test_indian_english_classes(sentence, spoken):
    assert english(sentence) == spoken


@pytest.mark.parametrize("sentence, spoken", [
    ("Call 020 7946 0018 before 6:15.",
     "Call ಸೊನ್ನೆ ಎರಡು ಸೊನ್ನೆ, ಏಳು ಒಂಬತ್ತು ನಾಲ್ಕು ಆರು, ಸೊನ್ನೆ ಸೊನ್ನೆ ಒಂದು ಎಂಟು before ಆರು ಗಂಟೆ ಹದಿನೈದು ನಿಮಿಷ."),
    ("Meet at 6:15 pm.", "Meet at ಸಂಜೆ ಆರು ಗಂಟೆ ಹದಿನೈದು ನಿಮಿಷ."),
    ("Open 9am-5pm.", "Open ಬೆಳಿಗ್ಗೆ ಒಂಬತ್ತು ಗಂಟೆಯಿಂದ ಸಂಜೆ ಐದು ಗಂಟೆ."),
    ("On Aug 15, 2024 we met.", "On ಹದಿನೈದು ಆಗಸ್ಟ್ ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತನಾಲ್ಕು we met."),
    ("Pay $5.99.", "Pay ಐದು ಡಾಲರ್ ತೊಂಬತ್ತೊಂಬತ್ತು ಸೆಂಟ್."),
    ("It is -5 today.", "It is ಮೈನಸ್ ಐದು today."),
    ("Up 5-10%.", "Up ಶೇಕಡಾ ಐದರಿಂದ ಹತ್ತು."),
    ("Area 100 m².", "Area ನೂರು ಚದರ ಮೀಟರ್."),
    ("Rating 4.5/5.", "Rating ಐದರಲ್ಲಿ ನಾಲ್ಕು ಪಾಯಿಂಟ್ ಐದು."),
    ("Dr. Rao met Mr. Shah.", "ಡಾಕ್ಟರ್ Rao met ಮಿಸ್ಟರ್ Shah."),
])
def test_kannada_classes(sentence, spoken):
    assert normalize_kannada_text(sentence) == spoken


def test_every_class_reaches_valid_tokens():
    import ast
    source = ROOT / "training" / "data_pipeline_multi" / "tests" / "test_semiotic.py"
    if not source.exists():
        pytest.skip("training/data_pipeline_multi not present")
    tree = ast.parse(source.read_text(encoding="utf-8"))                     # the ten-language sentence list
    SENTENCES = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SENTENCES")
    for sentence in SENTENCES:
        front = kannada.phonemize(sentence)
        assert front.symbols[0] == "<bos>" and front.symbols[-1] == "<eos>", sentence
        english(sentence)                                                     # strict: raises on leftovers


def test_semiotic_copies_are_identical():
    """src/pivo keeps its own copy so the package installs without the training folder."""
    pipeline = ROOT / "training" / "data_pipeline_multi" / "pivo_multi"
    if not pipeline.exists():
        pytest.skip("training/data_pipeline_multi not present")
    package = ROOT / "src" / "pivo"
    read = lambda p: p.read_text(encoding="utf-8").replace("\r\n", "\n")  # noqa: E731
    assert read(package / "semiotic.py") == read(pipeline / "semiotic.py")
    assert json.loads(read(package / "resources" / "semiotic.json")) == json.loads(read(pipeline / "semiotic.json"))
