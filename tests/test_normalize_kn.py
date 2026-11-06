import pytest

from pivo import kannada
from pivo.normalize_kn import (attach_suffix, indian_english_cardinal, kannada_cardinal,
                               normalize_indian_english, normalize_kannada_text)


@pytest.mark.parametrize('number, words', [
    (0, 'ಸೊನ್ನೆ'), (11, 'ಹನ್ನೊಂದು'), (19, 'ಹತ್ತೊಂಬತ್ತು'), (21, 'ಇಪ್ಪತ್ತೊಂದು'),
    (23, 'ಇಪ್ಪತ್ತಮೂರು'), (25, 'ಇಪ್ಪತ್ತೈದು'), (26, 'ಇಪ್ಪತ್ತಾರು'), (99, 'ತೊಂಬತ್ತೊಂಬತ್ತು'),
    (100, 'ನೂರು'), (250, 'ಇನ್ನೂರ ಐವತ್ತು'), (999, 'ಒಂಬೈನೂರ ತೊಂಬತ್ತೊಂಬತ್ತು'),
    (1500, 'ಸಾವಿರದ ಐನೂರು'), (2026, 'ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು'), (100000, 'ಒಂದು ಲಕ್ಷ'),
    (125000, 'ಒಂದು ಲಕ್ಷದ ಇಪ್ಪತ್ತೈದು ಸಾವಿರ'), (25000000, 'ಎರಡು ಕೋಟಿ ಐವತ್ತು ಲಕ್ಷ'),
])
def test_kannada_cardinals(number, words):
    assert kannada_cardinal(number) == words


@pytest.mark.parametrize('words, suffix, out', [
    ('ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು', 'ರಲ್ಲಿ', 'ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರರಲ್ಲಿ'),
    ('ಮೂರು', 'ನೇ', 'ಮೂರನೇ'), ('ಹತ್ತು', 'ಕ್ಕೆ', 'ಹತ್ತಕ್ಕೆ'), ('ಐದು', 'ಗೆ', 'ಐದಕ್ಕೆ'),
    ('ಸಾವಿರ', 'ಕ್ಕೂ', 'ಸಾವಿರಕ್ಕೂ'), ('ಒಂಬತ್ತು ಗಂಟೆ', 'ರಿಂದ', 'ಒಂಬತ್ತು ಗಂಟೆಯಿಂದ'),
    ('ಐನೂರು ರೂಪಾಯಿ', 'ಕ್ಕೆ', 'ಐನೂರು ರೂಪಾಯಿಗೆ'),
])
def test_case_suffixes(words, suffix, out):
    assert attach_suffix(words, suffix) == out


@pytest.mark.parametrize('text, spoken', [
    ('೧೫೦೦ ಮನೆಗಳು', 'ಸಾವಿರದ ಐನೂರು ಮನೆಗಳು'),
    ('2026ರಲ್ಲಿ ಚುನಾವಣೆ', 'ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರರಲ್ಲಿ ಚುನಾವಣೆ'),
    ('ಅವರ 25 ನೇ ಹುಟ್ಟುಹಬ್ಬ', 'ಅವರ ಇಪ್ಪತ್ತೈದನೇ ಹುಟ್ಟುಹಬ್ಬ'),
    ('ಮತದಾನ 72% ಆಗಿದೆ', 'ಮತದಾನ ಶೇಕಡಾ ಎಪ್ಪತ್ತೆರಡು ಆಗಿದೆ'),
    ('ಚಹಾ ₹10.50ಕ್ಕೆ', 'ಚಹಾ ಹತ್ತು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆಗೆ'),
    ('ದಂಡ ₹1,25,000', 'ದಂಡ ಒಂದು ಲಕ್ಷದ ಇಪ್ಪತ್ತೈದು ಸಾವಿರ ರೂಪಾಯಿ'),
    ('ರೈಲು 5:45 pm ಕ್ಕೆ', 'ರೈಲು ಸಂಜೆ ಐದು ಗಂಟೆ ನಲವತ್ತೈದು ನಿಮಿಷಕ್ಕೆ'),
    ('18:00ಕ್ಕೆ', 'ಸಂಜೆ ಆರು ಗಂಟೆಗೆ'),
    ('03-11-2026ರಂದು', 'ಮೂರು ನವೆಂಬರ್ ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರರಂದು'),
    ('98450 12345 ಗೆ ಕರೆ', 'ಒಂಬತ್ತು ಎಂಟು ನಾಲ್ಕು ಐದು ಸೊನ್ನೆ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದಕ್ಕೆ ಕರೆ'),
    ('ಪಿನ್ ಕೋಡ್ 560001', 'ಪಿನ್ ಕೋಡ್ ಐದು ಆರು ಸೊನ್ನೆ ಸೊನ್ನೆ ಸೊನ್ನೆ ಒಂದು'),
    ('ತಾಪಮಾನ 25-30 ಡಿಗ್ರಿ', 'ತಾಪಮಾನ ಇಪ್ಪತ್ತೈದರಿಂದ ಮೂವತ್ತು ಡಿಗ್ರಿ'),
    ('145 km ದೂರ', 'ನೂರ ನಲವತ್ತೈದು ಕಿಲೋಮೀಟರ್ ದೂರ'),
    ('ಡಾ. ರಾಜ್‌ಕುಮಾರ್', 'ಡಾಕ್ಟರ್ ರಾಜ್‌ಕುಮಾರ್'),
    ('ಎಸ್.ಎಂ. ಕೃಷ್ಣ', 'ಎಸ್ ಎಂ ಕೃಷ್ಣ'),
    ('BBMP ಚುನಾವಣೆ', 'b b m p ಚುನಾವಣೆ'),
    ('ISRO ಉಪಗ್ರಹ', 'ಇಸ್ರೋ ಉಪಗ್ರಹ'),
])
def test_kannada_normalization(text, spoken):
    assert normalize_kannada_text(text) == spoken


def test_indian_english():
    assert indian_english_cardinal(125000) == 'one lakh twenty five thousand'
    assert normalize_indian_english('It costs ₹500.') == 'It costs five hundred rupees.'
    assert normalize_indian_english('Rs. 2,50,00,000') == 'two crore fifty lakh rupees'
    assert normalize_indian_english('at 10:30 AM on 03/11/2026') == \
        'at ten thirty A M on third November, twenty twenty six'
    assert normalize_indian_english('Call 98450 12345') == 'Call nine eight four five zero one two three four five'
    assert normalize_indian_english('NASA is ready.') == 'NASA is ready.'


def test_every_rule_reaches_valid_tokens():
    for sentence in ['ಪರೀಕ್ಷೆ 03-11-2026ರಂದು 10:30ಕ್ಕೆ ನಡೆಯಲಿದೆ.', 'ಬೆಲೆ ₹15,999 ಮಾತ್ರ, ಶೇ 50ರಷ್ಟು ರಿಯಾಯಿತಿ!',
                     'The meeting in Bengaluru is at 5:05 pm.', 'Mysuru is 145 km away.']:
        front = kannada.phonemize(sentence)
        assert front.symbols[0] == '<bos>' and front.symbols[-1] == '<eos>'


def test_unknown_names_are_guessed_not_rejected():
    front = kannada.phonemize('ನಾನು Hubballi ಗೆ ಹೋಗುತ್ತೇನೆ.')
    assert front.unknown_words == ('hubballi',)
    assert 'kn_bh' not in front.symbols and front.symbols.count('kn_b') == 2
    assert kannada.romanized_word_phones('Shetty') == ['sh', 'e', 'tt', 'tt', 'i']
    with pytest.raises(ValueError, match='pronunciation overrides'):
        kannada.phonemize('ನಾನು Hubballi ಗೆ ಹೋಗುತ್ತೇನೆ.', guess_unknown=False)
