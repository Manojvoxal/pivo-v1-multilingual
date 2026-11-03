import io
import os
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pivo import kannada, text
from pivo.api import create_app


def test_symbol_table_extends_english_unchanged():
    assert len(kannada.SYMBOLS) == 127 and kannada.SYMBOLS[:79] == text.SYMBOLS
    assert kannada.SYMBOLS[79] == 'kn_a' and len(set(kannada.SYMBOLS)) == 127


def test_kannada_g2p_rules():
    assert kannada.kannada_word_phones('ನಮಸ್ಕಾರ') == ['n', 'a', 'm', 'a', 's', 'k', 'aa', 'r', 'a']
    assert kannada.kannada_word_phones('ಬೆಂಗಳೂರು')[:4] == ['b', 'e', 'ng', 'g']    # anusvara by place
    assert kannada.kannada_word_phones('ಅಂಬಾ')[:3] == ['a', 'm', 'b']
    assert kannada.kannada_word_phones('ದುಃಖ') == ['d', 'u', 'h', 'kh', 'a']        # visarga
    assert kannada.kannada_word_phones('ಕೃಷ್ಣ') == ['k', 'r', 'u', 'ssh', 'nn', 'a']  # vocalic r, virama
    assert kannada.kannada_word_phones('೫') is None


def test_bilingual_token_layout_and_code_mixing():
    front = kannada.phonemize('ನನ್ನ phone ಎಲ್ಲಿದೆ?')
    assert front.symbols[0] == '<bos>' and front.symbols[-2:] == ('?', '<eos>')
    assert ('F', 'OW1', 'N') == tuple(front.symbols[7:10])
    assert front.symbols.count('<space>') == 2
    assert kannada.phonemize('ಇದು ಒಳ್ಳೆಯದು।').symbols[-2] == '.'


def test_english_text_is_unchanged():
    for sentence in ['NASA is ready for the next mission.', 'Hello, my name is Vo.']:
        assert kannada.phonemize(sentence) == text.phonemize(sentence)


def test_kannada_errors_and_overrides():
    assert 'kn_ai' in kannada.phonemize('ನಾನು 5 ಗಂಟೆಗೆ ಬರುತ್ತೇನೆ').symbols       # 5 -> ಐದು
    with pytest.raises(ValueError, match='unsupported character'):
        kannada.phonemize('ನನ್ನ ☺ ಎಲ್ಲಿ')
    with pytest.raises(ValueError, match='pronunciation overrides'):
        kannada.phonemize('ನನ್ನ zzvo ಎಲ್ಲಿ', guess_unknown=False)
    front = kannada.phonemize('ನನ್ನ zzvo ಎಲ್ಲಿ', overrides={'zzvo': ['Z', 'IY1', 'V', 'OW0']})
    assert not front.unknown_words


@pytest.fixture(scope='module')
def model():
    directory = os.getenv('PIVO_TEST_KN_MODEL_DIR')
    if not directory:
        pytest.skip('set PIVO_TEST_KN_MODEL_DIR to a PiVo-KN model folder to run these tests')
    from pivo import PiVo
    return PiVo(directory)


@pytest.mark.parametrize('voice', ['en_f', 'en_m', 'kn_f', 'kn_m'])
def test_all_voices_both_languages(model, voice):
    for sentence in ['ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?', 'Hello, how are you?']:
        audio = model.synthesize(sentence, voice=voice)
        assert audio.dtype == np.float32 and len(audio) > 1600
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) < 1


def test_kn_chunk_parity_and_api(model):
    sentence = 'ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.'
    reference = model.synthesize(sentence, voice='kn_m')
    for size in [1, 47]:
        chunked = model.synthesize(sentence, voice='kn_m', chunk_frames=size)
        assert len(chunked) == len(reference) and np.allclose(chunked, reference, atol=1e-3)
    with TestClient(create_app(model, api_key='')) as client:
        ids = [v['id'] for v in client.get('/v1/voices').json()['voices']]
        assert ids == ['en_f', 'en_m', 'kn_f', 'kn_m']
        response = client.post('/v1/audio/speech', json={'input': sentence, 'voice': 'kn_f'})
        assert response.status_code == 200
        with wave.open(io.BytesIO(response.content)) as wav:
            assert wav.getframerate() == 16000 and wav.getnframes() > 1600
        assert client.post('/v1/audio/speech', json={'input': sentence, 'voice': 'Pi'}).status_code == 422
