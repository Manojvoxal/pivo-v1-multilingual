import io
import os
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pivo import PiVo
from pivo.api import create_app


@pytest.fixture(scope='module')
def model():
    directory = os.getenv('PIVO_TEST_MODEL_DIR')
    if not directory:
        pytest.skip('set PIVO_TEST_MODEL_DIR to run actual-model integration tests')
    return PiVo(directory)


@pytest.mark.parametrize('voice', ['Pi', 'Vo'])
def test_both_voices_finite_wav(model, voice):
    audio = model.synthesize('NASA is ready for the next mission.', voice=voice)
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) > 160
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < 1
    with TestClient(create_app(model, api_key='')) as client:
        response = client.post('/v1/audio/speech', json={'input': 'Hello, my name is ' + voice + '.', 'voice': voice})
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getnframes() > 160


def test_per_utterance_state_isolation_and_chunk_parity(model):
    text = 'NASA is ready for the next mission.'
    before = model.synthesize(text, voice='Pi')
    model.synthesize('Hello, my name is Vo.', voice='Vo')
    after = model.synthesize(text, voice='Pi')
    assert np.array_equal(before, after)
    for size in [1, 47]:
        chunked = model.synthesize(text, voice='Pi', chunk_frames=size)
        assert len(chunked) == len(before)
        assert np.allclose(chunked, before, atol=3e-5, rtol=1e-4)
