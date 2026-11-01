import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pivo.api import create_app
from pivo.engine import MODEL_FILES, PiVo, verify_model, wav_bytes
from pivo.text import normalize, phonemize


def test_nasa_word_and_unknown_override():
    assert normalize('NASA and CPU') == 'nasa and C P U'
    with pytest.raises(ValueError, match='pronunciation overrides'):
        phonemize('Hello, Zzvo.')
    front = phonemize('Hello, Zzvo.', overrides={'zzvo': ['Z', 'IY1', 'V', 'OW0']})
    assert not front.unknown_words
    assert ['N', 'AE1', 'S', 'AH0'] == [s for s in phonemize('NASA').symbols if s not in ['<bos>', '<eos>']]


def test_model_integrity_and_path_contract(tmp_path):
    rows = []
    for name in MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'fixture')
        rows.append({'path': name, 'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()})
    manifest = {'format': 'pivo-model-manifest-v1', 'files': rows}
    (tmp_path / 'model_manifest.json').write_text(json.dumps(manifest))
    assert verify_model(tmp_path)[0] == tmp_path
    (tmp_path / 'onnx/encoder.onnx').write_bytes(b'tamper!')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_model(tmp_path)
    rows[0]['path'] = '../outside.onnx'
    (tmp_path / 'model_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='five PiVo runtime assets'):
        verify_model(tmp_path)


def test_validation_before_inference():
    engine = object.__new__(PiVo)
    engine.max_chars, engine.max_tokens = 2000, 512
    for text, voice, speed, overrides in [
        ('', 'Pi', 1, None), ('...', 'Pi', 1, None),
        ('a' * 2001, 'Pi', 1, None), ('Hello', 'invalid', 1, None),
        ('Hello', 'Pi', float('nan'), None), ('Hello', 'Pi', 2.1, None),
        ('Hello', 'Pi', 1, {'hello': [123]}),
    ]:
        with pytest.raises(ValueError):
            engine._validate(text, voice, speed, overrides)
    engine.max_tokens = 3
    with pytest.raises(ValueError, match='frontend tokens'):
        engine._validate('NASA is ready.', 'Pi', 1, None)


def test_wav_finite_mono_contract():
    import wave
    data = wav_bytes(np.array([0, 0.25, -0.5], dtype=np.float32))
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 3)
    with pytest.raises(ValueError, match='finite mono'):
        wav_bytes([float('nan')])
    with pytest.raises(ValueError, match='finite mono'):
        wav_bytes([[0]])


class FakeModel:
    sample_rate = 16000
    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
    def synthesize(self, text, **kwargs):
        if text == 'unknown':
            raise ValueError('pronunciation override required')
        if text == 'wait':
            self.started.set()
            assert self.release.wait(5)
        return np.zeros(160, dtype=np.float32)


def test_api_auth_wav_and_validation():
    with TestClient(create_app(FakeModel(), api_key='test-secret')) as client:
        assert client.get('/health').json()['status'] == 'ready'
        assert client.get('/v1/voices').status_code == 401
        headers = {'Authorization': 'Bearer test-secret'}
        assert client.get('/v1/voices', headers=headers).json()['voices'][1]['id'] == 'Vo'
        response = client.post('/v1/audio/speech', headers=headers, json={'input': 'Hello', 'voice': 'Vo'})
        assert response.status_code == 200 and response.content[:4] == b'RIFF'
        assert response.headers['content-type'] == 'audio/wav'
        assert response.headers['x-pivo-samples'] == '160'
        for payload in [{'input': 'unknown'}, {'input': 'x', 'speed': 3},
                        {'input': 'x', 'response_format': 'mp3'}, {'input': 'x', 'unrecognized': 1}]:
            assert client.post('/v1/audio/speech', headers=headers, json=payload).status_code == 422
        # Failure must release the semaphore for subsequent synthesis.
        assert client.post('/v1/audio/speech', headers=headers, json={'input': 'Hello'}).status_code == 200


def test_api_busy_returns_retry_instead_of_queueing():
    model = FakeModel()
    with TestClient(create_app(model, api_key='')) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, '/v1/audio/speech', json={'input': 'wait'})
        try:
            assert model.started.wait(3)
            response = client.post('/v1/audio/speech', json={'input': 'Hello'})
            assert response.status_code == 429 and response.headers['retry-after'] == '1'
        finally:
            model.release.set()
        assert pending.result().status_code == 200
