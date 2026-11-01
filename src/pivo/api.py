"""Local, single-inference WAV API. No model downloads occur on request."""
import hmac
import os
import threading
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .engine import wav_bytes


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["pivo-1.4m"] = "pivo-1.4m"
    input: str = Field(min_length=1, max_length=2000)
    voice: Literal["Pi", "Vo"] = "Pi"
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)
    overrides: dict[str, list[str]] | None = None


def create_app(model, *, api_key=None):
    """Create an app around one already-loaded and hash-verified PiVo instance."""
    app = FastAPI(title="PiVo-1.4M", version="1.0.0",
                  description="Two English voices. 16 kHz mono PCM16 WAV. Buffered responses.")
    key = os.getenv("PIVO_API_KEY", "") if api_key is None else api_key
    bearer = HTTPBearer(auto_error=False)
    busy = threading.BoundedSemaphore(1)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if key and (credentials is None or not hmac.compare_digest(credentials.credentials, key)):
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health():
        return {"status": "ready", "model": "pivo-1.4m", "sample_rate": model.sample_rate}

    @app.get("/v1/voices", dependencies=[Depends(authenticate)])
    def voices():
        return {"voices": [{"id": "Pi", "language": "en"}, {"id": "Vo", "language": "en"}]}

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    def speech(request: SpeechRequest):
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "synthesizer busy; retry after the current request", headers={"Retry-After": "1"})
        try:
            audio = model.synthesize(request.input, voice=request.voice,
                                     speed=request.speed, overrides=request.overrides)
            return Response(wav_bytes(audio, model.sample_rate), media_type="audio/wav",
                            headers={"X-PiVo-Sample-Rate": str(model.sample_rate),
                                     "X-PiVo-Samples": str(len(audio))})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            busy.release()

    return app
