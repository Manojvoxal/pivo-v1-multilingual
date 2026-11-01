"""PiVo: compact two-voice English text-to-speech with local ONNX inference."""
from .engine import PiVo
from .runtime import Runtime
__all__ = ["PiVo", "Runtime"]
__version__ = "1.0.0"
