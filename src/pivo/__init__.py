"""PiVo: compact text-to-speech with local ONNX inference (English; Kannada + English with PiVo-KN models)."""
from .engine import PiVo
from .runtime import Runtime
__all__ = ["PiVo", "Runtime"]
__version__ = "1.0.0"
