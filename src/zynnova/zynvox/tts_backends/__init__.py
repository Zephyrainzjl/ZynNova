"""Speech-synthesis backend implementations."""

from .base import TTSBackend
from .cosyvoice3 import CosyVoice3Backend
from .external import ExternalTTSBackend
from .gpt_sovits import GPTSoVITSAPIBackend
from .indextts25 import IndexTTS25Backend
from .reference import ReferenceAudioTTSBaseline

__all__ = [
    "CosyVoice3Backend",
    "ExternalTTSBackend",
    "GPTSoVITSAPIBackend",
    "IndexTTS25Backend",
    "ReferenceAudioTTSBaseline",
    "TTSBackend",
]
