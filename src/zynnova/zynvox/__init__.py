"""ZynVox: isolated, consent-aware voice conversion and speech synthesis."""

from .backends import (
    ExternalVoiceBackend,
    IdentityVoiceBaseline,
    MeanVC2Backend,
    XVCBackend,
    launch_meanvc2_realtime,
)
from .benchmark import (
    ComparisonReport,
    ComparisonThresholds,
    FasterWhisperContentEvaluator,
    SpeechBrainSpeakerEvaluator,
    VoiceBenchmark,
    benchmark_voice,
    compare_benchmarks,
)
from .pipeline import run_voice_conversion
from .registry import PUBLIC_VOICE_SOURCES, VOICE_BACKENDS
from .schema import ConsentBasis, ConsentRecord, VoiceConfig, VoiceMode, VoiceRequest
from .tts_backends import (
    CosyVoice3Backend,
    ExternalTTSBackend,
    GPTSoVITSAPIBackend,
    IndexTTS25Backend,
    ReferenceAudioTTSBaseline,
)
from .tts_benchmark import (
    FasterWhisperTextEvaluator,
    SpeechBrainTTSSpeakerEvaluator,
    TTSBenchmark,
    benchmark_tts,
)
from .tts_pipeline import run_speech_synthesis
from .tts_registry import PUBLIC_TTS_SOURCES, TTS_BACKENDS
from .tts_schema import TTSConfig, TTSRequest
from .tts_types import TTSBackendOutput, TTSResult
from .types import VoiceBackendOutput, VoiceResult

__all__ = [
    "ComparisonReport",
    "ComparisonThresholds",
    "ConsentBasis",
    "ConsentRecord",
    "CosyVoice3Backend",
    "ExternalTTSBackend",
    "ExternalVoiceBackend",
    "FasterWhisperContentEvaluator",
    "FasterWhisperTextEvaluator",
    "GPTSoVITSAPIBackend",
    "IdentityVoiceBaseline",
    "IndexTTS25Backend",
    "MeanVC2Backend",
    "PUBLIC_TTS_SOURCES",
    "PUBLIC_VOICE_SOURCES",
    "ReferenceAudioTTSBaseline",
    "SpeechBrainSpeakerEvaluator",
    "SpeechBrainTTSSpeakerEvaluator",
    "TTS_BACKENDS",
    "TTSBackendOutput",
    "TTSBenchmark",
    "TTSConfig",
    "TTSRequest",
    "TTSResult",
    "VOICE_BACKENDS",
    "VoiceBackendOutput",
    "VoiceBenchmark",
    "VoiceConfig",
    "VoiceMode",
    "VoiceRequest",
    "VoiceResult",
    "XVCBackend",
    "benchmark_tts",
    "benchmark_voice",
    "compare_benchmarks",
    "launch_meanvc2_realtime",
    "run_speech_synthesis",
    "run_voice_conversion",
]
