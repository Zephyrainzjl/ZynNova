"""Audited registry for zero-shot and comparison TTS backends."""

from __future__ import annotations

from ..core import BackendDescriptor, BackendRegistry
from .tts_backends import (
    CosyVoice3Backend,
    ExternalTTSBackend,
    GPTSoVITSAPIBackend,
    IndexTTS25Backend,
    ReferenceAudioTTSBaseline,
    TTSBackend,
)

TTS_BACKENDS: BackendRegistry[TTSBackend] = BackendRegistry("speech-synthesis")
TTS_BACKENDS.register(
    BackendDescriptor(
        name="cosyvoice-3",
        task="speech-synthesis",
        factory=CosyVoice3Backend,
        summary=(
            "Apache-licensed multilingual zero-shot TTS with cross-lingual, instruction, "
            "streaming, and voice-conversion modes through the official API."
        ),
        license_id="Apache-2.0",
        source="https://github.com/QwenAudio/CosyVoice",
        default_rank=5,
        extras=("isolated CosyVoice environment and model weights",),
    )
)
TTS_BACKENDS.register(
    BackendDescriptor(
        name="indextts-2.5",
        task="speech-synthesis",
        factory=IndexTTS25Backend,
        summary=(
            "Controllable zero-shot TTS with multilingual pronunciation, emotion, and "
            "speaking-speed controls through the official IndexTTS-2.5 API."
        ),
        license_id="bilibili Model Use License Agreement",
        source="https://github.com/index-tts/index-tts",
        default_rank=10,
        extras=("isolated IndexTTS environment", "explicit model-license acceptance"),
    )
)
TTS_BACKENDS.register(
    BackendDescriptor(
        name="gpt-sovits-api",
        task="speech-synthesis",
        factory=GPTSoVITSAPIBackend,
        summary="Official api_v2.py HTTP client used for same-hardware baseline comparisons.",
        license_id="MIT + model/dependency terms",
        source="https://github.com/RVC-Boss/GPT-SoVITS",
        default_rank=50,
        extras=("running official GPT-SoVITS API", "allow_remote_backend=True"),
    )
)
TTS_BACKENDS.register(
    BackendDescriptor(
        name="external-tts-contract",
        task="speech-synthesis",
        factory=ExternalTTSBackend,
        summary="Shell-free JSON/argv contract for independently maintained TTS engines.",
        license_id="user-supplied",
        default_rank=100,
    )
)
TTS_BACKENDS.register(
    BackendDescriptor(
        name="reference-audio-baseline",
        task="speech-synthesis",
        factory=ReferenceAudioTTSBaseline,
        summary="Explicit test-only reference passthrough; never claims to synthesize speech.",
        license_id="MIT (ZynNova implementation)",
        default_rank=1000,
        extras=("allow_baseline=True",),
    )
)

PUBLIC_TTS_SOURCES = (
    {
        "name": "Fun-CosyVoice 3.0",
        "source": "https://github.com/QwenAudio/CosyVoice",
        "role": "preferred multilingual and streaming zero-shot TTS backend",
        "adapter": "CosyVoice3Backend",
    },
    {
        "name": "IndexTTS-2.5",
        "source": "https://github.com/index-tts/index-tts",
        "role": "fine-grained emotion, speed, and pronunciation-controlled TTS backend",
        "adapter": "IndexTTS25Backend",
    },
    {
        "name": "GPT-SoVITS",
        "source": "https://github.com/RVC-Boss/GPT-SoVITS",
        "role": "official comparison baseline through api_v2.py",
        "adapter": "GPTSoVITSAPIBackend",
    },
)

__all__ = ["PUBLIC_TTS_SOURCES", "TTS_BACKENDS"]
