"""Audited voice-conversion backend registry."""

from __future__ import annotations

from ..core import BackendDescriptor, BackendRegistry
from .backends import (
    ExternalVoiceBackend,
    IdentityVoiceBaseline,
    MeanVC2Backend,
    VoiceBackend,
    XVCBackend,
)

VOICE_BACKENDS: BackendRegistry[VoiceBackend] = BackendRegistry("voice-conversion")
VOICE_BACKENDS.register(
    BackendDescriptor(
        name="meanvc2",
        task="voice-conversion",
        factory=MeanVC2Backend,
        summary=(
            "Low-latency zero-shot conditional-flow voice conversion with official "
            "offline and streaming entry points."
        ),
        license_id="Apache-2.0",
        source="https://github.com/ASLP-lab/MeanVC2",
        default_rank=5,
        extras=("external isolated audio environment",),
    )
)
VOICE_BACKENDS.register(
    BackendDescriptor(
        name="xvc",
        task="voice-conversion",
        factory=XVCBackend,
        summary="Zero-shot codec-space voice conversion with offline/streaming controls.",
        license_id="MIT",
        source="https://github.com/Jerrister/X-VC",
        default_rank=10,
        extras=("external isolated audio environment",),
    )
)
VOICE_BACKENDS.register(
    BackendDescriptor(
        name="external-voice-contract",
        task="voice-conversion",
        factory=ExternalVoiceBackend,
        summary="Shell-free argv/file contract for an independently maintained VC stack.",
        license_id="user-supplied",
        default_rank=50,
    )
)
VOICE_BACKENDS.register(
    BackendDescriptor(
        name="identity-baseline",
        task="voice-conversion",
        factory=IdentityVoiceBaseline,
        summary="Explicit test-only passthrough; it never claims to perform conversion.",
        license_id="MIT (ZynNova implementation)",
        default_rank=1000,
        extras=("allow_baseline=True",),
    )
)

PUBLIC_VOICE_SOURCES = (
    {
        "name": "MeanVC2",
        "source": "https://github.com/ASLP-lab/MeanVC2",
        "role": "preferred low-latency zero-shot and real-time voice conversion",
        "license": "Apache-2.0",
    },
    {
        "name": "X-VC",
        "source": "https://github.com/Jerrister/X-VC",
        "role": "codec-space offline and streaming voice conversion",
        "license": "MIT",
    },
)

__all__ = ["PUBLIC_VOICE_SOURCES", "VOICE_BACKENDS"]
