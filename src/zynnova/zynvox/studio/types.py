"""Stable Studio schemas independent of a particular acoustic model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..schema import ConsentRecord


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    voice_id: str
    reference_audio: Path
    consent: ConsentRecord
    reference_text: str | None = None
    language: str = "auto"
    model: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path=Path(self.reference_audio)
        if not path.is_file(): raise FileNotFoundError(path)
        if not self.voice_id.strip(): raise ValueError("voice_id cannot be empty")
        object.__setattr__(self,"reference_audio",path.resolve())
        object.__setattr__(self,"metadata",dict(self.metadata))


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    text: str
    voice_id: str
    language: str = "auto"
    output_name: str = "speech"
    model: str | None = None
    seed: int = -1
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed: float = 1.0
    repetition_penalty: float = 1.35
    batch_size: int = 1
    split_method: str = "auto"
    streaming: bool = False
    parallel_infer: bool = True
    fragment_interval: float = 0.3
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip(): raise ValueError("text cannot be empty")
        if not self.voice_id.strip(): raise ValueError("voice_id cannot be empty")
        if self.top_k < 1: raise ValueError("top_k must be positive")
        if not 0 < self.top_p <= 1: raise ValueError("top_p must lie in (0,1]")
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        if not 0.25 <= self.speed <= 4.0: raise ValueError("speed must lie in [0.25,4]")
        object.__setattr__(self,"extra",dict(self.extra))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    audio: Path
    engine: str
    model: str | None
    elapsed_s: float
    sample_rate: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


__all__ = ["GenerationRequest", "GenerationResult", "VoiceProfile"]
