"""Result types shared by ZynVox backends and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class VoiceBackendOutput:
    backend: str
    audio: Path
    elapsed_s: float
    first_packet_latency_ms: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.audio)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.elapsed_s < 0.0:
            raise ValueError("elapsed_s cannot be negative")
        object.__setattr__(self, "audio", path)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class VoiceResult:
    output_audio: Path
    raw_backend_audio: Path | None
    run_directory: Path
    manifest_path: Path
    provenance_path: Path | None
    benchmark_path: Path | None
    metrics: Mapping[str, object]


__all__ = ["VoiceBackendOutput", "VoiceResult"]
