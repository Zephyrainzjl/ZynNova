"""Validated configuration for consent-aware voice conversion workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping

from ..core import ConfigurationError, ConsentRequiredError


class VoiceMode(str, Enum):
    """Execution modes supported by the voice backend contract."""

    OFFLINE = "offline"
    STREAMING_FILE = "streaming-file"
    REALTIME = "realtime"


class ConsentBasis(str, Enum):
    """Basis under which a target reference may be used."""

    SELF = "self"
    DIRECT_AUTHORIZATION = "direct-authorization"
    LICENSED_DATASET = "licensed-dataset"
    PUBLIC_DOMAIN = "public-domain"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """Minimal, auditable authorization record.

    The record intentionally does not require a speaker name or other personal data.
    ``evidence`` may point to an authorization document, but the document is hashed
    rather than copied into a run directory.
    """

    confirmed: bool
    basis: ConsentBasis
    purpose: str
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evidence: Path | None = None

    def __post_init__(self) -> None:
        if not self.confirmed:
            raise ConsentRequiredError(
                "target-speaker authorization must be explicitly confirmed"
            )
        purpose = self.purpose.strip()
        if len(purpose) < 3:
            raise ConfigurationError("consent purpose must contain at least 3 characters")
        evidence = None if self.evidence is None else Path(self.evidence)
        if evidence is not None and not evidence.is_file():
            raise FileNotFoundError(evidence)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True, slots=True)
class VoiceRequest:
    """One source/reference conversion request."""

    source_audio: Path
    target_reference: Path
    consent: ConsentRecord
    backend: str = "auto"
    mode: VoiceMode = VoiceMode.OFFLINE
    output_name: str = "converted"
    language: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = Path(self.source_audio)
        target = Path(self.target_reference)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.is_file():
            raise FileNotFoundError(target)
        output_name = self.output_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
            raise ConfigurationError(
                "output_name may contain only letters, digits, dot, underscore, and dash"
            )
        output_name = Path(output_name).stem
        if not output_name:
            raise ConfigurationError("output_name cannot be empty")
        object.__setattr__(self, "source_audio", source)
        object.__setattr__(self, "target_reference", target)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Post-processing, auditing, and backend configuration."""

    output_directory: str = "zynnova_runs/zynvox"
    output_sample_rate: int | None = None
    peak_dbfs: float | None = -1.0
    benchmark: bool = True
    provenance_sidecar: bool = True
    preserve_raw_backend_audio: bool = True
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_sample_rate is not None and self.output_sample_rate < 8_000:
            raise ConfigurationError("output_sample_rate must be at least 8000 Hz")
        if self.peak_dbfs is not None and not -30.0 <= self.peak_dbfs <= 0.0:
            raise ConfigurationError("peak_dbfs must lie in [-30, 0] or be None")
        object.__setattr__(self, "backend_options", dict(self.backend_options))


__all__ = [
    "ConsentBasis",
    "ConsentRecord",
    "VoiceConfig",
    "VoiceMode",
    "VoiceRequest",
]
