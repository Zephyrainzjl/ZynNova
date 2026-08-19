"""Validated requests for consent-aware zero-shot speech synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..core import ConfigurationError
from .schema import ConsentRecord

_INDEX_LANGUAGES = {"AUTO", "ZH", "EN", "JA", "ES", "AR"}


@dataclass(frozen=True, slots=True)
class TTSRequest:
    """One authorized reference-conditioned speech-synthesis request.

    ``reference_transcript`` is optional for backends that infer conditioning directly
    from audio (for example IndexTTS), but is required by prompt-text systems such as
    GPT-SoVITS and recommended for CosyVoice zero-shot synthesis.
    """

    text: str
    target_reference: Path
    consent: ConsentRecord
    backend: str = "auto"
    output_name: str = "synthesized"
    language: str = "AUTO"
    reference_transcript: str | None = None
    emotion_reference: Path | None = None
    emotion_text: str | None = None
    emotion_vector: Sequence[float] | None = None
    emotion_alpha: float = 1.0
    duration_factor: float = 1.0
    style_instruction: str | None = None
    streaming: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = self.text.strip()
        if not text:
            raise ConfigurationError("TTS text cannot be empty")
        reference = Path(self.target_reference)
        if not reference.is_file():
            raise FileNotFoundError(reference)
        output_name = Path(self.output_name.strip()).stem
        if not output_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
            raise ConfigurationError(
                "output_name may contain only letters, digits, dot, underscore, and dash"
            )
        language = self.language.strip().upper()
        if not language:
            raise ConfigurationError("language cannot be empty")
        emotion_reference = (
            None if self.emotion_reference is None else Path(self.emotion_reference)
        )
        if emotion_reference is not None and not emotion_reference.is_file():
            raise FileNotFoundError(emotion_reference)
        vector: tuple[float, ...] | None = None
        if self.emotion_vector is not None:
            vector = tuple(float(item) for item in self.emotion_vector)
            if len(vector) != 8:
                raise ConfigurationError(
                    "emotion_vector must contain 8 values: happy, angry, sad, afraid, "
                    "disgusted, melancholic, surprised, calm"
                )
            if any(not 0.0 <= value <= 1.0 for value in vector):
                raise ConfigurationError("emotion_vector values must lie in [0, 1]")
        if not 0.0 <= self.emotion_alpha <= 1.0:
            raise ConfigurationError("emotion_alpha must lie in [0, 1]")
        if not 0.5 <= self.duration_factor <= 2.0:
            raise ConfigurationError("duration_factor must lie in [0.5, 2.0]")
        transcript = (
            None
            if self.reference_transcript is None
            else self.reference_transcript.strip() or None
        )
        emotion_text = (
            None if self.emotion_text is None else self.emotion_text.strip() or None
        )
        style = (
            None if self.style_instruction is None else self.style_instruction.strip() or None
        )
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "target_reference", reference)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "reference_transcript", transcript)
        object.__setattr__(self, "emotion_reference", emotion_reference)
        object.__setattr__(self, "emotion_text", emotion_text)
        object.__setattr__(self, "emotion_vector", vector)
        object.__setattr__(self, "style_instruction", style)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def index_language(self) -> str:
        """Return a language accepted by IndexTTS-2.5."""

        if self.language not in _INDEX_LANGUAGES:
            raise ConfigurationError(
                f"IndexTTS-2.5 language must be one of {sorted(_INDEX_LANGUAGES)}"
            )
        return self.language


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """Synthesis post-processing, auditing, and optional-backend settings."""

    output_directory: str = "zynnova_runs/zynvox_tts"
    output_sample_rate: int | None = None
    peak_dbfs: float | None = -1.0
    benchmark: bool = True
    provenance_sidecar: bool = True
    embed_disclosure_marker: bool = True
    preserve_raw_backend_audio: bool = True
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_sample_rate is not None and self.output_sample_rate < 8_000:
            raise ConfigurationError("output_sample_rate must be at least 8000 Hz")
        if self.peak_dbfs is not None and not -30.0 <= self.peak_dbfs <= 0.0:
            raise ConfigurationError("peak_dbfs must lie in [-30, 0] or be None")
        if not self.provenance_sidecar:
            raise ConfigurationError(
                "ZynVox TTS requires a provenance sidecar for synthesized speech"
            )
        object.__setattr__(self, "backend_options", dict(self.backend_options))


__all__ = ["TTSConfig", "TTSRequest"]
