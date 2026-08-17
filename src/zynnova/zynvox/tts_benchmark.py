"""Measured zero-shot TTS performance with optional perceptual evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..core.serialization import dump_json
from .audio import audio_integrity, read_audio
from .benchmark import SpeechBrainSpeakerEvaluator, _edit_distance, _tokens


class TTSEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        text: str,
        target_reference: Path,
        generated_audio: Path,
    ) -> Mapping[str, float | str]:
        """Return measured metrics without substituting invented values."""


@dataclass(frozen=True, slots=True)
class TTSBenchmark:
    backend: str
    elapsed_s: float
    output_duration_s: float
    generation_realtime_factor: float
    first_packet_latency_ms: float | None
    text_characters: int
    characters_per_second: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_fraction: float
    dc_offset: float
    sample_rate: int
    channels: int
    optional_metrics: Mapping[str, float | str] = field(default_factory=dict)


class SpeechBrainTTSSpeakerEvaluator:
    """Adapt the official SpeechBrain embedding metric to TTS outputs."""

    name = "speechbrain"

    def __init__(self, **kwargs: object) -> None:
        self._delegate = SpeechBrainSpeakerEvaluator(**kwargs)

    def evaluate(
        self,
        text: str,
        target_reference: Path,
        generated_audio: Path,
    ) -> Mapping[str, float]:
        del text
        return self._delegate.evaluate(target_reference, target_reference, generated_audio)


class FasterWhisperTextEvaluator:
    """Measure output transcription error against the requested text."""

    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("TTS content evaluation requires faster-whisper") from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def evaluate(
        self,
        text: str,
        target_reference: Path,
        generated_audio: Path,
    ) -> Mapping[str, float | str]:
        del target_reference
        model = self._load()
        segments, _ = model.transcribe(
            str(generated_audio),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        output_text = " ".join(segment.text.strip() for segment in segments).strip()
        expected_tokens = _tokens(text)
        output_tokens = _tokens(output_text)
        error = _edit_distance(expected_tokens, output_tokens) / max(1, len(expected_tokens))
        return {
            "content_error_rate": float(error),
            "expected_text": text,
            "output_transcript": output_text,
        }


def benchmark_tts(
    *,
    backend: str,
    text: str,
    target_reference: str | Path,
    generated_audio: str | Path,
    elapsed_s: float,
    first_packet_latency_ms: float | None = None,
    evaluators: Sequence[TTSEvaluator] = (),
    output_path: str | Path | None = None,
) -> TTSBenchmark:
    output = Path(generated_audio)
    reference = Path(target_reference)
    buffer = read_audio(output)
    integrity = audio_integrity(buffer)
    optional: dict[str, float | str] = {}
    for evaluator in evaluators:
        for key, value in evaluator.evaluate(text, reference, output).items():
            optional[key if "." in key else f"{evaluator.name}.{key}"] = value
    duration = buffer.duration_s
    report = TTSBenchmark(
        backend=str(backend),
        elapsed_s=float(elapsed_s),
        output_duration_s=duration,
        generation_realtime_factor=float(elapsed_s / max(duration, 1.0e-12)),
        first_packet_latency_ms=first_packet_latency_ms,
        text_characters=len(text),
        characters_per_second=float(len(text) / max(duration, 1.0e-12)),
        peak_dbfs=integrity.peak_dbfs,
        rms_dbfs=integrity.rms_dbfs,
        clipping_fraction=integrity.clipping_fraction,
        dc_offset=integrity.dc_offset,
        sample_rate=integrity.sample_rate,
        channels=integrity.channels,
        optional_metrics=optional,
    )
    if output_path is not None:
        dump_json(output_path, report)
    return report


__all__ = [
    "FasterWhisperTextEvaluator",
    "SpeechBrainTTSSpeakerEvaluator",
    "TTSBenchmark",
    "TTSEvaluator",
    "benchmark_tts",
]
