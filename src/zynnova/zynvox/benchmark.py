"""Measured voice-conversion performance and optional perceptual evaluators."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ..core.serialization import dump_json, to_jsonable
from .audio import audio_integrity, read_audio, resample


class VoiceEvaluator(Protocol):
    """Optional heavyweight metric provider."""

    name: str

    def evaluate(
        self,
        source_audio: Path,
        target_reference: Path,
        converted_audio: Path,
    ) -> Mapping[str, float | str]:
        """Return measured metrics without inventing missing values."""


@dataclass(frozen=True, slots=True)
class VoiceBenchmark:
    backend: str
    elapsed_s: float
    source_duration_s: float
    output_duration_s: float
    realtime_factor: float
    first_packet_latency_ms: float | None
    duration_ratio: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_fraction: float
    dc_offset: float
    sample_rate: int
    channels: int
    optional_metrics: Mapping[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComparisonThresholds:
    """Thresholds for an evidence-based candidate-versus-baseline decision."""

    maximum_rtf_ratio: float = 1.0
    maximum_latency_ratio: float = 1.0
    minimum_speaker_similarity_delta: float = 0.0
    maximum_content_error_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.maximum_rtf_ratio <= 0.0 or self.maximum_latency_ratio <= 0.0:
            raise ValueError("performance ratios must be positive")


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    candidate_backend: str
    baseline_backend: str
    checks: Mapping[str, Mapping[str, object]]
    all_available_checks_passed: bool
    unavailable_checks: tuple[str, ...]


def benchmark_voice(
    *,
    backend: str,
    source_audio: str | Path,
    target_reference: str | Path,
    converted_audio: str | Path,
    elapsed_s: float,
    first_packet_latency_ms: float | None = None,
    evaluators: Sequence[VoiceEvaluator] = (),
    output_path: str | Path | None = None,
) -> VoiceBenchmark:
    source = Path(source_audio)
    target = Path(target_reference)
    converted = Path(converted_audio)
    source_buffer = read_audio(source)
    output_buffer = read_audio(converted)
    integrity = audio_integrity(output_buffer)
    optional: dict[str, float | str] = {}
    for evaluator in evaluators:
        values = evaluator.evaluate(source, target, converted)
        for key, value in values.items():
            full_key = key if "." in key else f"{evaluator.name}.{key}"
            optional[full_key] = value
    source_duration = source_buffer.duration_s
    report = VoiceBenchmark(
        backend=str(backend),
        elapsed_s=float(elapsed_s),
        source_duration_s=source_duration,
        output_duration_s=output_buffer.duration_s,
        realtime_factor=float(elapsed_s / max(source_duration, 1.0e-12)),
        first_packet_latency_ms=first_packet_latency_ms,
        duration_ratio=float(output_buffer.duration_s / max(source_duration, 1.0e-12)),
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


def compare_benchmarks(
    candidate: VoiceBenchmark,
    baseline: VoiceBenchmark,
    *,
    thresholds: ComparisonThresholds | None = None,
    speaker_metric: str = "speechbrain.speaker_similarity",
    content_metric: str = "faster_whisper.content_error_rate",
) -> ComparisonReport:
    """Compare actual reports; unavailable perceptual metrics remain explicit."""

    limits = thresholds or ComparisonThresholds()
    checks: dict[str, Mapping[str, object]] = {}
    unavailable: list[str] = []

    rtf_ratio = candidate.realtime_factor / max(baseline.realtime_factor, 1.0e-12)
    checks["realtime_factor"] = {
        "candidate": candidate.realtime_factor,
        "baseline": baseline.realtime_factor,
        "ratio": rtf_ratio,
        "threshold": limits.maximum_rtf_ratio,
        "passed": rtf_ratio <= limits.maximum_rtf_ratio,
    }

    if (
        candidate.first_packet_latency_ms is not None
        and baseline.first_packet_latency_ms is not None
    ):
        latency_ratio = candidate.first_packet_latency_ms / max(
            baseline.first_packet_latency_ms, 1.0e-12
        )
        checks["first_packet_latency"] = {
            "candidate": candidate.first_packet_latency_ms,
            "baseline": baseline.first_packet_latency_ms,
            "ratio": latency_ratio,
            "threshold": limits.maximum_latency_ratio,
            "passed": latency_ratio <= limits.maximum_latency_ratio,
        }
    else:
        unavailable.append("first_packet_latency")

    _metric_delta_check(
        checks,
        unavailable,
        "speaker_similarity",
        candidate.optional_metrics,
        baseline.optional_metrics,
        speaker_metric,
        minimum_delta=limits.minimum_speaker_similarity_delta,
    )
    _metric_delta_check(
        checks,
        unavailable,
        "content_error_rate",
        candidate.optional_metrics,
        baseline.optional_metrics,
        content_metric,
        maximum_delta=limits.maximum_content_error_delta,
    )
    passed = all(bool(value["passed"]) for value in checks.values())
    return ComparisonReport(
        candidate_backend=candidate.backend,
        baseline_backend=baseline.backend,
        checks=checks,
        all_available_checks_passed=passed,
        unavailable_checks=tuple(unavailable),
    )


def _metric_delta_check(
    checks: dict[str, Mapping[str, object]],
    unavailable: list[str],
    check_name: str,
    candidate: Mapping[str, float | str],
    baseline: Mapping[str, float | str],
    metric: str,
    *,
    minimum_delta: float | None = None,
    maximum_delta: float | None = None,
) -> None:
    candidate_value = candidate.get(metric)
    baseline_value = baseline.get(metric)
    if not isinstance(candidate_value, (int, float)) or not isinstance(
        baseline_value, (int, float)
    ):
        unavailable.append(check_name)
        return
    delta = float(candidate_value) - float(baseline_value)
    if minimum_delta is not None:
        passed = delta >= minimum_delta
        threshold = minimum_delta
        relation = "minimum_delta"
    else:
        assert maximum_delta is not None
        passed = delta <= maximum_delta
        threshold = maximum_delta
        relation = "maximum_delta"
    checks[check_name] = {
        "metric": metric,
        "candidate": float(candidate_value),
        "baseline": float(baseline_value),
        "delta": delta,
        relation: threshold,
        "passed": passed,
    }


class SpeechBrainSpeakerEvaluator:
    """Cosine target/output similarity using an official SpeechBrain encoder."""

    name = "speechbrain"

    def __init__(
        self,
        *,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.savedir = None if savedir is None else str(Path(savedir))
        self.device = device
        self._classifier: object | None = None

    def _load(self) -> object:
        if self._classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError as exc:
                raise RuntimeError(
                    "SpeechBrain speaker evaluation requires speechbrain and torch"
                ) from exc
            kwargs: dict[str, object] = {"run_opts": {"device": self.device}}
            if self.savedir is not None:
                kwargs["savedir"] = self.savedir
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_id,
                **kwargs,
            )
        return self._classifier

    def evaluate(
        self,
        source_audio: Path,
        target_reference: Path,
        converted_audio: Path,
    ) -> Mapping[str, float]:
        import torch

        classifier = self._load()
        target = resample(read_audio(target_reference, mono=True), 16_000)
        output = resample(read_audio(converted_audio, mono=True), 16_000)
        target_tensor = torch.from_numpy(target.samples[:, 0]).unsqueeze(0).to(self.device)
        output_tensor = torch.from_numpy(output.samples[:, 0]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            target_embedding = classifier.encode_batch(target_tensor).reshape(1, -1)
            output_embedding = classifier.encode_batch(output_tensor).reshape(1, -1)
            similarity = torch.nn.functional.cosine_similarity(
                target_embedding,
                output_embedding,
            ).item()
        return {"speaker_similarity": float(similarity)}


class FasterWhisperContentEvaluator:
    """Source/output content error measured by an actual ASR model."""

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
                raise RuntimeError(
                    "content evaluation requires faster-whisper"
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _transcribe(self, path: Path) -> str:
        model = self._load()
        segments, _ = model.transcribe(
            str(path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def evaluate(
        self,
        source_audio: Path,
        target_reference: Path,
        converted_audio: Path,
    ) -> Mapping[str, float | str]:
        source_text = self._transcribe(source_audio)
        output_text = self._transcribe(converted_audio)
        source_tokens = _tokens(source_text)
        output_tokens = _tokens(output_text)
        error = _edit_distance(source_tokens, output_tokens) / max(1, len(source_tokens))
        return {
            "content_error_rate": float(error),
            "source_transcript": source_text,
            "output_transcript": output_text,
        }


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", text.casefold()).strip()
    if not normalized:
        return []
    if re.search(r"[\u3400-\u9fff]", normalized):
        return [character for character in normalized if not character.isspace()]
    return normalized.split()


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, start=1):
        current = [row]
        for column, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def save_comparison(path: str | Path, report: ComparisonReport) -> Path:
    return dump_json(path, to_jsonable(report))


__all__ = [
    "ComparisonReport",
    "ComparisonThresholds",
    "FasterWhisperContentEvaluator",
    "SpeechBrainSpeakerEvaluator",
    "VoiceBenchmark",
    "VoiceEvaluator",
    "benchmark_voice",
    "compare_benchmarks",
    "save_comparison",
]
