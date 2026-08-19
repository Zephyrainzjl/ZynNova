"""Consent-gated TTS orchestration with benchmarks and provenance."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from ..core import RunManifest, sha256_file
from ..core.serialization import to_jsonable
from .audio import peak_normalize, read_audio, resample, write_wav
from .disclosure import embed_wav_disclosure
from .policy import enforce_consent_record
from .provenance import write_tts_provenance
from .tts_benchmark import TTSEvaluator, benchmark_tts
from .tts_registry import TTS_BACKENDS
from .tts_schema import TTSConfig, TTSRequest
from .tts_types import TTSResult


def run_speech_synthesis(
    request: TTSRequest,
    config: TTSConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
    evaluators: Sequence[TTSEvaluator] = (),
) -> TTSResult:
    """Run one authorized synthesis request and retain a complete artifact graph."""

    config = config or TTSConfig()
    policy = enforce_consent_record(request.consent)
    options = dict(config.backend_options)
    options.update(backend_options or {})
    backend = TTS_BACKENDS.choose(request.backend, **options)
    provenance = {
        "target_reference_sha256": sha256_file(request.target_reference),
        "consent_basis": request.consent.basis.value,
        "consent_record_id": request.consent.record_id,
        "consent_recorded_at": request.consent.recorded_at,
        "authorization_policy": to_jsonable(policy),
    }
    if request.emotion_reference is not None:
        provenance["emotion_reference_sha256"] = sha256_file(request.emotion_reference)
    if request.consent.evidence is not None:
        provenance["consent_evidence_sha256"] = sha256_file(request.consent.evidence)
    manifest = RunManifest(
        workflow="zynnova.zynvox.speech_synthesis",
        backend=backend.name,
        configuration={"request": to_jsonable(request), "config": to_jsonable(config)},
        provenance=provenance,
    )
    run_directory = Path(config.output_directory).expanduser().resolve() / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    raw_copy: Path | None = None
    benchmark_path: Path | None = None
    provenance_path: Path | None = None
    try:
        backend_output = backend.run(request, config, run_directory / "backend")
        if config.preserve_raw_backend_audio:
            raw_directory = run_directory / "exports" / "raw"
            raw_directory.mkdir(parents=True, exist_ok=True)
            raw_copy = raw_directory / f"backend_output{backend_output.audio.suffix or '.wav'}"
            shutil.copy2(backend_output.audio, raw_copy)
            manifest.add_artifact(raw_copy, role="tts_raw_backend_audio", media_type="audio/wav")
        audio = read_audio(backend_output.audio)
        if config.output_sample_rate is not None:
            audio = resample(audio, config.output_sample_rate)
        if config.peak_dbfs is not None:
            audio = peak_normalize(audio, config.peak_dbfs)
        output = write_wav(run_directory / "exports" / f"{request.output_name}.wav", audio)
        metrics: Mapping[str, object] = {}
        if config.benchmark:
            benchmark_path = run_directory / "benchmark.json"
            report = benchmark_tts(
                backend=backend.name,
                text=request.text,
                target_reference=request.target_reference,
                generated_audio=output,
                elapsed_s=backend_output.elapsed_s,
                first_packet_latency_ms=backend_output.first_packet_latency_ms,
                evaluators=evaluators,
                output_path=benchmark_path,
            )
            metrics = to_jsonable(report)
            manifest.add_artifact(benchmark_path, role="tts_benchmark", media_type="application/json")
        disclosure_embedded = False
        if config.embed_disclosure_marker:
            embed_wav_disclosure(
                output,
                workflow="zero-shot-tts",
                backend=backend.name,
                consent_record_id=request.consent.record_id,
            )
            disclosure_embedded = True
            manifest.event("audio_disclosure_embedded", mechanism="RIFF ZYNV")
        if config.provenance_sidecar:
            provenance_path = output.with_suffix(".provenance.json")
            write_tts_provenance(
                provenance_path,
                request=request,
                output_audio=output,
                backend=backend.name,
                model_metadata=backend_output.metadata,
                disclosure_embedded=disclosure_embedded,
            )
            manifest.add_artifact(provenance_path, role="tts_provenance", media_type="application/json")
        manifest.add_artifact(output, role="tts_synthesized_audio", media_type="audio/wav")
        manifest.event(
            "synthesis_completed",
            elapsed_s=backend_output.elapsed_s,
            first_packet_latency_ms=backend_output.first_packet_latency_ms,
            output=str(output),
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(run_directory / "manifest.json")
        raise
    manifest_path = manifest.save(run_directory / "manifest.json")
    return TTSResult(
        output_audio=output,
        raw_backend_audio=raw_copy,
        run_directory=run_directory,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        benchmark_path=benchmark_path,
        metrics=metrics,
    )


__all__ = ["run_speech_synthesis"]
