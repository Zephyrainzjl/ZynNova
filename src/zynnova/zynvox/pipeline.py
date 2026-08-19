"""Consent-gated voice conversion, post-processing, metrics, and provenance."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from ..core import RunManifest, sha256_file
from ..core.serialization import to_jsonable
from .audio import peak_normalize, read_audio, resample, write_wav
from .benchmark import VoiceEvaluator, benchmark_voice
from .disclosure import embed_wav_disclosure
from .policy import enforce_consent_record
from .provenance import write_voice_provenance
from .registry import VOICE_BACKENDS
from .schema import VoiceConfig, VoiceMode, VoiceRequest
from .types import VoiceResult


def run_voice_conversion(
    request: VoiceRequest,
    config: VoiceConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
    evaluators: Sequence[VoiceEvaluator] = (),
) -> VoiceResult:
    """Run one authorized file conversion and retain an auditable artifact graph."""

    if request.mode is VoiceMode.REALTIME:
        raise ValueError(
            "microphone streaming is interactive; use launch_meanvc2_realtime() "
            "with an explicit ConsentRecord"
        )
    config = config or VoiceConfig()
    policy = enforce_consent_record(request.consent)
    options = dict(config.backend_options)
    options.update(backend_options or {})
    backend = VOICE_BACKENDS.choose(request.backend, **options)
    provenance = {
        "source_audio_sha256": sha256_file(request.source_audio),
        "target_reference_sha256": sha256_file(request.target_reference),
        "consent_basis": request.consent.basis.value,
        "consent_record_id": request.consent.record_id,
        "consent_recorded_at": request.consent.recorded_at,
        "authorization_policy": to_jsonable(policy),
    }
    if request.consent.evidence is not None:
        provenance["consent_evidence_sha256"] = sha256_file(request.consent.evidence)
    manifest = RunManifest(
        workflow="zynnova.zynvox.voice_conversion",
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
            manifest.add_artifact(raw_copy, role="voice_raw_backend_audio", media_type="audio/wav")
        audio = read_audio(backend_output.audio)
        if config.output_sample_rate is not None:
            audio = resample(audio, config.output_sample_rate)
        if config.peak_dbfs is not None:
            audio = peak_normalize(audio, config.peak_dbfs)
        output = write_wav(
            run_directory / "exports" / f"{request.output_name}.wav",
            audio,
        )
        metrics: Mapping[str, object] = {}
        if config.benchmark:
            benchmark_path = run_directory / "benchmark.json"
            report = benchmark_voice(
                backend=backend.name,
                source_audio=request.source_audio,
                target_reference=request.target_reference,
                converted_audio=output,
                elapsed_s=backend_output.elapsed_s,
                first_packet_latency_ms=backend_output.first_packet_latency_ms,
                evaluators=evaluators,
                output_path=benchmark_path,
            )
            metrics = to_jsonable(report)
            manifest.add_artifact(
                benchmark_path,
                role="voice_benchmark",
                media_type="application/json",
            )
        disclosure_embedded = False
        if config.embed_disclosure_marker:
            embed_wav_disclosure(
                output,
                workflow="voice-conversion",
                backend=backend.name,
                consent_record_id=request.consent.record_id,
            )
            disclosure_embedded = True
            manifest.event("audio_disclosure_embedded", mechanism="RIFF ZYNV")
        if config.provenance_sidecar:
            provenance_path = output.with_suffix(".provenance.json")
            write_voice_provenance(
                provenance_path,
                request=request,
                output_audio=output,
                backend=backend.name,
                model_metadata=backend_output.metadata,
                disclosure_embedded=disclosure_embedded,
            )
            manifest.add_artifact(
                provenance_path,
                role="voice_provenance",
                media_type="application/json",
            )
        manifest.add_artifact(output, role="voice_converted_audio", media_type="audio/wav")
        manifest.event(
            "conversion_completed",
            elapsed_s=backend_output.elapsed_s,
            output=str(output),
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(run_directory / "manifest.json")
        raise
    manifest_path = manifest.save(run_directory / "manifest.json")
    return VoiceResult(
        output_audio=output,
        raw_backend_audio=raw_copy,
        run_directory=run_directory,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        benchmark_path=benchmark_path,
        metrics=metrics,
    )


__all__ = ["run_voice_conversion"]
