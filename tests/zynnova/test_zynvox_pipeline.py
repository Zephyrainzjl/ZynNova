from __future__ import annotations

import json

import pytest

from zynnova.core import ConsentRequiredError
from zynnova.zynvox import (
    ConsentBasis,
    ConsentRecord,
    TTSConfig,
    TTSRequest,
    VoiceConfig,
    VoiceMode,
    VoiceRequest,
    run_speech_synthesis,
    run_voice_conversion,
)


def _consent() -> ConsentRecord:
    return ConsentRecord(
        confirmed=True,
        basis=ConsentBasis.SELF,
        purpose="local framework verification",
    )


def test_consent_is_mandatory() -> None:
    with pytest.raises(ConsentRequiredError):
        ConsentRecord(
            confirmed=False,
            basis=ConsentBasis.SELF,
            purpose="not authorized",
        )


def test_voice_conversion_baseline_audit_and_benchmark(tmp_path, wav_factory) -> None:
    source = wav_factory(tmp_path / "source.wav", frequency_hz=220.0)
    reference = wav_factory(tmp_path / "reference.wav", frequency_hz=330.0)
    result = run_voice_conversion(
        VoiceRequest(
            source_audio=source,
            target_reference=reference,
            consent=_consent(),
            backend="identity-baseline",
            mode=VoiceMode.OFFLINE,
            output_name="converted_test",
        ),
        VoiceConfig(
            output_directory=str(tmp_path / "voice-runs"),
            output_sample_rate=16_000,
            backend_options={"allow_baseline": True},
        ),
    )
    assert result.output_audio.is_file()
    assert result.raw_backend_audio is not None and result.raw_backend_audio.is_file()
    assert result.benchmark_path is not None and result.benchmark_path.is_file()
    assert result.provenance_path is not None and result.provenance_path.is_file()
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["consent"]["basis"] == "self"
    assert provenance["output"]["sha256"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


def test_tts_baseline_audit_and_benchmark(tmp_path, wav_factory) -> None:
    reference = wav_factory(tmp_path / "tts-reference.wav", frequency_hz=260.0)
    result = run_speech_synthesis(
        TTSRequest(
            text="This is a deterministic plumbing test.",
            target_reference=reference,
            consent=_consent(),
            backend="reference-audio-baseline",
            output_name="tts_test",
            language="EN",
            reference_transcript="Reference speech.",
        ),
        TTSConfig(
            output_directory=str(tmp_path / "tts-runs"),
            output_sample_rate=16_000,
            backend_options={"allow_baseline": True},
        ),
    )
    assert result.output_audio.is_file()
    assert result.raw_backend_audio is not None and result.raw_backend_audio.is_file()
    assert result.benchmark_path is not None and result.benchmark_path.is_file()
    assert result.provenance_path is not None and result.provenance_path.is_file()
    payload = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert payload["consent"]["basis"] == "self"
    assert payload["requested_text"] == "This is a deterministic plumbing test."
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
