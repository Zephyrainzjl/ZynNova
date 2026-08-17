"""Non-invasive provenance sidecars for converted speech."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..core import sha256_file
from ..core.serialization import dump_json
from .schema import VoiceRequest


def write_voice_provenance(
    path: str | Path,
    *,
    request: VoiceRequest,
    output_audio: str | Path,
    backend: str,
    model_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write hashes and consent basis without embedding private reference audio."""

    evidence = request.consent.evidence
    payload = {
        "schema": "zynnova.voice-provenance/1.0",
        "synthetic_or_converted_audio": True,
        "backend": backend,
        "source": {
            "filename": request.source_audio.name,
            "sha256": sha256_file(request.source_audio),
        },
        "target_reference": {
            "filename": request.target_reference.name,
            "sha256": sha256_file(request.target_reference),
        },
        "output": {
            "filename": Path(output_audio).name,
            "sha256": sha256_file(output_audio),
        },
        "consent": {
            "confirmed": request.consent.confirmed,
            "basis": request.consent.basis.value,
            "purpose": request.consent.purpose,
            "recorded_at": request.consent.recorded_at,
            "evidence_sha256": None if evidence is None else sha256_file(evidence),
        },
        "model_metadata": dict(model_metadata or {}),
        "notice": (
            "This sidecar records provenance; it is not a cryptographic audio "
            "watermark and must remain associated with the output file."
        ),
    }
    return dump_json(path, payload)


__all__ = ["write_tts_provenance", "write_voice_provenance"]


def write_tts_provenance(
    path: str | Path,
    *,
    request: "TTSRequest",
    output_audio: str | Path,
    backend: str,
    model_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write consent, hashes, and synthesis controls for a TTS result."""

    from .tts_schema import TTSRequest

    if not isinstance(request, TTSRequest):
        raise TypeError("request must be a TTSRequest")
    evidence = request.consent.evidence
    payload = {
        "schema": "zynnova.tts-provenance/1.0",
        "synthetic_or_converted_audio": True,
        "backend": backend,
        "requested_text": request.text,
        "language": request.language,
        "target_reference": {
            "filename": request.target_reference.name,
            "sha256": sha256_file(request.target_reference),
        },
        "emotion_reference": None
        if request.emotion_reference is None
        else {
            "filename": request.emotion_reference.name,
            "sha256": sha256_file(request.emotion_reference),
        },
        "controls": {
            "emotion_text": request.emotion_text,
            "emotion_vector": request.emotion_vector,
            "emotion_alpha": request.emotion_alpha,
            "duration_factor": request.duration_factor,
            "style_instruction": request.style_instruction,
            "streaming": request.streaming,
        },
        "output": {
            "filename": Path(output_audio).name,
            "sha256": sha256_file(output_audio),
        },
        "consent": {
            "confirmed": request.consent.confirmed,
            "basis": request.consent.basis.value,
            "purpose": request.consent.purpose,
            "recorded_at": request.consent.recorded_at,
            "evidence_sha256": None if evidence is None else sha256_file(evidence),
        },
        "model_metadata": dict(model_metadata or {}),
        "notice": (
            "This sidecar records synthetic-audio provenance; it is not a cryptographic "
            "watermark and must remain associated with the audio file."
        ),
    }
    return dump_json(path, payload)
