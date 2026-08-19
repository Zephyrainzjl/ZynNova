"""Auditable provenance sidecars for converted and synthesized speech."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ..core import sha256_file
from ..core.serialization import dump_json
from .schema import VoiceRequest


def _finalize(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    payload["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _consent_payload(consent: object) -> dict[str, object]:
    value = consent
    evidence = value.evidence  # type: ignore[attr-defined]
    return {
        "record_id": value.record_id,  # type: ignore[attr-defined]
        "confirmed": value.confirmed,  # type: ignore[attr-defined]
        "basis": value.basis.value,  # type: ignore[attr-defined]
        "purpose": value.purpose,  # type: ignore[attr-defined]
        "recorded_at": value.recorded_at,  # type: ignore[attr-defined]
        "evidence_filename": None if evidence is None else evidence.name,
        "evidence_sha256": None if evidence is None else sha256_file(evidence),
    }


def write_voice_provenance(
    path: str | Path,
    *,
    request: VoiceRequest,
    output_audio: str | Path,
    backend: str,
    model_metadata: Mapping[str, object] | None = None,
    disclosure_embedded: bool = False,
) -> Path:
    """Write hashes and authorization evidence without copying private reference audio."""

    payload: dict[str, object] = {
        "schema": "zynnova.voice-provenance/2.0",
        "synthetic_or_converted_audio": True,
        "workflow": "voice-conversion",
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
        "consent": _consent_payload(request.consent),
        "media_disclosure": {
            "embedded_marker": bool(disclosure_embedded),
            "mechanism": "RIFF ZYNV metadata chunk" if disclosure_embedded else None,
            "robust_watermark": False,
        },
        "model_metadata": dict(model_metadata or {}),
        "notice": (
            "This record is tamper-evident provenance metadata, not a claim of "
            "adversarially robust audio watermarking or speaker authorization by itself."
        ),
    }
    return dump_json(path, _finalize(payload))


def write_tts_provenance(
    path: str | Path,
    *,
    request: "TTSRequest",
    output_audio: str | Path,
    backend: str,
    model_metadata: Mapping[str, object] | None = None,
    disclosure_embedded: bool = False,
) -> Path:
    """Write consent, hashes, synthesis controls and disclosure status for TTS."""

    from .tts_schema import TTSRequest

    if not isinstance(request, TTSRequest):
        raise TypeError("request must be a TTSRequest")
    payload: dict[str, object] = {
        "schema": "zynnova.tts-provenance/2.0",
        "synthetic_or_converted_audio": True,
        "workflow": "zero-shot-tts",
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
        "consent": _consent_payload(request.consent),
        "media_disclosure": {
            "embedded_marker": bool(disclosure_embedded),
            "mechanism": "RIFF ZYNV metadata chunk" if disclosure_embedded else None,
            "robust_watermark": False,
        },
        "model_metadata": dict(model_metadata or {}),
        "notice": (
            "This record is tamper-evident provenance metadata, not a claim of "
            "adversarially robust audio watermarking or speaker authorization by itself."
        ),
    }
    return dump_json(path, _finalize(payload))


__all__ = ["write_tts_provenance", "write_voice_provenance"]
