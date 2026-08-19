"""Non-robust in-file disclosure markers for ZynVox WAV outputs.

The marker is deliberately described as disclosure metadata, not as an adversarially
robust watermark.  Robust watermarking requires a separately validated signal/model
scheme; provenance code must not over-claim that property.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Mapping


def embed_wav_disclosure(
    path: str | Path,
    *,
    workflow: str,
    backend: str,
    consent_record_id: str,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("embedded disclosure currently supports RIFF/WAVE output only")
    payload = json.dumps(
        {
            "schema": "zynnova.audio-disclosure/1.0",
            "synthetic_or_converted_audio": True,
            "workflow": str(workflow),
            "backend": str(backend),
            "consent_record_id": str(consent_record_id),
            "notice": "AI-generated or voice-converted audio; see provenance sidecar.",
            "metadata": dict(metadata or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) % 2:
        payload += b"\x00"
    chunk = b"ZYNV" + struct.pack("<I", len(payload)) + payload
    updated = bytearray(raw + chunk)
    updated[4:8] = struct.pack("<I", len(updated) - 8)
    target.write_bytes(updated)
    return target


def has_wav_disclosure(path: str | Path) -> bool:
    raw = Path(path).read_bytes()
    return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and b"ZYNV" in raw[12:]


__all__ = ["embed_wav_disclosure", "has_wav_disclosure"]
