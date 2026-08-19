"""Central authorization policy for every ZynVox execution path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core import ConsentRequiredError
from .schema import ConsentBasis, ConsentRecord


@dataclass(frozen=True, slots=True)
class ConsentPolicyResult:
    record_id: str
    basis: str
    evidence_required: bool
    evidence_present: bool
    authorized: bool


def enforce_consent_record(consent: ConsentRecord) -> ConsentPolicyResult:
    """Enforce the same policy for file, streaming, realtime VC and zero-shot TTS.

    Self-use may be attested directly.  Any use of another person's voice or a
    dataset/public-domain voice needs a concrete authorization/license/source record
    so that selecting a permissive enum value cannot bypass the audit boundary.
    """

    if not consent.confirmed:
        raise ConsentRequiredError("voice use requires explicit authorization")
    evidence_required = consent.basis is not ConsentBasis.SELF
    evidence_present = consent.evidence is not None and Path(consent.evidence).is_file()
    if evidence_required and not evidence_present:
        raise ConsentRequiredError(
            f"consent basis {consent.basis.value!r} requires an evidence file "
            "(authorization, dataset license, or public-domain/source record)"
        )
    return ConsentPolicyResult(
        record_id=consent.record_id,
        basis=consent.basis.value,
        evidence_required=evidence_required,
        evidence_present=evidence_present,
        authorized=True,
    )


__all__ = ["ConsentPolicyResult", "enforce_consent_record"]
