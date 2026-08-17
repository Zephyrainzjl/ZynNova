from __future__ import annotations

from pathlib import Path

import pytest

from zynnova.ml.zivar.maturity import (
    MANDATORY_TARGET_GATES,
    assert_release_ready,
    assess_maturity,
    evidence_template,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_audit_passes_but_empty_target_evidence_fails_closed() -> None:
    report = assess_maturity(ROOT)
    assert report.source_ready
    assert not report.production_ready
    assert set(report.missing_target_gates) == set(MANDATORY_TARGET_GATES)
    with pytest.raises(RuntimeError, match="production release gate failed"):
        assert_release_ready(ROOT, evidence_template(ROOT))


def test_self_reported_pass_artifacts_are_rejected(tmp_path: Path) -> None:
    import hashlib

    evidence = evidence_template(ROOT)
    evidence["hardware"] = {"gpu": "recorded"}
    evidence["software"] = {"torch": "recorded"}
    evidence["dataset"] = {"split_hash": "recorded"}
    gates = {}
    for name in MANDATORY_TARGET_GATES:
        artifact = tmp_path / f"{name}.json"
        artifact.write_text('{"status":"pass"}\n', encoding="utf-8")
        gates[name] = {
            "status": "pass",
            "command": "pytest ...",
            "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "metrics": {"passed": True},
        }
    evidence["gates"] = gates
    report = assess_maturity(ROOT, evidence)
    assert not report.production_ready
    assert set(report.missing_target_gates) == set(MANDATORY_TARGET_GATES)
