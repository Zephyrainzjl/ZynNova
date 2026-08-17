from __future__ import annotations

import hashlib
import json
from pathlib import Path

from zynnova.ml.zivar.__main__ import main
from zynnova.ml.zivar.gates import (
    FIXED_GATE_REGISTRY,
    canonical_gate_command,
    run_gate,
)
from zynnova.ml.zivar.maturity import (
    GATE_RESULT_SCHEMA,
    MANDATORY_TARGET_GATES,
    assess_maturity,
    evidence_template,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "zynnova" / "ml" / "zivar"


def test_fixed_registry_is_complete_and_evidence_template_uses_it() -> None:
    assert tuple(FIXED_GATE_REGISTRY) == MANDATORY_TARGET_GATES
    template = evidence_template(PACKAGE_ROOT)
    for name in MANDATORY_TARGET_GATES:
        assert template["gates"][name]["command"] == canonical_gate_command(name)


def test_real_scf_gate_writes_hash_bound_artifact_accepted_by_maturity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "scf.json"
    run = run_gate("scf_stationarity_float64", artifact, package_root=PACKAGE_ROOT)
    assert run.status == "pass"
    assert run.metrics["projected_gradient_rms"] <= 1.0e-9
    assert run.metrics["charge_residual"] <= 1.0e-10
    assert run.metrics["energy_error_eV"] <= 1.0e-12
    assert run.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema"] == GATE_RESULT_SCHEMA
    assert payload["gate"] == "scf_stationarity_float64"
    assert payload["command"] == canonical_gate_command(payload["gate"])
    assert payload["metrics"] == run.metrics

    evidence = evidence_template(PACKAGE_ROOT)
    evidence["hardware"] = {"cpu": "test"}
    evidence["software"] = {"environment": "zynnova"}
    evidence["dataset"] = {"fixture": "deterministic"}
    evidence["gates"][run.gate] = run.evidence_record()
    report = assess_maturity(PACKAGE_ROOT, evidence)
    assert run.gate not in report.missing_target_gates
    assert not report.production_ready


def test_evidence_command_is_never_executed_and_cannot_replace_registry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "scf.json"
    run = run_gate("scf_stationarity_float64", artifact, package_root=PACKAGE_ROOT)
    sentinel = tmp_path / "must-not-exist"
    malicious = f"touch {sentinel}"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["command"] = malicious
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    evidence = evidence_template(PACKAGE_ROOT)
    record = run.evidence_record()
    record["command"] = malicious
    record["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    evidence["gates"][run.gate] = record
    report = assess_maturity(PACKAGE_ROOT, evidence)
    assert run.gate in report.missing_target_gates
    assert not sentinel.exists()


def test_unimplemented_hardware_gate_fails_explicitly_through_cli(
    tmp_path: Path,
    capsys: object,
) -> None:
    artifact = tmp_path / "cuda.json"
    assert main(["run-gate", "cuda_float64", "--artifact", str(artifact)]) == 2
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["status"] == "fail"
    assert "CUDA/MPI/LAMMPS" in result["failure"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["source_hash"]
    assert payload["command"] == "zivar run-gate cuda_float64"
