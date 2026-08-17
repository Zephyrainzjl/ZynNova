"""Fail-closed source and target-runtime maturity gates.

Source completeness and production validation are intentionally separate.
Passing the static audit never upgrades an untested CUDA/MPI/LAMMPS build to a
production release.  Target evidence is tied to a deterministic source hash.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "zivar-variational-release-evidence-0.2.0"
GATE_RESULT_SCHEMA = "zivar-gate-result-0.2.0"
MANDATORY_TARGET_GATES = (
    "unit_cpu_float64",
    "scf_stationarity_float64",
    "qeq_dense_matrix_free",
    "direct_ewald_reference",
    "pme_error_convergence",
    "energy_force_finite_difference",
    "stress_finite_difference",
    "spin_field_finite_difference",
    "o3_including_reflection",
    "time_reversal",
    "batch_single_parity",
    "checkpoint_resume",
    "cuda_float32",
    "cuda_float64",
    "cpu_cuda_energy_force_parity",
    "large_system_complexity",
    "multi_gpu_local_rank",
    "lammps_serial",
    "lammps_neighbor_restart",
    "lammps_mpi_2",
    "lammps_mpi_4",
    "spin_lattice_nve_drift",
    "oxidation_calibration_external_split",
    "chgnet_fair_split_benchmark",
)

# Every passing artifact must carry these measured quantities.  This makes a
# hand-written ``{"status": "pass"}`` file insufficient by construction.
GATE_THRESHOLDS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "unit_cpu_float64": (("failed_tests", "max", 0.0), ("passed_tests", "min", 1.0)),
    "scf_stationarity_float64": (
        ("projected_gradient_rms", "max", 1.0e-9),
        ("charge_residual", "max", 1.0e-10),
        ("energy_error_eV", "max", 1.0e-12),
    ),
    "qeq_dense_matrix_free": (("max_charge_error", "max", 1.0e-9),),
    "direct_ewald_reference": (("energy_error_eV", "max", 1.0e-8),),
    "pme_error_convergence": (
        ("energy_error_eV", "max", 1.0e-5),
        ("force_error_eV_per_A", "max", 1.0e-4),
    ),
    "energy_force_finite_difference": (("max_error_eV_per_A", "max", 1.0e-5),),
    "stress_finite_difference": (("max_error_eV_per_A3", "max", 1.0e-6),),
    "spin_field_finite_difference": (("max_error_eV_per_muB", "max", 1.0e-6),),
    "o3_including_reflection": (("max_equivariance_error", "max", 1.0e-9),),
    "time_reversal": (("max_energy_error_eV", "max", 1.0e-9),),
    "batch_single_parity": (("max_error", "max", 1.0e-9),),
    "checkpoint_resume": (("next_step_parameter_error", "max", 0.0),),
    "cuda_float32": (("nonfinite_values", "max", 0.0),),
    "cuda_float64": (("nonfinite_values", "max", 0.0),),
    "cpu_cuda_energy_force_parity": (("max_force_error_eV_per_A", "max", 1.0e-4),),
    "large_system_complexity": (("memory_exponent", "max", 1.25),),
    "multi_gpu_local_rank": (("max_rank_error", "max", 1.0e-5),),
    "lammps_serial": (("max_force_error_eV_per_A", "max", 1.0e-5),),
    "lammps_neighbor_restart": (("restart_energy_error_eV", "max", 1.0e-8),),
    "lammps_mpi_2": (("max_force_error_eV_per_A", "max", 1.0e-5),),
    "lammps_mpi_4": (("max_force_error_eV_per_A", "max", 1.0e-5),),
    "spin_lattice_nve_drift": (("relative_energy_drift_per_ps", "max", 1.0e-4),),
    "oxidation_calibration_external_split": (("ece", "max", 0.05),),
    "chgnet_fair_split_benchmark": (("split_overlap", "max", 0.0),),
}


@dataclass(frozen=True, slots=True)
class MaturityReport:
    source_hash: str
    source_checks: dict[str, bool]
    evidence_checks: dict[str, bool]
    missing_target_gates: tuple[str, ...]
    source_ready: bool
    production_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_tree_hash(package_root: str | Path) -> str:
    root = Path(package_root).resolve()
    digest = hashlib.sha256()
    paths = {
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".json"}
        and "__pycache__" not in path.parts
    }
    repository_root = root.parents[3] if len(root.parents) > 3 else None
    if repository_root is not None and (repository_root / "pyproject.toml").is_file():
        paths.update(
            path
            for path in (
                repository_root / "pyproject.toml",
                repository_root / "CMakeLists.txt",
                repository_root / "cpp" / "CMakeLists.txt",
            )
            if path.is_file()
        )
        for native_root in (
            repository_root / "cpp" / "include" / "zynnova" / "zivar",
            repository_root / "cpp" / "src" / "zivar",
        ):
            if native_root.is_dir():
                paths.update(path for path in native_root.rglob("*") if path.is_file())
        native_test = repository_root / "cpp" / "tests" / "test_zivar_kokkos.cpp"
        if native_test.is_file():
            paths.add(native_test)
    for path in sorted(paths):
        if repository_root is not None and path.is_relative_to(repository_root):
            label = path.relative_to(repository_root).as_posix()
        else:
            label = path.relative_to(root).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit_source_tree(package_root: str | Path) -> tuple[str, dict[str, bool]]:
    root = Path(package_root).resolve()
    required = {
        "functional.py": ("ElectroSpinFunctional", "EnergyBreakdown"),
        "scf.py": ("solve_quadratic_scf", "SCFConvergenceError"),
        "pme.py": ("pme_energy", "PMEPlan"),
        "gates.py": ("FIXED_GATE_REGISTRY", "run_gate"),
        "magnetism.py": ("SpinLatticeHamiltonian", "dmi_energy"),
        "spin_dynamics.py": ("llg_midpoint_step", "run_spin_lattice_dynamics"),
        "lammps.py": ("allgather", "fix external pf/callback"),
        "oxidation.py": ("exact_charge_balanced_states", "confidence"),
        "trainer.py": ("assert_model_optimizer_finite", "allow_amp_second_order"),
    }
    checks: dict[str, bool] = {"package_root": root.is_dir()}
    syntax_ok = True
    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            syntax_ok = False
    checks["python_ast"] = syntax_ok
    for name, tokens in required.items():
        path = root / name
        source = path.read_text(encoding="utf-8") if path.is_file() else ""
        checks[f"required_{name}"] = bool(source) and all(token in source for token in tokens)
    tests = root / "tests"
    checks["symmetry_tests"] = all(
        (tests / name).is_file()
        for name in ("test_magnetism.py", "test_polar_electrostatics.py")
    )
    checks["variational_scf_present"] = (root / "scf.py").is_file()
    repository_root = root.parents[3] if len(root.parents) > 3 else None
    if repository_root is not None and (repository_root / "pyproject.toml").is_file():
        native_files = (
            repository_root / "cpp" / "include" / "zynnova" / "zivar" / "matrix_free.hpp",
            repository_root / "cpp" / "src" / "zivar" / "matrix_free.cpp",
            repository_root / "cpp" / "tests" / "test_zivar_kokkos.cpp",
        )
        checks["native_kokkos_sources"] = all(path.is_file() for path in native_files)
        cmake = (repository_root / "cpp" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        checks["native_kokkos_build_rule"] = all(
            token in cmake
            for token in (
                "ZYNNOVA_BUILD_ZIVAR_KOKKOS",
                "Kokkos::kokkos",
                "zynnova_zivar_kokkos_core_tests",
            )
        )
    return source_tree_hash(root), checks


def _load_evidence(evidence: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {}
    if isinstance(evidence, dict):
        return evidence
    return json.loads(Path(evidence).read_text(encoding="utf-8"))


def _artifact_matches(name: str, gate: Any, source_hash: str) -> bool:
    from .gates import canonical_gate_command

    if not isinstance(gate, dict):
        return False
    canonical_command = canonical_gate_command(name)
    artifact = gate.get("artifact")
    expected = gate.get("artifact_sha256")
    metrics = gate.get("metrics")
    if (
        not artifact or not expected or len(str(expected)) != 64
        or not isinstance(metrics, dict) or not metrics
    ):
        return False
    path = Path(artifact).expanduser()
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(expected).lower():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        result.get("schema") != GATE_RESULT_SCHEMA
        or result.get("gate") != name
        or result.get("source_hash") != source_hash
        or result.get("status") != "pass"
        or gate.get("command") != canonical_command
        or result.get("command") != canonical_command
        or result.get("metrics") != metrics
    ):
        return False
    for metric, operation, threshold in GATE_THRESHOLDS[name]:
        value = metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if operation == "max" and float(value) > threshold:
            return False
        if operation == "min" and float(value) < threshold:
            return False
    return True


def assess_maturity(
    package_root: str | Path,
    evidence: str | Path | dict[str, Any] | None = None,
) -> MaturityReport:
    source_hash, source_checks = audit_source_tree(package_root)
    payload = _load_evidence(evidence)
    evidence_checks = {
        "schema": payload.get("schema") == EVIDENCE_SCHEMA,
        "source_hash": payload.get("source_hash") == source_hash,
        "hardware_recorded": bool(payload.get("hardware")),
        "software_recorded": bool(payload.get("software")),
        "dataset_recorded": bool(payload.get("dataset")),
    }
    gates = payload.get("gates", {})
    missing = tuple(
        name for name in MANDATORY_TARGET_GATES
        if not isinstance(gates.get(name), dict)
        or gates[name].get("status") != "pass"
        or not gates[name].get("command")
        or not _artifact_matches(name, gates[name], source_hash)
    )
    evidence_checks["all_target_gates"] = not missing
    source_ready = all(source_checks.values())
    production_ready = source_ready and all(evidence_checks.values())
    return MaturityReport(
        source_hash=source_hash,
        source_checks=source_checks,
        evidence_checks=evidence_checks,
        missing_target_gates=missing,
        source_ready=source_ready,
        production_ready=production_ready,
    )


def assert_release_ready(
    package_root: str | Path,
    evidence: str | Path | dict[str, Any],
) -> MaturityReport:
    report = assess_maturity(package_root, evidence)
    if not report.production_ready:
        failed_source = tuple(name for name, value in report.source_checks.items() if not value)
        failed_evidence = tuple(name for name, value in report.evidence_checks.items() if not value)
        raise RuntimeError(
            "ZIVAR production release gate failed; "
            f"source={failed_source}, evidence={failed_evidence}, "
            f"target_gates={report.missing_target_gates}"
        )
    return report


def evidence_template(package_root: str | Path) -> dict[str, Any]:
    from .gates import canonical_gate_command

    source_hash, _ = audit_source_tree(package_root)
    return {
        "schema": EVIDENCE_SCHEMA,
        "source_hash": source_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {},
        "software": {},
        "dataset": {},
        "gates": {
            name: {
                "status": "not_run",
                "command": canonical_gate_command(name),
                "artifact": "",
                "artifact_sha256": "", "metrics": {},
            }
            for name in MANDATORY_TARGET_GATES
        },
    }


__all__ = [
    "EVIDENCE_SCHEMA", "GATE_RESULT_SCHEMA", "GATE_THRESHOLDS",
    "MANDATORY_TARGET_GATES", "MaturityReport",
    "assert_release_ready", "assess_maturity", "audit_source_tree",
    "evidence_template", "source_tree_hash",
]
