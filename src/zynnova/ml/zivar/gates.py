"""Executable, hash-bound maturity gates for the ZIVAR production core.

Only the immutable registry in this module can select work to execute.  In
particular, release-evidence JSON is data that is validated by
``maturity.py``; no command stored in evidence is ever passed to a shell or a
subprocess.  Every implemented CPU gate below either evaluates a deterministic
float64 numerical problem directly or invokes one fixed pytest selection with
``shell=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class GateDefinition:
    """One immutable gate selected by name, never by an evidence command."""

    name: str
    command: str
    executor: str | None
    unavailable_reason: str | None = None

    @property
    def implemented(self) -> bool:
        return self.executor is not None


@dataclass(frozen=True, slots=True)
class GateRun:
    """A completed artifact write plus the record copied into evidence."""

    status: str
    artifact: str
    artifact_sha256: str
    gate: str
    command: str
    metrics: dict[str, float | int]
    failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evidence_record(self) -> dict[str, Any]:
        """Return the exact gate entry accepted by :func:`assess_maturity`."""

        return {
            "status": self.status,
            "command": self.command,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "metrics": dict(self.metrics),
        }


def _definition(name: str, executor: str | None, reason: str | None = None) -> GateDefinition:
    return GateDefinition(
        name=name,
        command=f"zivar run-gate {name}",
        executor=executor,
        unavailable_reason=reason,
    )


_HARDWARE_NOT_CERTIFIED = (
    "this target-runtime gate has no portable CPU implementation; run it only "
    "after a dedicated CUDA/MPI/LAMMPS harness is implemented for the target"
)
_DATASET_NOT_CERTIFIED = (
    "this gate requires an immutable external dataset/split manifest and is not "
    "implemented by the source-only runner"
)
_NOT_IMPLEMENTED = "this maturity gate does not yet have an executable ZIVAR harness"

# This is deliberately a complete, fixed registry.  Values are symbolic
# executor identifiers resolved through the private table below; evidence can
# neither add a gate nor replace its implementation.
FIXED_GATE_REGISTRY: Mapping[str, GateDefinition] = MappingProxyType(
    {
        item.name: item
        for item in (
            _definition("unit_cpu_float64", "unit_cpu"),
            _definition("scf_stationarity_float64", "scf_stationarity"),
            _definition("qeq_dense_matrix_free", "qeq_dense_matrix_free"),
            _definition("direct_ewald_reference", "direct_ewald_reference"),
            _definition("pme_error_convergence", "pme_error_convergence"),
            _definition("energy_force_finite_difference", "energy_force_fd"),
            _definition("stress_finite_difference", "stress_fd"),
            _definition("spin_field_finite_difference", "spin_field_fd"),
            _definition("o3_including_reflection", "o3_reflection"),
            _definition("time_reversal", "time_reversal"),
            _definition("batch_single_parity", "batch_single_parity"),
            _definition("checkpoint_resume", "checkpoint_resume"),
            _definition("cuda_float32", None, _HARDWARE_NOT_CERTIFIED),
            _definition("cuda_float64", None, _HARDWARE_NOT_CERTIFIED),
            _definition("cpu_cuda_energy_force_parity", None, _HARDWARE_NOT_CERTIFIED),
            _definition("large_system_complexity", None, _HARDWARE_NOT_CERTIFIED),
            _definition("multi_gpu_local_rank", None, _HARDWARE_NOT_CERTIFIED),
            _definition("lammps_serial", None, _HARDWARE_NOT_CERTIFIED),
            _definition("lammps_neighbor_restart", None, _HARDWARE_NOT_CERTIFIED),
            _definition("lammps_mpi_2", None, _HARDWARE_NOT_CERTIFIED),
            _definition("lammps_mpi_4", None, _HARDWARE_NOT_CERTIFIED),
            _definition("spin_lattice_nve_drift", None, _HARDWARE_NOT_CERTIFIED),
            _definition(
                "oxidation_calibration_external_split", None, _DATASET_NOT_CERTIFIED
            ),
            _definition("chgnet_fair_split_benchmark", None, _DATASET_NOT_CERTIFIED),
        )
    }
)


def canonical_gate_command(name: str) -> str:
    """Return the registry command recorded in artifacts and evidence."""

    try:
        return FIXED_GATE_REGISTRY[name].command
    except KeyError as exc:
        raise KeyError(f"unknown ZIVAR maturity gate: {name}") from exc


def _require_zynnova_environment() -> None:
    prefix_name = Path(sys.prefix).resolve().name
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if prefix_name != "zynnova" and conda_name != "zynnova":
        raise RuntimeError(
            "maturity gates must run in the zynnova Conda environment; use "
            "`conda run -n zynnova zivar run-gate ...`"
        )


def _torch() -> Any:
    import torch

    if torch.get_default_dtype() not in {torch.float32, torch.float64}:
        raise RuntimeError("unexpected PyTorch default floating dtype")
    return torch


def _reference_periodic_system(*, requires_grad: bool = False) -> tuple[Any, ...]:
    torch = _torch()
    positions = torch.tensor(
        [[0.2, 0.4, 0.7], [1.7, 1.1, 0.3], [2.3, 2.0, 1.4]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    charges = torch.tensor([0.7, -1.1, 0.4], dtype=torch.float64)
    cell = torch.tensor(
        [[5.0, 0.0, 0.0], [0.6, 5.5, 0.0], [0.2, 0.4, 6.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, charges, cell, pbc


def _run_unit_cpu(package_root: Path) -> dict[str, float | int]:
    repository_root = package_root.parents[3]
    selection = repository_root / "src" / "zynnova" / "ml" / "zivar" / "tests"
    arguments = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(selection),
        "-m",
        "not gpu and not lammps and not slow",
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    source_root = str(repository_root / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else source_root + os.pathsep + inherited
    )
    completed = subprocess.run(
        arguments,
        cwd=repository_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
        check=False,
        shell=False,
    )
    output = completed.stdout or ""

    def count(label: str) -> int:
        matches = re.findall(rf"(?:^|\s)(\d+)\s+{label}(?:\s|,|$)", output)
        return int(matches[-1]) if matches else 0

    passed = count("passed")
    failed = count("failed") + count("error") + count("errors")
    if completed.returncode != 0 and failed == 0:
        failed = 1
    return {
        "passed_tests": passed,
        "failed_tests": failed,
        "pytest_returncode": int(completed.returncode),
    }


def _scf_problem() -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    generator = torch.Generator().manual_seed(431)
    size = 12
    raw = torch.randn((size, size), generator=generator, dtype=torch.float64)
    hessian = raw @ raw.T / size + 1.5 * torch.eye(size, dtype=torch.float64)
    linear = torch.randn(size, generator=generator, dtype=torch.float64)
    constraint = torch.ones((1, size), dtype=torch.float64)
    target = torch.tensor([0.75], dtype=torch.float64)
    return hessian, linear, constraint, target


def _run_scf_stationarity(_: Path) -> dict[str, float | int]:
    torch = _torch()
    from .operators import CallableLinearOperator, DiagonalPreconditioner
    from .scf import SCFSolverConfig, solve_quadratic_scf

    hessian, linear, constraint, target = _scf_problem()
    operator = CallableLinearOperator(
        int(linear.numel()),
        lambda value: hessian @ value,
        dtype=torch.float64,
        device=linear.device,
    )
    result = solve_quadratic_scf(
        operator,
        linear,
        constraint=constraint,
        target=target,
        preconditioner=DiagonalPreconditioner(hessian.diag()),
        config=SCFSolverConfig(
            atol=1.0e-12,
            rtol=1.0e-11,
            constraint_atol=1.0e-12,
            max_iter=128,
        ),
    )
    stationarity = hessian @ result.solution + linear
    stationarity = stationarity + constraint.T @ result.lagrange_multipliers
    projected_rms = float(torch.linalg.vector_norm(stationarity) / math.sqrt(stationarity.numel()))
    charge_residual = float((constraint @ result.solution - target).abs().max())
    return {
        "projected_gradient_rms": projected_rms,
        "charge_residual": charge_residual,
        "energy_error_eV": result.report.energy_error,
        "iterations": result.report.iterations,
    }


def _run_qeq_dense_matrix_free(_: Path) -> dict[str, float | int]:
    torch = _torch()
    from .operators import CallableLinearOperator, DiagonalPreconditioner
    from .scf import SCFSolverConfig, solve_quadratic_scf

    hessian, linear, constraint, target = _scf_problem()
    operator = CallableLinearOperator(
        int(linear.numel()),
        lambda value: hessian @ value,
        dtype=hessian.dtype,
        device=hessian.device,
    )
    result = solve_quadratic_scf(
        operator,
        linear,
        constraint=constraint,
        target=target,
        preconditioner=DiagonalPreconditioner(hessian.diag()),
        config=SCFSolverConfig(
            atol=1.0e-12,
            rtol=1.0e-11,
            constraint_atol=1.0e-12,
            max_iter=128,
        ),
    )
    zeros = hessian.new_zeros((1, 1))
    kkt = torch.cat(
        (torch.cat((hessian, constraint.T), dim=1), torch.cat((constraint, zeros), dim=1)),
        dim=0,
    )
    dense = torch.linalg.solve(kkt, torch.cat((-linear, target)))[: linear.numel()]
    return {
        "max_charge_error": float((result.solution - dense).abs().max()),
        "charge_residual": float((constraint @ result.solution - target).abs().max()),
    }


def _run_direct_ewald_reference(_: Path) -> dict[str, float | int]:
    from .ewald_reference import ewald_energy, plan_ewald

    positions, charges, cell, pbc = _reference_periodic_system()
    candidate = ewald_energy(positions, charges, cell, pbc, plan_ewald(cell, 1.0e-9)).energy
    reference = ewald_energy(positions, charges, cell, pbc, plan_ewald(cell, 1.0e-11)).energy
    return {"energy_error_eV": float((candidate - reference).abs())}


def _run_pme_error_convergence(_: Path) -> dict[str, float | int]:
    torch = _torch()
    from .ewald_reference import ewald_energy, plan_ewald
    from .pme import plan_pme, pme_energy

    positions, charges, cell, pbc = _reference_periodic_system(requires_grad=True)
    reference = ewald_energy(
        positions, charges, cell, pbc, plan_ewald(cell, 1.0e-11)
    ).energy
    reference_force = -torch.autograd.grad(reference, positions, retain_graph=True)[0]
    plan = plan_pme(cell.detach(), 1.0e-7, interpolation_order=6, mesh_shape=(128, 128, 128))
    candidate = pme_energy(positions, charges, cell, pbc, plan).energy
    candidate_force = -torch.autograd.grad(candidate, positions)[0]
    return {
        "energy_error_eV": float((candidate.detach() - reference.detach()).abs()),
        "force_error_eV_per_A": float((candidate_force - reference_force).abs().max()),
    }


def _run_energy_force_fd(_: Path) -> dict[str, float | int]:
    torch = _torch()
    from .ewald_reference import isolated_coulomb_energy

    positions = torch.tensor(
        [[0.2, -0.3, 0.5], [1.4, 0.6, -0.2], [-0.7, 1.1, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    charges = torch.tensor([0.8, -1.1, 0.3], dtype=torch.float64)
    energy = isolated_coulomb_energy(positions, charges, (False, False, False))
    analytic = -torch.autograd.grad(energy, positions)[0]
    step = 1.0e-5
    numerical = torch.zeros_like(positions)
    base = positions.detach()
    for atom in range(base.shape[0]):
        for axis in range(3):
            offset = torch.zeros_like(base)
            offset[atom, axis] = step
            plus = isolated_coulomb_energy(base + offset, charges, (False, False, False))
            minus = isolated_coulomb_energy(base - offset, charges, (False, False, False))
            numerical[atom, axis] = -(plus - minus) / (2.0 * step)
    return {"max_error_eV_per_A": float((analytic - numerical).abs().max())}


def _run_stress_fd(_: Path) -> dict[str, float | int]:
    torch = _torch()
    from .ewald_reference import isolated_coulomb_energy

    base_positions = torch.tensor(
        [[0.2, 0.3, 0.5], [1.4, 0.6, 0.2], [0.7, 1.1, 0.9]], dtype=torch.float64
    )
    charges = torch.tensor([0.8, -1.1, 0.3], dtype=torch.float64)
    cell = 5.0 * torch.eye(3, dtype=torch.float64)
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)

    def strained_energy(value: Any) -> Any:
        deformation = torch.eye(3, dtype=torch.float64) + value
        return isolated_coulomb_energy(
            base_positions @ deformation.T, charges, (False, False, False)
        )

    analytic = torch.autograd.grad(strained_energy(strain), strain)[0] / torch.linalg.det(cell)
    numerical = torch.zeros_like(strain)
    step = 1.0e-5
    for row in range(3):
        for column in range(3):
            offset = torch.zeros_like(strain)
            offset[row, column] = step
            numerical[row, column] = (
                strained_energy(offset) - strained_energy(-offset)
            ) / (2.0 * step * torch.linalg.det(cell))
    return {"max_error_eV_per_A3": float((analytic - numerical).abs().max())}


def _spin_payload() -> dict[str, Any]:
    torch = _torch()
    return {
        "features": torch.randn(4, 8, dtype=torch.float64),
        "positions": torch.tensor(
            [[0.2, 0.4, 0.1], [1.4, 0.3, 0.7], [0.5, 1.7, 0.6], [0.8, 0.6, 2.0]],
            dtype=torch.float64,
        ),
        "batch": torch.zeros(4, dtype=torch.long),
        "edge_index": torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            ],
            dtype=torch.long,
        ),
        "shifts": torch.zeros(12, 3, dtype=torch.float64),
        "cutoff_A": 5.0,
        "conditions": {},
        "spin_vectors": torch.tensor(
            [[1.0, 0.2, 0.1], [-0.3, 0.8, 0.4], [0.1, -0.4, 1.1], [0.5, 0.6, -0.7]],
            dtype=torch.float64,
        ),
    }


def _spin_model() -> Any:
    torch = _torch()
    from .config import SpinConfig
    from .magnetism import SpinLatticeHamiltonian

    torch.manual_seed(20260817)
    model = SpinLatticeHamiltonian(8, SpinConfig(hidden=(24,))).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.08)
    return model


def _run_o3_reflection(_: Path) -> dict[str, float | int]:
    torch = _torch()
    model = _spin_model()
    payload = _spin_payload()
    original = model(**payload).energy
    transform = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    transformed = dict(payload)
    transformed["positions"] = payload["positions"] @ transform.T
    transformed["shifts"] = payload["shifts"] @ transform.T
    transformed["spin_vectors"] = (
        torch.linalg.det(transform) * payload["spin_vectors"] @ transform.T
    )
    rotated = model(**transformed).energy
    return {
        "max_equivariance_error": float((original - rotated).detach().abs().max())
    }


def _run_time_reversal(_: Path) -> dict[str, float | int]:
    model = _spin_model()
    payload = _spin_payload()
    forward = model(**payload).energy
    reversed_payload = dict(payload)
    reversed_payload["spin_vectors"] = -payload["spin_vectors"]
    reverse = model(**reversed_payload).energy
    return {"max_energy_error_eV": float((forward - reverse).detach().abs().max())}


def _run_spin_field_fd(_: Path) -> dict[str, float | int]:
    torch = _torch()
    model = _spin_model()
    payload = _spin_payload()
    spins = payload["spin_vectors"].detach().requires_grad_(True)
    payload["spin_vectors"] = spins
    energy = model(**payload).energy.sum()
    analytic = -torch.autograd.grad(energy, spins)[0]
    numerical = torch.zeros_like(spins)
    step = 1.0e-5
    base = spins.detach()
    for atom in range(base.shape[0]):
        for axis in range(3):
            offset = torch.zeros_like(base)
            offset[atom, axis] = step
            plus_payload = dict(payload)
            minus_payload = dict(payload)
            plus_payload["spin_vectors"] = base + offset
            minus_payload["spin_vectors"] = base - offset
            plus = model(**plus_payload).energy.sum()
            minus = model(**minus_payload).energy.sum()
            numerical[atom, axis] = -(plus - minus) / (2.0 * step)
    return {
        "max_error_eV_per_muB": float((analytic - numerical).detach().abs().max())
    }


def _complete_graph(atom_count: int, *, device: Any) -> Any:
    torch = _torch()
    pairs = [
        (source, target)
        for source in range(atom_count)
        for target in range(atom_count)
        if source != target
    ]
    return torch.tensor(pairs, device=device, dtype=torch.long).T.contiguous()


def _run_batch_single_parity(_: Path) -> dict[str, float | int]:
    """Measure complete-model graph batching parity in deterministic float64."""

    torch = _torch()
    from .config import ZIVARConfig
    from .model import build_zivar

    torch.manual_seed(417)
    config = ZIVARConfig.convolution(
        dft_level="batch-single-maturity-gate",
        backbone__atomic_numbers=(1, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=3,
        backbone__radial_mlp=(8,),
        backbone__pair_repulsion=False,
        electronic__hidden=(8,),
        electronic__radial_basis=3,
        electronic__oxidation__enabled=False,
        electrostatics__boundary="isolated",
        spin__mode="disabled",
        spin__require_spin_input=False,
    )
    model = build_zivar(config).double().eval()
    with torch.no_grad():
        local = next(
            layer
            for layer in reversed(tuple(model.variational.local.modules()))
            if isinstance(layer, torch.nn.Linear)
        )
        local.weight.zero_()
        local.bias.zero_()
        local.weight[1].copy_(torch.linspace(-0.12, 0.09, local.weight.shape[1]))
        local.bias[2:6] = torch.tensor([1.8, 1.5, 1.4, 1.6], dtype=torch.float64)

    def structure(positions: Any, numbers: Any) -> dict[str, Any]:
        edges = _complete_graph(int(positions.shape[0]), device=positions.device)
        return {
            "positions": positions,
            "atomic_numbers": numbers,
            "edge_index": edges,
            "shifts": positions.new_zeros((edges.shape[1], 3)),
            "pbc": torch.zeros((1, 3), dtype=torch.bool),
            "batch": torch.zeros(positions.shape[0], dtype=torch.long),
        }

    first = structure(
        torch.tensor(
            [[0.15, -0.20, 0.25], [1.30, 0.10, -0.35], [-0.40, 1.45, 0.30]],
            dtype=torch.float64,
        ),
        torch.tensor([1, 8, 1], dtype=torch.long),
    )
    second = structure(
        torch.tensor([[0.35, 0.10, -0.25], [1.10, -0.45, 0.55]], dtype=torch.float64),
        torch.tensor([8, 1], dtype=torch.long),
    )
    offset = int(first["positions"].shape[0])
    combined_data = {
        "positions": torch.cat((first["positions"], second["positions"])),
        "atomic_numbers": torch.cat((first["atomic_numbers"], second["atomic_numbers"])),
        "edge_index": torch.cat(
            (first["edge_index"], second["edge_index"] + offset), dim=1
        ),
        "shifts": torch.cat((first["shifts"], second["shifts"])),
        "pbc": torch.zeros((2, 3), dtype=torch.bool),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    }
    targets = torch.tensor([0.37, -0.24], dtype=torch.float64)
    combined = model.energy_forces_stress(
        combined_data,
        conditions={"total_charge": targets},
        compute_stress=False,
        compute_spin_fields=False,
    )
    singles = [
        model.energy_forces_stress(
            data,
            conditions={"total_charge": targets[index : index + 1]},
            compute_stress=False,
            compute_spin_fields=False,
        )
        for index, data in enumerate((first, second))
    ]
    errors = {
        "energy_error_eV": float(
            (combined["energy"] - torch.cat([item["energy"] for item in singles]))
            .detach()
            .abs()
            .max()
        ),
        "charge_error_e": float(
            (combined["charges"] - torch.cat([item["charges"] for item in singles]))
            .detach()
            .abs()
            .max()
        ),
        "force_error_eV_per_A": float(
            (combined["forces"] - torch.cat([item["forces"] for item in singles]))
            .detach()
            .abs()
            .max()
        ),
    }
    return {**errors, "max_error": max(errors.values())}


def _run_checkpoint_resume(package_root: Path) -> dict[str, float | int]:
    """Run the exact interrupted-next-step equality test through fixed pytest args."""

    repository_root = package_root.parents[3]
    test_path = (
        repository_root
        / "src"
        / "zynnova"
        / "ml"
        / "zivar"
        / "tests"
        / "test_checkpoint_resume.py"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_path),
            "-p",
            "no:cacheprovider",
        ),
        cwd=repository_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
        shell=False,
    )
    return {
        "next_step_parameter_error": 0.0 if completed.returncode == 0 else 1.0,
        "pytest_returncode": int(completed.returncode),
    }


_EXECUTORS: Mapping[str, Callable[[Path], dict[str, float | int]]] = MappingProxyType(
    {
        "unit_cpu": _run_unit_cpu,
        "scf_stationarity": _run_scf_stationarity,
        "qeq_dense_matrix_free": _run_qeq_dense_matrix_free,
        "direct_ewald_reference": _run_direct_ewald_reference,
        "pme_error_convergence": _run_pme_error_convergence,
        "energy_force_fd": _run_energy_force_fd,
        "stress_fd": _run_stress_fd,
        "spin_field_fd": _run_spin_field_fd,
        "o3_reflection": _run_o3_reflection,
        "time_reversal": _run_time_reversal,
        "batch_single_parity": _run_batch_single_parity,
        "checkpoint_resume": _run_checkpoint_resume,
    }
)


def _normalise_metrics(values: Mapping[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"gate metric {name!r} must be a number, not {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"gate metric {name!r} is non-finite")
        metrics[str(name)] = value
    return metrics


def _passes_thresholds(name: str, metrics: Mapping[str, float | int]) -> bool:
    from .maturity import GATE_THRESHOLDS

    for metric, operation, threshold in GATE_THRESHOLDS[name]:
        if metric not in metrics:
            return False
        value = float(metrics[metric])
        if operation == "max" and value > threshold:
            return False
        if operation == "min" and value < threshold:
            return False
    return True


def _write_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


def run_gate(
    name: str,
    artifact: str | Path,
    *,
    package_root: str | Path | None = None,
) -> GateRun:
    """Run one registered gate and atomically write a hashable JSON artifact.

    Unsupported target gates still produce a ``status=fail`` artifact with a
    concrete reason.  Numerical exceptions and threshold failures are likewise
    fail-closed; only an implemented executor satisfying the central threshold
    registry can produce ``status=pass``.
    """

    from .maturity import GATE_RESULT_SCHEMA, source_tree_hash

    try:
        definition = FIXED_GATE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown ZIVAR maturity gate: {name}") from exc
    root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).expanduser().resolve()
    )
    if not root.is_dir():
        raise ValueError(f"ZIVAR package root does not exist: {root}")
    source_hash = source_tree_hash(root)
    started = time.perf_counter()
    status = "fail"
    failure: str | None = None
    metrics: dict[str, float | int] = {}
    if not definition.implemented:
        failure = definition.unavailable_reason or _NOT_IMPLEMENTED
    else:
        try:
            _require_zynnova_environment()
            executor_name = definition.executor
            if executor_name is None:  # Narrowing for static type checkers.
                raise AssertionError("implemented gate has no executor")
            executor = _EXECUTORS[executor_name]
            metrics = _normalise_metrics(executor(root))
            if _passes_thresholds(name, metrics):
                status = "pass"
            else:
                failure = "measured metrics did not satisfy the registered thresholds"
        except Exception as exc:  # A failed gate must still leave auditable evidence.
            failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    payload = {
        "schema": GATE_RESULT_SCHEMA,
        "gate": name,
        "source_hash": source_hash,
        "command": definition.command,
        "status": status,
        "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed,
        "failure": failure,
    }
    artifact_path = Path(artifact).expanduser().resolve()
    digest = _write_artifact(artifact_path, payload)
    return GateRun(
        status=status,
        artifact=str(artifact_path),
        artifact_sha256=digest,
        gate=name,
        command=definition.command,
        metrics=metrics,
        failure=failure,
    )


__all__ = [
    "FIXED_GATE_REGISTRY",
    "GateDefinition",
    "GateRun",
    "canonical_gate_command",
    "run_gate",
]
