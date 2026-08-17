"""Fair, same-structure accuracy comparison against a released CHGNet model.

This module deliberately reports measurements instead of embedding a marketing
claim. A ZIVAR configuration is certified only when every requested metric is
available on the same immutable set of labelled ASE structures and its MAE is
strictly lower than CHGNet's MAE.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AccuracyMetrics:
    structures: int
    atoms: int
    energy_mae_eV_per_atom: float | None
    force_mae_eV_per_A: float | None
    stress_mae_GPa: float | None
    magmom_mae_muB: float | None
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CHGNetParityResult:
    dataset_fingerprint: str
    split_fingerprint: str
    zivar: AccuracyMetrics
    chgnet: AccuracyMetrics
    requested_metrics: tuple[str, ...]
    certified_accuracy_win: bool
    metric_winners: dict[str, str]
    zivar_parameters: int
    chgnet_parameters: int
    parameter_ratio: float
    parameter_budget_matched: bool
    zivar_architecture: str
    chgnet_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Any) -> None:
        from pathlib import Path

        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _first(mapping: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _reference(structure: Any, metric: str) -> Any | None:
    calculator_results = getattr(getattr(structure, "calc", None), "results", {})
    if metric == "energy":
        value = _first(structure.info, ("energy", "REF_energy", "dft_energy"))
        return calculator_results.get("energy") if value is None else value
    if metric == "forces":
        value = _first(structure.arrays, ("forces", "REF_forces", "dft_forces"))
        return calculator_results.get("forces") if value is None else value
    if metric == "stress":
        value = _first(structure.info, ("stress", "REF_stress", "dft_stress"))
        return calculator_results.get("stress") if value is None else value
    if metric == "magmoms":
        value = _first(
            structure.arrays,
            ("magmoms", "magmom", "magnetic_moments", "REF_magmoms"),
        )
        return calculator_results.get("magmoms") if value is None else value
    raise KeyError(metric)


def _fingerprint(structures: list[Any], metrics: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metrics).encode())
    for atoms in structures:
        digest.update(np.asarray(atoms.numbers, dtype="<i4").tobytes())
        digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
        digest.update(np.asarray(atoms.cell.array, dtype="<f8").tobytes())
        digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
        for metric in metrics:
            reference = _reference(atoms, metric)
            if reference is None:
                digest.update(b"missing")
            else:
                digest.update(np.asarray(reference, dtype="<f8").tobytes())
    return digest.hexdigest()


def _stress6(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (6,):
        return array
    if array.shape == (3, 3):
        symmetric = 0.5 * (array + array.T)
        return symmetric[(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]
    raise ValueError("stress labels must have shape [6] or [3,3]")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _evaluate(
    structures: list[Any], calculator: Any, requested: tuple[str, ...]
) -> AccuracyMetrics:
    energy_error: list[np.ndarray] = []
    force_error: list[np.ndarray] = []
    stress_error: list[np.ndarray] = []
    magmom_error: list[np.ndarray] = []
    atoms_total = 0
    started = time.perf_counter()
    for reference_atoms in structures:
        atoms = reference_atoms.copy()
        atoms.calc = calculator
        atoms_total += len(atoms)
        if "energy" in requested:
            reference = _reference(reference_atoms, "energy")
            if reference is None:
                raise ValueError("every benchmark structure requires an energy label")
            prediction = atoms.get_potential_energy()
            if not np.isfinite(float(reference)) or not np.isfinite(float(prediction)):
                raise FloatingPointError("non-finite energy in parity benchmark")
            energy_error.append(
                np.asarray([abs(float(prediction) - float(reference)) / len(atoms)])
            )
        if "forces" in requested:
            reference = _reference(reference_atoms, "forces")
            if reference is None:
                raise ValueError("every benchmark structure requires force labels")
            predicted_force = _finite_array(atoms.get_forces(), (len(atoms), 3), "force prediction")
            reference_force = _finite_array(reference, (len(atoms), 3), "force label")
            force_error.append(np.abs(predicted_force - reference_force))
        if "stress" in requested:
            reference = _reference(reference_atoms, "stress")
            if reference is None:
                raise ValueError("every benchmark structure requires stress labels")
            # Both ASE calculators return eV/A^3. Report in GPa.
            predicted_stress = _finite_array(
                atoms.get_stress(voigt=True), (6,), "stress prediction"
            )
            reference_stress = _finite_array(_stress6(reference), (6,), "stress label")
            stress_error.append(np.abs(predicted_stress - reference_stress) * 160.21766208)
        if "magmoms" in requested:
            reference = _reference(reference_atoms, "magmoms")
            if reference is None:
                raise ValueError("every benchmark structure requires magmom labels")
            predicted_moment = _finite_array(
                atoms.get_magnetic_moments(), (len(atoms),), "magmom prediction"
            )
            reference_moment = _finite_array(reference, (len(atoms),), "magmom label")
            magmom_error.append(np.abs(predicted_moment - reference_moment))
    elapsed = time.perf_counter() - started

    def mean(values: list[np.ndarray]) -> float | None:
        if not values:
            return None
        return float(np.concatenate([value.reshape(-1) for value in values]).mean())

    return AccuracyMetrics(
        structures=len(structures),
        atoms=atoms_total,
        energy_mae_eV_per_atom=mean(energy_error),
        force_mae_eV_per_A=mean(force_error),
        stress_mae_GPa=mean(stress_error),
        magmom_mae_muB=mean(magmom_error),
        wall_time_s=elapsed,
    )


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def compare_with_chgnet(
    structures: Iterable[Any],
    zivar_potential: Any,
    *,
    chgnet_model: Any | None = None,
    device: str = "cpu",
    dtype: str | None = None,
    requested_metrics: tuple[str, ...] = ("energy", "forces", "stress", "magmoms"),
    split_fingerprint: str,
    parameter_tolerance: float = 0.10,
    minimum_structures: int = 100,
) -> CHGNetParityResult:
    """Evaluate ZIVAR and CHGNet on the exact same labelled ASE structures."""

    allowed = {"energy", "forces", "stress", "magmoms"}
    if not requested_metrics or not set(requested_metrics).issubset(allowed):
        raise ValueError(f"requested_metrics must be a nonempty subset of {allowed}")
    samples = list(structures)
    if len(samples) < minimum_structures or minimum_structures < 1:
        raise ValueError(
            f"parity evaluation requires at least {minimum_structures} structures"
        )
    if len(split_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in split_fingerprint.lower()
    ):
        raise ValueError("split_fingerprint must be a 64-character SHA-256 digest")
    if not 0 <= parameter_tolerance <= 1:
        raise ValueError("parameter_tolerance must lie in [0,1]")
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    from .calculator import zivar_calculator

    baseline = CHGNet.load() if chgnet_model is None else chgnet_model
    zivar = zivar_calculator(zivar_potential, device=device, dtype=dtype)
    if "magmoms" in requested_metrics and zivar.model.config.spin.mode != "magnitude_auxiliary":
        raise ValueError(
            "CHGNet scalar-moment parity requires ZIVARConfig.chgnet_compatible(); "
            "it cannot certify the non-collinear spin-lattice model"
        )
    chgnet = CHGNetCalculator(model=baseline, use_device=device)
    zivar_metrics = _evaluate(samples, zivar, requested_metrics)
    chgnet_metrics = _evaluate(samples, chgnet, requested_metrics)
    field = {
        "energy": "energy_mae_eV_per_atom",
        "forces": "force_mae_eV_per_A",
        "stress": "stress_mae_GPa",
        "magmoms": "magmom_mae_muB",
    }
    winners: dict[str, str] = {}
    for metric in requested_metrics:
        zivar_value = getattr(zivar_metrics, field[metric])
        chgnet_value = getattr(chgnet_metrics, field[metric])
        if zivar_value is None or chgnet_value is None:
            raise RuntimeError(f"metric {metric!r} was requested but not evaluated")
        winners[metric] = (
            "zivar" if zivar_value < chgnet_value else
            "chgnet" if chgnet_value < zivar_value else "tie"
        )
    zivar_parameters = _parameter_count(zivar.model)
    chgnet_parameters = _parameter_count(baseline)
    parameter_ratio = zivar_parameters / max(1, chgnet_parameters)
    budget_matched = abs(parameter_ratio - 1.0) <= parameter_tolerance
    from .config import ARCHITECTURE_REVISION

    return CHGNetParityResult(
        dataset_fingerprint=_fingerprint(samples, requested_metrics),
        split_fingerprint=split_fingerprint.lower(),
        zivar=zivar_metrics,
        chgnet=chgnet_metrics,
        requested_metrics=requested_metrics,
        certified_accuracy_win=(
            budget_matched and all(value == "zivar" for value in winners.values())
        ),
        metric_winners=winners,
        zivar_parameters=zivar_parameters,
        chgnet_parameters=chgnet_parameters,
        parameter_ratio=parameter_ratio,
        parameter_budget_matched=budget_matched,
        zivar_architecture=ARCHITECTURE_REVISION,
        chgnet_version=importlib.metadata.version("chgnet"),
    )


__all__ = [
    "AccuracyMetrics",
    "CHGNetParityResult",
    "compare_with_chgnet",
]
