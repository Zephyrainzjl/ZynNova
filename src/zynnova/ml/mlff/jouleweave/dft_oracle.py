"""DFT-oracle contracts and uncertainty accounting for active learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class DFTReferenceResult:
    energy_eV: float
    forces_eV_A: np.ndarray
    stress_eV_A3: np.ndarray | None = None
    fermi_level_eV: float | None = None
    charges_e: np.ndarray | None = None
    electron_count: float | None = None
    converged: bool = True
    source: str = "DFT"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forces = np.asarray(self.forces_eV_A, dtype=float)
        if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
            raise ValueError("forces_eV_A must have finite shape (n_atoms, 3)")
        object.__setattr__(self, "forces_eV_A", forces)
        if not np.isfinite(self.energy_eV):
            raise ValueError("DFT energy must be finite")
        if self.stress_eV_A3 is not None:
            stress = np.asarray(self.stress_eV_A3, dtype=float)
            if stress.shape not in {(3, 3), (6,)} or not np.isfinite(stress).all():
                raise ValueError("stress must be finite with shape (3,3) or (6,)")
            object.__setattr__(self, "stress_eV_A3", stress)
        if self.charges_e is not None:
            charges = np.asarray(self.charges_e, dtype=float).reshape(-1)
            if not np.isfinite(charges).all() or len(charges) != len(forces):
                raise ValueError("charges_e must contain one finite value per atom")
            object.__setattr__(self, "charges_e", charges)


class DFTOracle(Protocol):
    def evaluate(
        self,
        structure: Any,
        *,
        electrode_potential_V: float | None = None,
        electron_count: float | None = None,
    ) -> DFTReferenceResult: ...


@dataclass(slots=True)
class CallableDFTOracle:
    evaluator: Callable[..., DFTReferenceResult]
    source: str = "callable-dft"

    def evaluate(
        self,
        structure: Any,
        *,
        electrode_potential_V: float | None = None,
        electron_count: float | None = None,
    ) -> DFTReferenceResult:
        result = self.evaluator(
            structure,
            electrode_potential_V=electrode_potential_V,
            electron_count=electron_count,
        )
        if not isinstance(result, DFTReferenceResult):
            raise TypeError("DFT evaluator must return DFTReferenceResult")
        return result


@dataclass(frozen=True, slots=True)
class DFTUncertaintyAssessment:
    consensus: DFTReferenceResult
    energy_standard_uncertainty_eV: float
    force_standard_uncertainty_eV_A: float
    fermi_standard_uncertainty_eV: float | None
    numerical_disagreement: Mapping[str, float]
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(slots=True)
class RedundantDFTOracle:
    """Run multiple DFT settings/functionals and expose reference disagreement."""

    oracles: Sequence[DFTOracle]
    maximum_energy_disagreement_eV_per_atom: float = 0.05
    maximum_force_disagreement_eV_A: float = 0.20
    maximum_fermi_disagreement_eV: float = 0.10

    def __post_init__(self) -> None:
        if len(self.oracles) < 2:
            raise ValueError("RedundantDFTOracle requires at least two oracles")

    def assess(
        self,
        structure: Any,
        *,
        electrode_potential_V: float | None = None,
        electron_count: float | None = None,
    ) -> DFTUncertaintyAssessment:
        results = [
            oracle.evaluate(
                structure,
                electrode_potential_V=electrode_potential_V,
                electron_count=electron_count,
            )
            for oracle in self.oracles
        ]
        converged = [result for result in results if result.converged]
        if not converged:
            raise RuntimeError("all DFT replicas failed to converge")
        atom_count = len(converged[0].forces_eV_A)
        if any(len(result.forces_eV_A) != atom_count for result in converged):
            raise ValueError("DFT replicas returned different atom counts")
        energies = np.asarray([result.energy_eV for result in converged])
        forces = np.stack([result.forces_eV_A for result in converged], axis=0)
        energy_std = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
        force_std = float(np.sqrt(np.mean(np.var(forces, axis=0, ddof=1)))) if len(forces) > 1 else 0.0
        fermi_values = np.asarray(
            [result.fermi_level_eV for result in converged if result.fermi_level_eV is not None],
            dtype=float,
        )
        fermi_std = (
            float(np.std(fermi_values, ddof=1)) if len(fermi_values) > 1 else (0.0 if len(fermi_values) == 1 else None)
        )
        energy_per_atom = energy_std / max(atom_count, 1)
        reasons: list[str] = []
        if energy_per_atom > self.maximum_energy_disagreement_eV_per_atom:
            reasons.append("energy disagreement")
        if force_std > self.maximum_force_disagreement_eV_A:
            reasons.append("force disagreement")
        if fermi_std is not None and fermi_std > self.maximum_fermi_disagreement_eV:
            reasons.append("Fermi-level disagreement")

        mean_stress = None
        stresses = [result.stress_eV_A3 for result in converged if result.stress_eV_A3 is not None]
        if len(stresses) == len(converged):
            mean_stress = np.mean(np.stack(stresses), axis=0)
        mean_charges = None
        charges = [result.charges_e for result in converged if result.charges_e is not None]
        if len(charges) == len(converged):
            mean_charges = np.mean(np.stack(charges), axis=0)
        consensus = DFTReferenceResult(
            energy_eV=float(np.mean(energies)),
            forces_eV_A=np.mean(forces, axis=0),
            stress_eV_A3=mean_stress,
            fermi_level_eV=float(np.mean(fermi_values)) if len(fermi_values) else None,
            charges_e=mean_charges,
            electron_count=converged[0].electron_count,
            converged=True,
            source="ensemble:" + ",".join(result.source for result in converged),
            metadata={"replicas": len(converged)},
        )
        return DFTUncertaintyAssessment(
            consensus=consensus,
            energy_standard_uncertainty_eV=energy_std,
            force_standard_uncertainty_eV_A=force_std,
            fermi_standard_uncertainty_eV=fermi_std,
            numerical_disagreement={
                "energy_eV_per_atom": energy_per_atom,
                "force_eV_A": force_std,
                "fermi_eV": 0.0 if fermi_std is None else fermi_std,
            },
            accepted=not reasons,
            reasons=tuple(reasons),
        )

    def evaluate(
        self,
        structure: Any,
        *,
        electrode_potential_V: float | None = None,
        electron_count: float | None = None,
    ) -> DFTReferenceResult:
        assessment = self.assess(
            structure,
            electrode_potential_V=electrode_potential_V,
            electron_count=electron_count,
        )
        if not assessment.accepted:
            raise RuntimeError(
                "DFT reference failed uncertainty gates: " + ", ".join(assessment.reasons)
            )
        return assessment.consensus


__all__ = [
    "CallableDFTOracle",
    "DFTOracle",
    "DFTReferenceResult",
    "DFTUncertaintyAssessment",
    "RedundantDFTOracle",
]
