"""Committee and calibrated uncertainty estimates for JouleWeave potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ...common import require_torch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class CommitteePrediction:
    energy_mean_eV: Any
    energy_standard_uncertainty_eV: Any
    forces_mean_eV_A: Any | None
    forces_standard_uncertainty_eV_A: Any | None
    maximum_atomic_force_uncertainty_eV_A: Any | None
    member_outputs: tuple[Mapping[str, Any], ...]

    def scalar_score(self) -> float:
        energy = float(torch.max(self.energy_standard_uncertainty_eV).detach().cpu().item())
        if self.maximum_atomic_force_uncertainty_eV_A is None:
            return energy
        force = float(
            torch.max(self.maximum_atomic_force_uncertainty_eV_A).detach().cpu().item()
        )
        return max(energy, force)


class JouleWeaveCommittee:
    """Ensemble predictor with graph-energy and atom-force disagreement."""

    def __init__(self, models: Sequence[Any]) -> None:
        if len(models) < 2:
            raise ValueError("a committee requires at least two models")
        self.models = tuple(models)

    def predict(
        self,
        inputs: Mapping[str, Any],
        *,
        compute_forces: bool = True,
        create_graph: bool = False,
    ) -> CommitteePrediction:
        outputs: list[Mapping[str, Any]] = []
        for model in self.models:
            if compute_forces:
                if hasattr(model, "grand_potential_and_forces"):
                    output = model.grand_potential_and_forces(
                        inputs, create_graph=create_graph
                    )
                elif hasattr(model, "energy_and_forces"):
                    output = model.energy_and_forces(inputs, create_graph=create_graph)
                else:
                    raise TypeError("committee model cannot compute forces")
            else:
                output = model(inputs)
            outputs.append(output)
        energy_key = "grand_potential" if "grand_potential" in outputs[0] else "energy"
        energies = torch.stack([output[energy_key] for output in outputs], dim=0)
        energy_mean = torch.mean(energies, dim=0)
        energy_std = torch.std(energies, dim=0, unbiased=True)
        if not compute_forces:
            return CommitteePrediction(
                energy_mean,
                energy_std,
                None,
                None,
                None,
                tuple(outputs),
            )
        forces = torch.stack([output["forces"] for output in outputs], dim=0)
        force_mean = torch.mean(forces, dim=0)
        force_std = torch.std(forces, dim=0, unbiased=True)
        atomic_norm = torch.linalg.vector_norm(force_std, dim=-1)
        return CommitteePrediction(
            energy_mean,
            energy_std,
            force_mean,
            force_std,
            atomic_norm,
            tuple(outputs),
        )


@dataclass(slots=True)
class ConformalUncertaintyCalibrator:
    """Scale model uncertainties to an empirical target coverage."""

    coverage: float = 0.95
    scale_: float | None = None

    def __post_init__(self) -> None:
        if not 0.5 < self.coverage < 1.0:
            raise ValueError("coverage must lie in (0.5, 1)")

    def fit(self, residuals: Iterable[float], predicted_sigma: Iterable[float]) -> float:
        residual = np.abs(np.asarray(tuple(residuals), dtype=float))
        sigma = np.asarray(tuple(predicted_sigma), dtype=float)
        if residual.shape != sigma.shape or residual.size == 0:
            raise ValueError("residual and sigma arrays must be non-empty and aligned")
        if np.any(sigma <= 0.0) or not np.isfinite(residual).all() or not np.isfinite(sigma).all():
            raise ValueError("calibration values must be finite and sigma positive")
        scores = residual / sigma
        rank = int(np.ceil((len(scores) + 1) * self.coverage)) - 1
        rank = int(np.clip(rank, 0, len(scores) - 1))
        self.scale_ = float(np.partition(scores, rank)[rank])
        return self.scale_

    def calibrate(self, sigma: Any) -> Any:
        if self.scale_ is None:
            raise RuntimeError("uncertainty calibrator must be fitted first")
        return sigma * self.scale_


__all__ = [
    "CommitteePrediction",
    "ConformalUncertaintyCalibrator",
    "JouleWeaveCommittee",
]
