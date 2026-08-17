"""Dynamic switching between DFN/P2D and microstructure-resolved solvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class FidelityLevel(str, Enum):
    DFN = "dfn"
    MICROSTRUCTURE = "microstructure"


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    c_rate: float
    temperature_gradient_K: float
    maximum_damage: float
    parameter_relative_uncertainty: float
    voltage_residual_V: float = 0.0


@dataclass(slots=True)
class AdaptiveFidelityPolicy:
    enter_microstructure_c_rate: float = 2.0
    exit_microstructure_c_rate: float = 1.0
    enter_temperature_gradient_K: float = 3.0
    exit_temperature_gradient_K: float = 1.0
    enter_damage: float = 0.05
    exit_damage: float = 0.02
    enter_uncertainty: float = 0.15
    exit_uncertainty: float = 0.08
    minimum_microstructure_steps: int = 5

    def choose(
        self,
        current: FidelityLevel,
        metrics: FidelityMetrics,
        microstructure_steps: int,
    ) -> FidelityLevel:
        if current == FidelityLevel.DFN:
            if (
                metrics.c_rate >= self.enter_microstructure_c_rate
                or metrics.temperature_gradient_K >= self.enter_temperature_gradient_K
                or metrics.maximum_damage >= self.enter_damage
                or metrics.parameter_relative_uncertainty >= self.enter_uncertainty
            ):
                return FidelityLevel.MICROSTRUCTURE
            return current
        if microstructure_steps < self.minimum_microstructure_steps:
            return current
        if (
            metrics.c_rate <= self.exit_microstructure_c_rate
            and metrics.temperature_gradient_K <= self.exit_temperature_gradient_K
            and metrics.maximum_damage <= self.exit_damage
            and metrics.parameter_relative_uncertainty <= self.exit_uncertainty
        ):
            return FidelityLevel.DFN
        return current


StateProjector = Callable[[Any], Any]


class AdaptiveFidelityModel:
    """Model facade exposing a stable initialize/step contract while switching."""

    def __init__(
        self,
        dfn_model: Any,
        microstructure_model: Any,
        *,
        policy: AdaptiveFidelityPolicy | None = None,
        dfn_to_microstructure: StateProjector | None = None,
        microstructure_to_dfn: StateProjector | None = None,
        metrics_provider: Callable[[Any, float], FidelityMetrics] | None = None,
    ) -> None:
        self.dfn_model = dfn_model
        self.microstructure_model = microstructure_model
        self.policy = policy or AdaptiveFidelityPolicy()
        self.dfn_to_microstructure = dfn_to_microstructure
        self.microstructure_to_dfn = microstructure_to_dfn
        self.metrics_provider = metrics_provider or self._default_metrics
        self.level = FidelityLevel.DFN
        self.microstructure_steps = 0

    def initialize(self, soc: float = 1.0, *, temperature_K: float | None = None) -> Any:
        self.level = FidelityLevel.DFN
        self.microstructure_steps = 0
        return self.dfn_model.initialize(soc, temperature_K=temperature_K)

    def step(self, state: Any, current_A: float, dt_s: float) -> Any:
        metrics = self.metrics_provider(state, current_A)
        desired = self.policy.choose(self.level, metrics, self.microstructure_steps)
        if desired != self.level:
            state = self._project(state, self.level, desired)
            self.level = desired
            self.microstructure_steps = 0
        model = self.dfn_model if self.level == FidelityLevel.DFN else self.microstructure_model
        updated = model.step(state, current_A, dt_s)
        if self.level == FidelityLevel.MICROSTRUCTURE:
            self.microstructure_steps += 1
        metadata = getattr(updated, "metadata", None)
        if isinstance(metadata, dict):
            metadata["fidelity_level"] = self.level.value
            metadata["fidelity_metrics"] = metrics
        return updated

    def _project(self, state: Any, source: FidelityLevel, target: FidelityLevel) -> Any:
        projector = (
            self.dfn_to_microstructure
            if source == FidelityLevel.DFN
            else self.microstructure_to_dfn
        )
        if projector is None:
            raise RuntimeError(
                f"state projector is required for {source.value}->{target.value} switching"
            )
        return projector(state)

    @staticmethod
    def _default_metrics(state: Any, current_A: float) -> FidelityMetrics:
        metadata = getattr(state, "metadata", {})
        capacity_Ah = float(metadata.get("capacity_Ah", 1.0)) if isinstance(metadata, dict) else 1.0
        temperature = np.asarray(getattr(state, "temperature_K", 298.15), dtype=float)
        damage = np.asarray(getattr(state, "damage", 0.0), dtype=float)
        uncertainty = float(metadata.get("parameter_relative_uncertainty", 0.0)) if isinstance(metadata, dict) else 0.0
        return FidelityMetrics(
            c_rate=abs(float(current_A)) / max(capacity_Ah, 1.0e-12),
            temperature_gradient_K=float(np.ptp(temperature)),
            maximum_damage=float(np.max(damage)),
            parameter_relative_uncertainty=uncertainty,
        )


__all__ = [
    "AdaptiveFidelityModel",
    "AdaptiveFidelityPolicy",
    "FidelityLevel",
    "FidelityMetrics",
]
