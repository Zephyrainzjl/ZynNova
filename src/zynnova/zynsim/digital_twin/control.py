"""Risk-aware closed-loop charging and thermal-management control."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ControlAction:
    current_A: float
    cooling_W: float


@dataclass(frozen=True, slots=True)
class ControlDecision:
    action: ControlAction
    objective: float
    feasible: bool
    predicted_metrics: Mapping[str, float]


Predictor = Callable[[np.ndarray, ControlAction, float], Mapping[str, float]]


@dataclass(slots=True)
class ClosedLoopController:
    current_candidates_A: Sequence[float]
    cooling_candidates_W: Sequence[float]
    horizon_s: float = 60.0
    target_soc: float = 1.0
    maximum_temperature_K: float = 323.15
    maximum_voltage_V: float = 4.2
    minimum_voltage_V: float = 2.7
    maximum_risk_probability: float = 0.05
    current_slew_weight: float = 0.01
    cooling_weight: float = 0.001
    degradation_weight: float = 10.0

    def decide(
        self,
        state: np.ndarray,
        predictor: Predictor,
        *,
        previous_action: ControlAction | None = None,
    ) -> ControlDecision:
        decisions: list[ControlDecision] = []
        for current, cooling in product(self.current_candidates_A, self.cooling_candidates_W):
            action = ControlAction(float(current), float(cooling))
            metrics = dict(predictor(np.asarray(state, dtype=float), action, self.horizon_s))
            feasible = (
                metrics.get("maximum_temperature_K", 0.0) <= self.maximum_temperature_K
                and metrics.get("maximum_voltage_V", 0.0) <= self.maximum_voltage_V
                and metrics.get("minimum_voltage_V", self.minimum_voltage_V) >= self.minimum_voltage_V
                and metrics.get("risk_probability", 0.0) <= self.maximum_risk_probability
            )
            soc_error = self.target_soc - metrics.get("soc", 0.0)
            objective = soc_error**2
            objective += self.cooling_weight * action.cooling_W**2
            objective += self.degradation_weight * metrics.get("degradation_increment", 0.0)
            if previous_action is not None:
                objective += self.current_slew_weight * (
                    action.current_A - previous_action.current_A
                ) ** 2
            if not feasible:
                objective += 1.0e6
            decisions.append(ControlDecision(action, float(objective), feasible, metrics))
        if not decisions:
            raise ValueError("controller requires current and cooling candidates")
        return min(decisions, key=lambda decision: decision.objective)


__all__ = ["ClosedLoopController", "ControlAction", "ControlDecision"]
