"""Probabilistic thermal, voltage, plating, and fracture risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    maximum_temperature_K: float = 333.15
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.3
    maximum_damage: float = 0.3
    maximum_plated_lithium_mol: float = 1.0e-3


@dataclass(frozen=True, slots=True)
class RiskForecast:
    probabilities: Mapping[str, float]
    joint_failure_probability: float
    expected_time_to_first_violation_s: float | None
    sample_count: int


TrajectorySimulator = Callable[[np.ndarray, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]


class ProbabilisticRiskModel:
    def __init__(self, thresholds: RiskThresholds | None = None) -> None:
        self.thresholds = thresholds or RiskThresholds()

    def forecast(
        self,
        state_ensemble: np.ndarray,
        controls: Mapping[str, np.ndarray],
        simulator: TrajectorySimulator,
        *,
        time_s: np.ndarray | None = None,
    ) -> RiskForecast:
        ensemble = np.asarray(state_ensemble, dtype=float)
        if ensemble.ndim != 2 or len(ensemble) < 2:
            raise ValueError("risk forecast requires a state ensemble")
        events: dict[str, list[bool]] = {
            "over_temperature": [],
            "under_voltage": [],
            "over_voltage": [],
            "fracture": [],
            "lithium_plating": [],
        }
        first_times: list[float] = []
        for member in ensemble:
            trajectory = simulator(member, controls)
            temperature = np.asarray(trajectory.get("temperature_K", []), dtype=float)
            voltage = np.asarray(trajectory.get("voltage_V", []), dtype=float)
            damage = np.asarray(trajectory.get("damage", []), dtype=float)
            plating = np.asarray(trajectory.get("plated_lithium_mol", []), dtype=float)
            flags = {
                "over_temperature": bool(np.any(temperature > self.thresholds.maximum_temperature_K)),
                "under_voltage": bool(np.any(voltage < self.thresholds.minimum_voltage_V)),
                "over_voltage": bool(np.any(voltage > self.thresholds.maximum_voltage_V)),
                "fracture": bool(np.any(damage > self.thresholds.maximum_damage)),
                "lithium_plating": bool(np.any(plating > self.thresholds.maximum_plated_lithium_mol)),
            }
            for name, value in flags.items():
                events[name].append(value)
            if time_s is not None and any(flags.values()):
                masks = []
                if temperature.size:
                    masks.append(temperature > self.thresholds.maximum_temperature_K)
                if voltage.size:
                    masks.append((voltage < self.thresholds.minimum_voltage_V) | (voltage > self.thresholds.maximum_voltage_V))
                if damage.size:
                    masks.append(damage > self.thresholds.maximum_damage)
                if plating.size:
                    masks.append(plating > self.thresholds.maximum_plated_lithium_mol)
                combined = np.logical_or.reduce(masks)
                indices = np.flatnonzero(combined)
                if len(indices):
                    first_times.append(float(np.asarray(time_s)[indices[0]]))
        probabilities = {
            name: float(np.mean(values)) for name, values in events.items()
        }
        joint = float(
            np.mean(
                np.logical_or.reduce(
                    [np.asarray(values, dtype=bool) for values in events.values()]
                )
            )
        )
        return RiskForecast(
            probabilities=probabilities,
            joint_failure_probability=joint,
            expected_time_to_first_violation_s=(
                float(np.mean(first_times)) if first_times else None
            ),
            sample_count=len(ensemble),
        )


__all__ = [
    "ProbabilisticRiskModel",
    "RiskForecast",
    "RiskThresholds",
]
