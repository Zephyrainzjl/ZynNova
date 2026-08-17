"""State and diagnostics for electrochemical multiphysics coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class CoupledBatteryState:
    time_s: float
    electrochemical: Any
    temperature_K: np.ndarray
    displacement_m: np.ndarray
    stress_Pa: np.ndarray
    damage: np.ndarray
    sei_thickness_m: np.ndarray
    cei_thickness_m: np.ndarray
    plated_lithium_mol: np.ndarray
    active_material_fraction: np.ndarray
    lithium_inventory_fraction: float = 1.0
    current_A: float = 0.0
    terminal_voltage_V: float = 0.0
    soc: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def copy(self) -> "CoupledBatteryState":
        return CoupledBatteryState(
            time_s=float(self.time_s),
            electrochemical=_copy_state(self.electrochemical),
            temperature_K=np.asarray(self.temperature_K, dtype=float).copy(),
            displacement_m=np.asarray(self.displacement_m, dtype=float).copy(),
            stress_Pa=np.asarray(self.stress_Pa, dtype=float).copy(),
            damage=np.asarray(self.damage, dtype=float).copy(),
            sei_thickness_m=np.asarray(self.sei_thickness_m, dtype=float).copy(),
            cei_thickness_m=np.asarray(self.cei_thickness_m, dtype=float).copy(),
            plated_lithium_mol=np.asarray(self.plated_lithium_mol, dtype=float).copy(),
            active_material_fraction=np.asarray(
                self.active_material_fraction, dtype=float
            ).copy(),
            lithium_inventory_fraction=float(self.lithium_inventory_fraction),
            current_A=float(self.current_A),
            terminal_voltage_V=float(self.terminal_voltage_V),
            soc=float(self.soc),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class CouplingDiagnostics:
    converged: bool
    iterations: int
    residual_norm: float
    electrochemical_residual: float
    thermal_residual: float
    mechanical_residual: float
    damage_residual: float
    aging_residual: float
    model_fidelity: str
    messages: tuple[str, ...] = ()


@dataclass(slots=True)
class CoupledTrajectory:
    states: list[CoupledBatteryState]
    diagnostics: list[CouplingDiagnostics]

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray([state.time_s for state in self.states], dtype=float)

    @property
    def voltage_V(self) -> np.ndarray:
        return np.asarray([state.terminal_voltage_V for state in self.states], dtype=float)

    @property
    def temperature_K(self) -> np.ndarray:
        return np.asarray(
            [float(np.mean(state.temperature_K)) for state in self.states], dtype=float
        )

    @property
    def damage(self) -> np.ndarray:
        return np.asarray(
            [float(np.max(state.damage)) for state in self.states], dtype=float
        )


def infer_state_observables(
    electrochemical_state: Any,
    *,
    parameters: Any | None = None,
) -> Mapping[str, float]:
    voltage = float(getattr(electrochemical_state, "terminal_voltage_V", 0.0))
    current = float(getattr(electrochemical_state, "current_A", 0.0))
    time_s = float(getattr(electrochemical_state, "time_s", 0.0))
    soc = _infer_soc(electrochemical_state, parameters)
    return {"voltage_V": voltage, "current_A": current, "time_s": time_s, "soc": soc}


def _infer_soc(state: Any, parameters: Any | None = None) -> float:
    for name in ("soc", "state_of_charge"):
        if hasattr(state, name):
            value = getattr(state, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    if parameters is None:
                        continue
                    value = value(parameters)
            try:
                return float(np.clip(value, 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    metadata = getattr(state, "metadata", {})
    if isinstance(metadata, Mapping) and "soc" in metadata:
        return float(np.clip(metadata["soc"], 0.0, 1.0))
    return 0.5


def _copy_state(state: Any) -> Any:
    if hasattr(state, "copy"):
        return state.copy()
    import copy

    return copy.deepcopy(state)


__all__ = [
    "CoupledBatteryState",
    "CoupledTrajectory",
    "CouplingDiagnostics",
    "infer_state_observables",
]
