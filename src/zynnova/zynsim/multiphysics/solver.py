"""Partitioned electrochemical-thermal-mechanical-damage-aging solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from ..exceptions import ConvergenceError
from .constitutive import CoupledConstitutiveModel
from .damage import DamageModel
from .state import (
    CoupledBatteryState,
    CoupledTrajectory,
    CouplingDiagnostics,
    infer_state_observables,
)


class ElectrochemicalModel(Protocol):
    def initialize(self, soc: float = 1.0, *, temperature_K: float | None = None) -> Any: ...

    def step(self, state: Any, current_A: float, dt_s: float) -> Any: ...


OpenCircuitVoltage = Callable[[float, float], float]


@dataclass(slots=True)
class CoupledSolverConfig:
    maximum_coupling_iterations: int = 12
    relative_tolerance: float = 1.0e-5
    relaxation: float = 0.65
    field_size: int = 1
    active_volume_m3: float = 1.0e-5
    fail_on_nonconvergence: bool = True
    model_fidelity: str = "auto"

    def __post_init__(self) -> None:
        if self.maximum_coupling_iterations < 1 or self.relative_tolerance <= 0.0:
            raise ValueError("coupled solver iteration controls are invalid")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("coupling relaxation must lie in (0,1]")
        if self.field_size < 1 or self.active_volume_m3 <= 0.0:
            raise ValueError("coupled field size and volume must be positive")


class FullyCoupledBatterySolver:
    """Strong fixed-point coupling across five battery-physics domains.

    The electrochemical model can be the existing P2D model, the existing 3-D
    porous-electrode model, or an adaptive fidelity wrapper exposing the same
    ``initialize``/``step`` contract.
    """

    def __init__(
        self,
        electrochemical_model: ElectrochemicalModel,
        *,
        constitutive: CoupledConstitutiveModel | None = None,
        damage: DamageModel | None = None,
        open_circuit_voltage: OpenCircuitVoltage | None = None,
        thermal_laplacian: np.ndarray | None = None,
        damage_laplacian: np.ndarray | None = None,
        config: CoupledSolverConfig | None = None,
    ) -> None:
        self.electrochemical_model = electrochemical_model
        self.constitutive = constitutive or CoupledConstitutiveModel()
        self.damage_model = damage or DamageModel()
        self.open_circuit_voltage = open_circuit_voltage
        self.thermal_laplacian = thermal_laplacian
        self.damage_laplacian = damage_laplacian
        self.config = config or CoupledSolverConfig()

    def initialize(
        self,
        soc: float = 1.0,
        *,
        temperature_K: float = 298.15,
    ) -> CoupledBatteryState:
        electrochemical = self.electrochemical_model.initialize(
            soc, temperature_K=temperature_K
        )
        observables = infer_state_observables(
            electrochemical,
            parameters=getattr(self.electrochemical_model, "parameters", None),
        )
        shape = (self.config.field_size,)
        return CoupledBatteryState(
            time_s=observables["time_s"],
            electrochemical=electrochemical,
            temperature_K=np.full(shape, temperature_K, dtype=float),
            displacement_m=np.zeros(shape + (3,), dtype=float),
            stress_Pa=np.zeros(shape + (6,), dtype=float),
            damage=np.zeros(shape, dtype=float),
            sei_thickness_m=np.zeros(shape, dtype=float),
            cei_thickness_m=np.zeros(shape, dtype=float),
            plated_lithium_mol=np.zeros(shape, dtype=float),
            active_material_fraction=np.ones(shape, dtype=float),
            lithium_inventory_fraction=1.0,
            current_A=0.0,
            terminal_voltage_V=observables["voltage_V"],
            soc=float(soc),
        )

    def step(
        self,
        state: CoupledBatteryState,
        current_A: float,
        dt_s: float,
    ) -> tuple[CoupledBatteryState, CouplingDiagnostics]:
        if dt_s <= 0.0 or not np.isfinite(current_A):
            raise ValueError("coupled step requires positive dt and finite current")
        previous = state.copy()
        iterate = state.copy()
        residuals = np.full(5, np.inf)
        messages: list[str] = []
        for iteration in range(1, self.config.maximum_coupling_iterations + 1):
            electrochemical_seed = _copy_electrochemical(previous.electrochemical)
            _inject_temperature(electrochemical_seed, iterate.temperature_K)
            electrochemical = self.electrochemical_model.step(
                electrochemical_seed, current_A, dt_s
            )
            obs = infer_state_observables(
                electrochemical,
                parameters=getattr(self.electrochemical_model, "parameters", None),
            )
            soc = float(np.clip(obs["soc"], 0.0, 1.0))
            ocv = (
                None
                if self.open_circuit_voltage is None
                else float(self.open_circuit_voltage(soc, float(np.mean(iterate.temperature_K))))
            )
            heat = self.constitutive.heat_source_W_m3(
                current_A=current_A,
                terminal_voltage_V=obs["voltage_V"],
                open_circuit_voltage_V=ocv,
                temperature_K=iterate.temperature_K,
                active_volume_m3=self.config.active_volume_m3,
            )
            temperature = self.constitutive.thermal_step(
                previous.temperature_K,
                heat,
                dt_s,
                laplacian=self.thermal_laplacian,
            )
            stress, displacement = self.constitutive.mechanical_response(
                soc,
                iterate.damage,
            )
            damage = self.damage_model.step(
                previous.damage,
                stress,
                dt_s,
                laplacian=self.damage_laplacian,
            )
            anode_overpotential = float(
                getattr(electrochemical, "anode_overpotential_V", 0.0)
            )
            sei, cei, plated, active, inventory = self.constitutive.aging_step(
                sei_thickness_m=previous.sei_thickness_m,
                cei_thickness_m=previous.cei_thickness_m,
                plated_lithium_mol=previous.plated_lithium_mol,
                active_material_fraction=previous.active_material_fraction,
                lithium_inventory_fraction=previous.lithium_inventory_fraction,
                time_s=previous.time_s,
                dt_s=dt_s,
                temperature_K=temperature,
                anode_overpotential_V=anode_overpotential,
                damage=damage,
            )
            candidate = CoupledBatteryState(
                time_s=previous.time_s + dt_s,
                electrochemical=electrochemical,
                temperature_K=_relax(iterate.temperature_K, temperature, self.config.relaxation),
                displacement_m=_relax(iterate.displacement_m, displacement, self.config.relaxation),
                stress_Pa=_relax(iterate.stress_Pa, stress, self.config.relaxation),
                damage=_relax(iterate.damage, damage, self.config.relaxation),
                sei_thickness_m=_relax(iterate.sei_thickness_m, sei, self.config.relaxation),
                cei_thickness_m=_relax(iterate.cei_thickness_m, cei, self.config.relaxation),
                plated_lithium_mol=_relax(iterate.plated_lithium_mol, plated, self.config.relaxation),
                active_material_fraction=_relax(iterate.active_material_fraction, active, self.config.relaxation),
                lithium_inventory_fraction=float(
                    (1.0 - self.config.relaxation) * iterate.lithium_inventory_fraction
                    + self.config.relaxation * inventory
                ),
                current_A=float(current_A),
                terminal_voltage_V=float(obs["voltage_V"]),
                soc=soc,
                metadata=dict(iterate.metadata),
            )
            residuals = np.asarray(
                [
                    _relative(candidate.terminal_voltage_V, iterate.terminal_voltage_V),
                    _array_relative(candidate.temperature_K, iterate.temperature_K),
                    _array_relative(candidate.stress_Pa, iterate.stress_Pa),
                    _array_relative(candidate.damage, iterate.damage),
                    max(
                        _array_relative(candidate.sei_thickness_m, iterate.sei_thickness_m),
                        _relative(candidate.lithium_inventory_fraction, iterate.lithium_inventory_fraction),
                    ),
                ]
            )
            iterate = candidate
            if float(np.max(residuals)) <= self.config.relative_tolerance:
                diagnostics = CouplingDiagnostics(
                    True,
                    iteration,
                    float(np.max(residuals)),
                    float(residuals[0]),
                    float(residuals[1]),
                    float(residuals[2]),
                    float(residuals[3]),
                    float(residuals[4]),
                    self.config.model_fidelity,
                    tuple(messages),
                )
                iterate.metadata["coupling_diagnostics"] = diagnostics
                return iterate, diagnostics
        messages.append("maximum coupling iterations reached")
        diagnostics = CouplingDiagnostics(
            False,
            self.config.maximum_coupling_iterations,
            float(np.max(residuals)),
            *map(float, residuals),
            self.config.model_fidelity,
            tuple(messages),
        )
        if self.config.fail_on_nonconvergence:
            raise ConvergenceError(
                f"multiphysics coupling failed with residual {diagnostics.residual_norm:.3e}"
            )
        iterate.metadata["coupling_diagnostics"] = diagnostics
        return iterate, diagnostics

    def run(
        self,
        initial_state: CoupledBatteryState,
        time_s: np.ndarray,
        current_A: np.ndarray,
        *,
        maximum_time_step_s: float,
    ) -> CoupledTrajectory:
        times = np.asarray(time_s, dtype=float)
        currents = np.asarray(current_A, dtype=float)
        if times.ndim != 1 or currents.shape != times.shape or len(times) < 2:
            raise ValueError("coupled drive cycle arrays are invalid")
        if np.any(np.diff(times) <= 0.0) or maximum_time_step_s <= 0.0:
            raise ValueError("coupled drive cycle time grid is invalid")
        state = initial_state.copy()
        states = [state.copy()]
        diagnostics: list[CouplingDiagnostics] = []
        for left, right, current in zip(times[:-1], times[1:], currents[:-1], strict=True):
            interval = right - left
            substeps = max(1, int(np.ceil(interval / maximum_time_step_s)))
            dt = interval / substeps
            for _ in range(substeps):
                state, diag = self.step(state, float(current), float(dt))
                diagnostics.append(diag)
            states.append(state.copy())
        return CoupledTrajectory(states, diagnostics)


def _inject_temperature(state: Any, temperature_K: np.ndarray) -> None:
    if hasattr(state, "temperature_K"):
        target = getattr(state, "temperature_K")
        value = float(np.mean(temperature_K)) if np.asarray(target).ndim == 0 else np.broadcast_to(np.mean(temperature_K), np.asarray(target).shape).copy()
        try:
            setattr(state, "temperature_K", value)
        except (AttributeError, TypeError):
            pass


def _copy_electrochemical(state: Any) -> Any:
    if hasattr(state, "copy"):
        return state.copy()
    import copy

    return copy.deepcopy(state)


def _relax(old: np.ndarray, new: np.ndarray, factor: float) -> np.ndarray:
    return (1.0 - factor) * np.asarray(old, dtype=float) + factor * np.asarray(new, dtype=float)


def _relative(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), abs(old), 1.0)


def _array_relative(new: np.ndarray, old: np.ndarray) -> float:
    difference = float(np.linalg.norm(np.asarray(new) - np.asarray(old)))
    scale = max(float(np.linalg.norm(new)), float(np.linalg.norm(old)), 1.0)
    return difference / scale


__all__ = ["CoupledSolverConfig", "ElectrochemicalModel", "FullyCoupledBatterySolver"]
