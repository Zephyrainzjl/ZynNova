"""Integrated atomistic-to-pack battery workflow.

This module intentionally orchestrates existing, injectable components instead
of hiding them behind a monolithic solver.  JouleWeave remains responsible for
low-scale labels and parameter surfaces; ZynSim consumes those surfaces through
``MultiscaleCoordinator`` and advances multiphysics, microstructure, inverse,
and digital-twin components on their own time scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..digital_twin import BatteryDigitalTwin, SensorPacket
from ..microstructure import MicrostructureEvolutionModel, validate_phase_labels
from ..multiphysics import (
    CoupledBatteryState,
    CoupledTrajectory,
    CouplingDiagnostics,
    FullyCoupledBatterySolver,
)
from ..multiscale import CrossScaleRecord, MultiscaleCoordinator


ParameterTarget = Any | Callable[[CoupledBatteryState], Any]
SensorSource = Callable[[CoupledBatteryState], SensorPacket | None]
EnergyDensitySource = Callable[[CoupledBatteryState], np.ndarray]
ControlSource = Callable[[CoupledBatteryState], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class ScaleCoupling:
    """Bind one property coordinator to a mutable continuum parameter target."""

    coordinator: MultiscaleCoordinator
    target: ParameterTarget
    name: str = "continuum"

    def resolve_target(self, state: CoupledBatteryState) -> Any:
        return self.target(state) if callable(self.target) else self.target


@dataclass(slots=True)
class FullScaleWorkflowConfig:
    maximum_time_step_s: float = 1.0
    microstructure_update_interval_s: float = 60.0
    propagate_microstructure_metrics: bool = True
    fail_on_nonfinite_state: bool = True

    def __post_init__(self) -> None:
        if self.maximum_time_step_s <= 0.0:
            raise ValueError("maximum_time_step_s must be positive")
        if self.microstructure_update_interval_s <= 0.0:
            raise ValueError("microstructure update interval must be positive")


@dataclass(frozen=True, slots=True)
class MicrostructureSnapshot:
    time_s: float
    phase_labels: np.ndarray
    phase_volume_fractions: Mapping[int, float]


@dataclass(slots=True)
class FullScaleWorkflowResult:
    trajectory: CoupledTrajectory
    property_records: Mapping[str, tuple[CrossScaleRecord, ...]]
    microstructure_snapshots: list[MicrostructureSnapshot] = field(default_factory=list)
    digital_twin_records: list[Any] = field(default_factory=list)


class FullScaleBatteryWorkflow:
    """Run a synchronized cross-scale, multiphysics, and digital-twin loop.

    The workflow is synchronous by design.  An uncertain property request can
    trigger JouleWeave active learning and DFT through an
    ``OnDemandAtomisticProvider`` before the same continuum step proceeds.
    This guarantees that no simulation step silently uses a pending background
    calculation or a property outside its uncertainty gate.
    """

    def __init__(
        self,
        solver: FullyCoupledBatterySolver,
        *,
        scale_couplings: Sequence[ScaleCoupling] = (),
        microstructure_labels: np.ndarray | None = None,
        microstructure_evolution: MicrostructureEvolutionModel | None = None,
        mechanical_energy_density: EnergyDensitySource | None = None,
        digital_twin: BatteryDigitalTwin | None = None,
        sensor_source: SensorSource | None = None,
        control_source: ControlSource | None = None,
        config: FullScaleWorkflowConfig | None = None,
    ) -> None:
        self.solver = solver
        self.scale_couplings = tuple(scale_couplings)
        self.microstructure_labels = (
            None
            if microstructure_labels is None
            else validate_phase_labels(microstructure_labels).copy()
        )
        self.microstructure_evolution = microstructure_evolution
        self.mechanical_energy_density = mechanical_energy_density
        self.digital_twin = digital_twin
        self.sensor_source = sensor_source
        self.control_source = control_source
        self.config = config or FullScaleWorkflowConfig()
        if self.microstructure_evolution is not None:
            if self.microstructure_labels is None:
                raise ValueError("microstructure evolution requires phase labels")
            if self.mechanical_energy_density is None:
                raise ValueError(
                    "microstructure evolution requires a mechanical-energy callback"
                )

    def run(
        self,
        initial_state: CoupledBatteryState,
        time_s: np.ndarray,
        current_A: np.ndarray,
    ) -> FullScaleWorkflowResult:
        times = np.asarray(time_s, dtype=float)
        currents = np.asarray(current_A, dtype=float)
        if times.ndim != 1 or currents.shape != times.shape or len(times) < 2:
            raise ValueError("time/current arrays must be aligned and one-dimensional")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time grid must be strictly increasing")

        state = initial_state.copy()
        states = [state.copy()]
        diagnostics: list[CouplingDiagnostics] = []
        snapshots: list[MicrostructureSnapshot] = []
        twin_start = 0 if self.digital_twin is None else len(self.digital_twin.records)
        last_microstructure_update = float(state.time_s)
        if self.microstructure_labels is not None:
            snapshots.append(self._snapshot(state.time_s))

        for left, right, current in zip(
            times[:-1], times[1:], currents[:-1], strict=True
        ):
            interval = float(right - left)
            substeps = max(
                1, int(np.ceil(interval / self.config.maximum_time_step_s))
            )
            dt_s = interval / substeps
            for _ in range(substeps):
                self._refresh_cross_scale_properties(state)
                state, diagnostic = self.solver.step(state, float(current), dt_s)
                diagnostics.append(diagnostic)
                self._validate_state(state)
                if (
                    self.microstructure_evolution is not None
                    and state.time_s - last_microstructure_update
                    >= self.config.microstructure_update_interval_s - 1.0e-12
                ):
                    self._advance_microstructure(
                        state,
                        elapsed_time_s=last_microstructure_update,
                        dt_s=state.time_s - last_microstructure_update,
                    )
                    last_microstructure_update = float(state.time_s)
                    snapshots.append(self._snapshot(state.time_s))
                self._advance_digital_twin(state, dt_s)
            states.append(state.copy())

        property_records = {
            coupling.name: tuple(coupling.coordinator.records)
            for coupling in self.scale_couplings
        }
        twin_records = (
            []
            if self.digital_twin is None
            else list(self.digital_twin.records[twin_start:])
        )
        return FullScaleWorkflowResult(
            trajectory=CoupledTrajectory(states, diagnostics),
            property_records=property_records,
            microstructure_snapshots=snapshots,
            digital_twin_records=twin_records,
        )

    def _refresh_cross_scale_properties(self, state: CoupledBatteryState) -> None:
        maximum_uncertainty = 0.0
        for coupling in self.scale_couplings:
            values = coupling.coordinator.update(
                coupling.resolve_target(state),
                float(state.soc),
                float(np.mean(state.temperature_K)),
            )
            relative = [
                value.relative_uncertainty()
                for value in values.values()
                if value.relative_uncertainty() is not None
            ]
            maximum_uncertainty = max(maximum_uncertainty, max(relative, default=0.0))
        state.metadata["parameter_relative_uncertainty"] = float(maximum_uncertainty)

    def _advance_microstructure(
        self,
        state: CoupledBatteryState,
        *,
        elapsed_time_s: float,
        dt_s: float,
    ) -> None:
        assert self.microstructure_evolution is not None
        assert self.microstructure_labels is not None
        assert self.mechanical_energy_density is not None
        energy = np.asarray(self.mechanical_energy_density(state), dtype=float)
        self.microstructure_labels = self.microstructure_evolution.step(
            self.microstructure_labels,
            mechanical_energy_density_J_m3=energy,
            elapsed_time_s=elapsed_time_s,
            dt_s=dt_s,
        )
        if self.config.propagate_microstructure_metrics:
            fractions = _phase_fractions(self.microstructure_labels)
            state.metadata["microstructure_phase_fractions"] = fractions
            state.metadata["crack_volume_fraction"] = float(
                fractions.get(9, 0.0)
            )

    def _advance_digital_twin(
        self, state: CoupledBatteryState, dt_s: float
    ) -> None:
        if self.digital_twin is None:
            return
        controls = (
            dict(self.control_source(state))
            if self.control_source is not None
            else {
                "current_A": float(state.current_A),
                "cooling_command": 0.0,
            }
        )
        packet = None if self.sensor_source is None else self.sensor_source(state)
        self.digital_twin.advance(controls, dt_s, sensor_packet=packet)

    def _snapshot(self, time_s: float) -> MicrostructureSnapshot:
        assert self.microstructure_labels is not None
        labels = self.microstructure_labels.copy()
        return MicrostructureSnapshot(
            time_s=float(time_s),
            phase_labels=labels,
            phase_volume_fractions=_phase_fractions(labels),
        )

    def _validate_state(self, state: CoupledBatteryState) -> None:
        if not self.config.fail_on_nonfinite_state:
            return
        arrays = (
            state.temperature_K,
            state.displacement_m,
            state.stress_Pa,
            state.damage,
            state.sei_thickness_m,
            state.cei_thickness_m,
            state.plated_lithium_mol,
            state.active_material_fraction,
        )
        if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
            raise FloatingPointError("full-scale workflow produced a non-finite field")
        scalars = (
            state.time_s,
            state.current_A,
            state.terminal_voltage_V,
            state.soc,
            state.lithium_inventory_fraction,
        )
        if not np.isfinite(np.asarray(scalars, dtype=float)).all():
            raise FloatingPointError("full-scale workflow produced a non-finite scalar")


def stress_energy_density(state: CoupledBatteryState, young_modulus_Pa: float) -> np.ndarray:
    """Estimate elastic energy density from Voigt stress for voxel evolution."""

    if young_modulus_Pa <= 0.0:
        raise ValueError("young_modulus_Pa must be positive")
    stress = np.asarray(state.stress_Pa, dtype=float)
    energy = 0.5 * np.sum(stress * stress, axis=-1) / young_modulus_Pa
    if energy.size == 1 and state.metadata.get("microstructure_shape") is not None:
        shape = tuple(int(value) for value in state.metadata["microstructure_shape"])
        return np.full(shape, float(energy.reshape(-1)[0]), dtype=float)
    return energy


def _phase_fractions(labels: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    total = float(labels.size)
    return {int(value): float(count / total) for value, count in zip(values, counts)}


__all__ = [
    "FullScaleBatteryWorkflow",
    "FullScaleWorkflowConfig",
    "FullScaleWorkflowResult",
    "MicrostructureSnapshot",
    "ScaleCoupling",
    "stress_energy_density",
]
