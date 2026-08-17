"""Operator-split P2D aging with SEI, plating, and lithium-inventory feedback."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import FARADAY
from ..exceptions import ConvergenceError
from .degradation import DegradationModel, DegradationState
from .experiments import CyclingProtocol
from .p2d import CurrentSegment, P2DModel, P2DState


@dataclass(frozen=True, slots=True)
class AgingConfig:
    """Controls the operator-split coupling between P2D and side reactions."""

    degradation_time_scale: float = 1.0
    lithium_inventory_feedback: bool = True
    film_resistance_feedback: bool = True
    minimum_capacity_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.degradation_time_scale)
            or self.degradation_time_scale < 0.0
        ):
            raise ValueError("degradation_time_scale cannot be negative")
        if (
            not np.isfinite(self.minimum_capacity_fraction)
            or not 0.0 < self.minimum_capacity_fraction <= 1.0
        ):
            raise ValueError("minimum_capacity_fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class AgingStepDiagnostics:
    lost_lithium_increment_Ah: float
    cumulative_lost_lithium_Ah: float
    mean_sei_thickness_m: float
    mean_film_resistance_ohm_m2: float
    maximum_plated_lithium_mol_m2: float
    film_voltage_drop_V: float
    capacity_retention: float


@dataclass(slots=True)
class AgingState:
    electrochemical: P2DState
    degradation: DegradationState
    nominal_capacity_Ah: float
    available_capacity_Ah: float
    cumulative_throughput_Ah: float = 0.0
    equivalent_full_cycles: float = 0.0
    diagnostics: AgingStepDiagnostics | None = None

    def copy(self) -> AgingState:
        degradation = self.degradation
        return AgingState(
            electrochemical=self.electrochemical.copy(),
            degradation=DegradationState(
                sei_thickness_m=degradation.sei_thickness_m.copy(),
                plated_lithium_mol_m2=degradation.plated_lithium_mol_m2.copy(),
                lost_lithium_C_m2=degradation.lost_lithium_C_m2.copy(),
                sei_current_A_m2=degradation.sei_current_A_m2.copy(),
                plating_current_A_m2=degradation.plating_current_A_m2.copy(),
            ),
            nominal_capacity_Ah=float(self.nominal_capacity_Ah),
            available_capacity_Ah=float(self.available_capacity_Ah),
            cumulative_throughput_Ah=float(self.cumulative_throughput_Ah),
            equivalent_full_cycles=float(self.equivalent_full_cycles),
            diagnostics=self.diagnostics,
        )

    @property
    def capacity_retention(self) -> float:
        return float(self.available_capacity_Ah / self.nominal_capacity_Ah)


@dataclass(slots=True)
class AgingTrajectory:
    states: list[AgingState]
    segment_labels: list[str]
    cycle_index: np.ndarray
    termination_reason: str = "completed"

    def __post_init__(self) -> None:
        self.cycle_index = np.asarray(self.cycle_index, dtype=np.int32)
        if not self.states or len(self.segment_labels) != len(self.states):
            raise ValueError("aging trajectory states and labels must be non-empty and aligned")
        if self.cycle_index.shape != (len(self.states),):
            raise ValueError("aging cycle_index must align with states")

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray([state.electrochemical.time_s for state in self.states])

    @property
    def voltage_V(self) -> np.ndarray:
        return np.asarray(
            [state.electrochemical.terminal_voltage_V for state in self.states]
        )

    @property
    def current_A(self) -> np.ndarray:
        return np.asarray([state.electrochemical.current_A for state in self.states])

    @property
    def soc(self) -> np.ndarray:
        return np.asarray(
            [
                state.electrochemical.metadata.get("aging_soc", np.nan)
                for state in self.states
            ],
            dtype=float,
        )

    @property
    def capacity_retention(self) -> np.ndarray:
        return np.asarray([state.capacity_retention for state in self.states])


@dataclass(frozen=True, slots=True)
class AgingCycleSummary:
    cycle: int
    discharge_capacity_Ah: float
    charge_capacity_Ah: float
    coulombic_efficiency: float
    capacity_retention: float
    lost_lithium_Ah: float
    mean_sei_thickness_m: float
    maximum_plated_lithium_mol_m2: float
    final_soc: float


@dataclass(slots=True)
class AgingCyclingResult:
    trajectory: AgingTrajectory
    summaries: tuple[AgingCycleSummary, ...]


class AgingP2DModel:
    """Couple an existing P2D model to mechanistic side-reaction states.

    The coupling is conservative with respect to the explicitly removed
    cyclable lithium.  It is operator split: the main P2D step is solved first,
    followed by SEI/plating, lithium-inventory removal, and film-voltage
    feedback.  This makes the approximation explicit and testable without
    claiming a monolithic side-reaction Newton solve.
    """

    def __init__(
        self,
        electrochemical_model: P2DModel,
        *,
        degradation_model: DegradationModel | None = None,
        config: AgingConfig | None = None,
    ) -> None:
        self.electrochemical_model = electrochemical_model
        self.degradation_model = degradation_model or DegradationModel()
        self.config = config or AgingConfig()

    @property
    def parameters(self):
        return self.electrochemical_model.parameters

    def initialize(
        self,
        soc: float = 1.0,
        *,
        temperature_K: float | None = None,
    ) -> AgingState:
        electrochemical = self.electrochemical_model.initialize(
            soc,
            temperature_K=temperature_K,
        )
        electrochemical.metadata["aging_soc"] = float(soc)
        nominal = self.parameters.theoretical_capacity_Ah()
        return AgingState(
            electrochemical=electrochemical,
            degradation=DegradationState.initialize(
                self.parameters.discretization.negative_cells,
                self.degradation_model.sei,
            ),
            nominal_capacity_Ah=nominal,
            available_capacity_Ah=nominal,
        )

    def step(
        self,
        state: AgingState,
        current_A: float,
        dt_s: float,
    ) -> AgingState:
        if dt_s <= 0.0:
            raise ValueError("aging time step must be positive")
        electrochemical = self.electrochemical_model.step(
            state.electrochemical,
            current_A,
            dt_s,
        )
        grid = self.electrochemical_model.grid
        negative_electrolyte = electrochemical.electrolyte_potential_V[grid.negative]
        effective_dt = dt_s * self.config.degradation_time_scale
        if effective_dt > 0.0:
            degradation = self.degradation_model.step(
                state.degradation,
                electrochemical.negative_solid_potential_V,
                negative_electrolyte,
                electrochemical.temperature_K,
                effective_dt,
            )
        else:
            degradation = state.copy().degradation

        lost_increment_C_m2 = (
            degradation.lost_lithium_C_m2 - state.degradation.lost_lithium_C_m2
        )
        if np.any(lost_increment_C_m2 < -1.0e-14):
            raise ConvergenceError("side-reaction lithium loss became negative")
        lost_increment_Ah = self._lost_lithium_Ah(lost_increment_C_m2)
        cumulative_lost_Ah = self._lost_lithium_Ah(
            degradation.lost_lithium_C_m2
        )

        if self.config.lithium_inventory_feedback and np.any(lost_increment_C_m2 > 0.0):
            active = self.parameters.negative.active_volume_fraction
            surface = self.parameters.negative.specific_surface_area_m2_m3
            concentration_loss = (
                lost_increment_C_m2 * surface / (FARADAY * active)
            )
            corrected = (
                electrochemical.negative_particle_concentration_mol_m3
                - concentration_loss[:, None]
            )
            if np.min(corrected) <= 0.0:
                raise ConvergenceError(
                    "aging side reactions exhausted negative-electrode cyclable lithium"
                )
            electrochemical.negative_particle_concentration_mol_m3 = corrected

        film_resistance = degradation.film_resistance_ohm_m2(
            self.degradation_model.sei
        )
        film_drop = 0.0
        if self.config.film_resistance_feedback:
            film_drop = float(
                np.mean(
                    electrochemical.negative_interfacial_current_A_m2
                    * film_resistance
                )
            )
            electrochemical.terminal_voltage_V -= film_drop

        minimum_capacity = (
            self.config.minimum_capacity_fraction * state.nominal_capacity_Ah
        )
        available = max(state.nominal_capacity_Ah - cumulative_lost_Ah, minimum_capacity)
        throughput = state.cumulative_throughput_Ah + abs(current_A) * dt_s / 3600.0
        equivalent_cycles = throughput / (2.0 * state.nominal_capacity_Ah)
        electrochemical.metadata.update(
            {
                "aging_soc": electrochemical.soc(self.parameters),
                "available_capacity_Ah": available,
                "capacity_retention": available / state.nominal_capacity_Ah,
                "equivalent_full_cycles": equivalent_cycles,
                "operator_split_aging": True,
            }
        )
        electrochemical.validate(self.parameters)
        diagnostics = AgingStepDiagnostics(
            lost_lithium_increment_Ah=lost_increment_Ah,
            cumulative_lost_lithium_Ah=cumulative_lost_Ah,
            mean_sei_thickness_m=float(np.mean(degradation.sei_thickness_m)),
            mean_film_resistance_ohm_m2=float(np.mean(film_resistance)),
            maximum_plated_lithium_mol_m2=float(
                np.max(degradation.plated_lithium_mol_m2)
            ),
            film_voltage_drop_V=film_drop,
            capacity_retention=available / state.nominal_capacity_Ah,
        )
        return AgingState(
            electrochemical=electrochemical,
            degradation=degradation,
            nominal_capacity_Ah=state.nominal_capacity_Ah,
            available_capacity_Ah=available,
            cumulative_throughput_Ah=throughput,
            equivalent_full_cycles=equivalent_cycles,
            diagnostics=diagnostics,
        )

    def run_cycles(
        self,
        protocol: CyclingProtocol,
        *,
        initial_state: AgingState | None = None,
        initial_soc: float = 1.0,
    ) -> AgingCyclingResult:
        state = self.initialize(initial_soc) if initial_state is None else initial_state.copy()
        states = [state.copy()]
        labels = ["initial"]
        cycle_ids = [1]
        summaries: list[AgingCycleSummary] = []

        for cycle in range(1, protocol.cycles + 1):
            cycle_start = len(states) - 1
            for segment in _aging_cycle_segments(protocol, cycle):
                local_time = 0.0
                while local_time < segment.duration_s - 1.0e-14:
                    dt_s = min(segment.time_step_s, segment.duration_s - local_time)
                    applied = segment.current(local_time, state.electrochemical)
                    state = self.step(state, applied, dt_s)
                    states.append(state.copy())
                    labels.append(segment.label)
                    cycle_ids.append(cycle)
                    local_time += dt_s
                    voltage = state.electrochemical.terminal_voltage_V
                    if (
                        segment.minimum_voltage_V is not None
                        and voltage <= segment.minimum_voltage_V
                    ):
                        break
                    if (
                        segment.maximum_voltage_V is not None
                        and voltage >= segment.maximum_voltage_V
                    ):
                        break

            selected = states[cycle_start:]
            times = np.asarray([item.electrochemical.time_s for item in selected])
            dt = np.diff(times)
            currents = np.asarray(
                [item.electrochemical.current_A for item in selected[1:]]
            )
            discharge = float(np.sum(np.maximum(currents, 0.0) * dt) / 3600.0)
            charge = float(np.sum(np.maximum(-currents, 0.0) * dt) / 3600.0)
            summaries.append(
                AgingCycleSummary(
                    cycle=cycle,
                    discharge_capacity_Ah=discharge,
                    charge_capacity_Ah=charge,
                    coulombic_efficiency=(
                        discharge / charge if charge > 0.0 else float("nan")
                    ),
                    capacity_retention=state.capacity_retention,
                    lost_lithium_Ah=(
                        state.diagnostics.cumulative_lost_lithium_Ah
                        if state.diagnostics is not None
                        else 0.0
                    ),
                    mean_sei_thickness_m=float(
                        np.mean(state.degradation.sei_thickness_m)
                    ),
                    maximum_plated_lithium_mol_m2=float(
                        np.max(state.degradation.plated_lithium_mol_m2)
                    ),
                    final_soc=state.electrochemical.soc(self.parameters),
                )
            )

        return AgingCyclingResult(
            trajectory=AgingTrajectory(
                states=states,
                segment_labels=labels,
                cycle_index=np.asarray(cycle_ids),
                termination_reason="completed",
            ),
            summaries=tuple(summaries),
        )

    def _lost_lithium_Ah(self, lost_C_m2: np.ndarray) -> float:
        negative = self.parameters.negative
        active_interface_area_m2 = (
            negative.specific_surface_area_m2_m3
            * self.parameters.area_m2
            * negative.thickness_m
        )
        return float(np.mean(lost_C_m2) * active_interface_area_m2 / 3600.0)


def _aging_cycle_segments(
    protocol: CyclingProtocol,
    cycle: int,
) -> tuple[CurrentSegment, ...]:
    segments: list[CurrentSegment] = [
        CurrentSegment(
            protocol.discharge_duration_s,
            protocol.discharge_current_A,
            protocol.time_step_s,
            minimum_voltage_V=protocol.minimum_voltage_V,
            label=f"cycle-{cycle}-discharge",
        )
    ]
    if protocol.rest_after_discharge_s > 0.0:
        segments.append(
            CurrentSegment(
                protocol.rest_after_discharge_s,
                0.0,
                protocol.time_step_s,
                label=f"cycle-{cycle}-rest-after-discharge",
            )
        )
    segments.append(
        CurrentSegment(
            protocol.charge_duration_s,
            -protocol.charge_current_A,
            protocol.time_step_s,
            maximum_voltage_V=protocol.maximum_voltage_V,
            label=f"cycle-{cycle}-charge",
        )
    )
    if protocol.rest_after_charge_s > 0.0:
        segments.append(
            CurrentSegment(
                protocol.rest_after_charge_s,
                0.0,
                protocol.time_step_s,
                label=f"cycle-{cycle}-rest-after-charge",
            )
        )
    return tuple(segments)


__all__ = [
    "AgingConfig",
    "AgingCycleSummary",
    "AgingCyclingResult",
    "AgingP2DModel",
    "AgingState",
    "AgingStepDiagnostics",
    "AgingTrajectory",
]
