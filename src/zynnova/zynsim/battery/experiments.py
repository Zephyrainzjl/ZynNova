"""Advanced battery protocols, cycle accounting, and CC-CV control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .p2d import CurrentSegment, P2DModel, P2DState, P2DTrajectory


@dataclass(frozen=True, slots=True)
class CyclingProtocol:
    """Repeated constant-current discharge/rest/charge/rest cycling."""

    cycles: int
    discharge_current_A: float
    charge_current_A: float
    discharge_duration_s: float
    charge_duration_s: float
    time_step_s: float
    rest_after_discharge_s: float = 0.0
    rest_after_charge_s: float = 0.0
    minimum_voltage_V: float | None = None
    maximum_voltage_V: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cycles, bool) or not isinstance(
            self.cycles, (int, np.integer)
        ):
            raise TypeError("cycles must be an integer")
        positive = np.asarray(
            (
                self.cycles,
                self.discharge_current_A,
                self.charge_current_A,
                self.discharge_duration_s,
                self.charge_duration_s,
                self.time_step_s,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("cycle counts, currents, durations, and time step must be positive")
        rests = np.asarray(
            (self.rest_after_discharge_s, self.rest_after_charge_s),
            dtype=float,
        )
        if np.any(~np.isfinite(rests)) or np.any(rests < 0.0):
            raise ValueError("rest durations must be finite and non-negative")
        limits = (
            value
            for value in (self.minimum_voltage_V, self.maximum_voltage_V)
            if value is not None
        )
        if any(not np.isfinite(value) for value in limits):
            raise ValueError("cycle voltage limits must be finite")


@dataclass(frozen=True, slots=True)
class CycleSummary:
    cycle: int
    discharge_capacity_Ah: float
    charge_capacity_Ah: float
    discharge_energy_Wh: float
    charge_energy_Wh: float
    coulombic_efficiency: float
    energy_efficiency: float
    minimum_voltage_V: float
    maximum_voltage_V: float
    final_soc: float
    final_temperature_K: float
    capacity_retention: float
    termination_reasons: tuple[str, ...]


@dataclass(slots=True)
class CyclingResult:
    trajectory: P2DTrajectory
    cycle_index: np.ndarray
    summaries: tuple[CycleSummary, ...]

    def __post_init__(self) -> None:
        self.cycle_index = np.asarray(self.cycle_index, dtype=np.int32)
        if self.cycle_index.shape != (len(self.trajectory.states),):
            raise ValueError("cycle_index must align with the trajectory")


@dataclass(frozen=True, slots=True)
class CCCVConfig:
    """Constant-current/constant-voltage charging controls."""

    charge_current_A: float
    voltage_limit_V: float
    taper_current_A: float
    time_step_s: float
    maximum_cc_duration_s: float
    maximum_cv_duration_s: float
    voltage_tolerance_V: float = 1.0e-5
    current_tolerance_A: float = 1.0e-8

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.charge_current_A,
                self.voltage_limit_V,
                self.taper_current_A,
                self.time_step_s,
                self.maximum_cc_duration_s,
                self.maximum_cv_duration_s,
                self.voltage_tolerance_V,
                self.current_tolerance_A,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("CC-CV configuration values must be positive")
        if self.taper_current_A >= self.charge_current_A:
            raise ValueError("taper current must be below the CC charge current")


@dataclass(frozen=True, slots=True)
class CCCVResult:
    trajectory: P2DTrajectory
    cc_duration_s: float
    cv_duration_s: float
    charged_capacity_Ah: float
    charged_energy_Wh: float
    termination_reason: str


class AdvancedProtocolRunner:
    """Execute bounded protocols while preserving segment-level cutoffs."""

    def __init__(self, model: P2DModel) -> None:
        self.model = model

    def run(
        self,
        segments: Iterable[CurrentSegment],
        *,
        initial_state: P2DState | None = None,
        initial_soc: float = 1.0,
    ) -> P2DTrajectory:
        state = (
            self.model.initialize(initial_soc)
            if initial_state is None
            else initial_state.copy()
        )
        states = [state.copy()]
        labels = ["initial"]
        reasons: list[str] = []
        for segment in segments:
            result = self.model.run((segment,), initial_state=state)
            states.extend(item.copy() for item in result.states[1:])
            labels.extend(result.segment_labels[1:])
            state = result.states[-1].copy()
            reasons.append(result.termination_reason)
        nontrivial = [reason for reason in reasons if reason != "completed"]
        return P2DTrajectory(
            states=states,
            segment_labels=labels,
            termination_reason="; ".join(nontrivial) if nontrivial else "completed",
        )

    def run_cycles(
        self,
        protocol: CyclingProtocol,
        *,
        initial_state: P2DState | None = None,
        initial_soc: float = 1.0,
    ) -> CyclingResult:
        state = (
            self.model.initialize(initial_soc)
            if initial_state is None
            else initial_state.copy()
        )
        all_states = [state.copy()]
        all_labels = ["initial"]
        cycle_ids = [1]
        summaries: list[CycleSummary] = []
        first_discharge_capacity: float | None = None

        for cycle in range(1, protocol.cycles + 1):
            cycle_states = [state.copy()]
            cycle_labels = ["initial"]
            reasons: list[str] = []
            for segment in _cycle_segments(protocol, cycle):
                result = self.model.run((segment,), initial_state=state)
                new_states = result.states[1:]
                all_states.extend(item.copy() for item in new_states)
                all_labels.extend(result.segment_labels[1:])
                cycle_ids.extend([cycle] * len(new_states))
                cycle_states.extend(item.copy() for item in new_states)
                cycle_labels.extend(result.segment_labels[1:])
                state = result.states[-1].copy()
                reasons.append(result.termination_reason)

            accounting = _trajectory_accounting(cycle_states)
            if first_discharge_capacity is None:
                first_discharge_capacity = accounting["discharge_capacity_Ah"]
            reference = max(first_discharge_capacity, np.finfo(float).tiny)
            summaries.append(
                CycleSummary(
                    cycle=cycle,
                    discharge_capacity_Ah=accounting["discharge_capacity_Ah"],
                    charge_capacity_Ah=accounting["charge_capacity_Ah"],
                    discharge_energy_Wh=accounting["discharge_energy_Wh"],
                    charge_energy_Wh=accounting["charge_energy_Wh"],
                    coulombic_efficiency=_safe_ratio(
                        accounting["discharge_capacity_Ah"],
                        accounting["charge_capacity_Ah"],
                    ),
                    energy_efficiency=_safe_ratio(
                        accounting["discharge_energy_Wh"],
                        accounting["charge_energy_Wh"],
                    ),
                    minimum_voltage_V=float(
                        min(item.terminal_voltage_V for item in cycle_states)
                    ),
                    maximum_voltage_V=float(
                        max(item.terminal_voltage_V for item in cycle_states)
                    ),
                    final_soc=state.soc(self.model.parameters),
                    final_temperature_K=state.temperature_K,
                    capacity_retention=accounting["discharge_capacity_Ah"] / reference,
                    termination_reasons=tuple(reasons),
                )
            )

        return CyclingResult(
            trajectory=P2DTrajectory(
                states=all_states,
                segment_labels=all_labels,
                termination_reason="completed",
            ),
            cycle_index=np.asarray(cycle_ids),
            summaries=tuple(summaries),
        )

    def run_cccv_charge(
        self,
        initial_state: P2DState,
        config: CCCVConfig,
    ) -> CCCVResult:
        """Charge first at constant current and then hold terminal voltage."""

        cc = self.model.run(
            (
                CurrentSegment(
                    config.maximum_cc_duration_s,
                    -config.charge_current_A,
                    config.time_step_s,
                    maximum_voltage_V=config.voltage_limit_V,
                    label="CC charge",
                ),
            ),
            initial_state=initial_state,
        )
        states = [item.copy() for item in cc.states]
        labels = list(cc.segment_labels)
        state = states[-1].copy()
        cc_duration = state.time_s - initial_state.time_s
        cv_start = state.time_s
        termination = "maximum CV duration reached"

        while state.time_s - cv_start < config.maximum_cv_duration_s - 1.0e-14:
            dt_s = min(
                config.time_step_s,
                config.maximum_cv_duration_s - (state.time_s - cv_start),
            )
            current, candidate = _voltage_control_step(
                self.model,
                state,
                target_voltage_V=config.voltage_limit_V,
                maximum_charge_current_A=config.charge_current_A,
                dt_s=dt_s,
                voltage_tolerance_V=config.voltage_tolerance_V,
                current_tolerance_A=config.current_tolerance_A,
            )
            state = candidate
            states.append(state.copy())
            labels.append("CV charge")
            if abs(current) <= config.taper_current_A:
                termination = "taper current reached"
                break

        trajectory = P2DTrajectory(
            states=states,
            segment_labels=labels,
            termination_reason=termination,
        )
        accounting = _trajectory_accounting(states)
        return CCCVResult(
            trajectory=trajectory,
            cc_duration_s=cc_duration,
            cv_duration_s=state.time_s - cv_start,
            charged_capacity_Ah=accounting["charge_capacity_Ah"],
            charged_energy_Wh=accounting["charge_energy_Wh"],
            termination_reason=termination,
        )


def gitt_protocol(
    *,
    current_A: float,
    pulse_duration_s: float,
    rest_duration_s: float,
    repetitions: int,
    time_step_s: float,
    minimum_voltage_V: float | None = None,
    maximum_voltage_V: float | None = None,
) -> tuple[CurrentSegment, ...]:
    """Build a galvanostatic intermittent titration protocol."""

    if isinstance(repetitions, bool) or not isinstance(
        repetitions, (int, np.integer)
    ):
        raise TypeError("GITT repetitions must be an integer")
    if repetitions < 1:
        raise ValueError("GITT repetitions must be positive")
    segments: list[CurrentSegment] = []
    for index in range(1, repetitions + 1):
        segments.append(
            CurrentSegment(
                pulse_duration_s,
                current_A,
                time_step_s,
                minimum_voltage_V=minimum_voltage_V,
                maximum_voltage_V=maximum_voltage_V,
                label=f"GITT pulse {index}",
            )
        )
        segments.append(
            CurrentSegment(
                rest_duration_s,
                0.0,
                time_step_s,
                label=f"GITT rest {index}",
            )
        )
    return tuple(segments)


def hppc_protocol(
    *,
    discharge_current_A: float,
    charge_current_A: float,
    pulse_duration_s: float,
    rest_duration_s: float,
    repetitions: int,
    time_step_s: float,
) -> tuple[CurrentSegment, ...]:
    """Build a symmetric discharge/rest/charge/rest HPPC pulse sequence."""

    if min(discharge_current_A, charge_current_A, pulse_duration_s, rest_duration_s) <= 0:
        raise ValueError("HPPC currents and durations must be positive")
    if isinstance(repetitions, bool) or not isinstance(
        repetitions, (int, np.integer)
    ):
        raise TypeError("HPPC repetitions must be an integer")
    if repetitions < 1:
        raise ValueError("HPPC repetitions must be positive")
    segments: list[CurrentSegment] = []
    for index in range(1, repetitions + 1):
        segments.extend(
            (
                CurrentSegment(
                    pulse_duration_s,
                    discharge_current_A,
                    time_step_s,
                    label=f"HPPC discharge {index}",
                ),
                CurrentSegment(
                    rest_duration_s,
                    0.0,
                    time_step_s,
                    label=f"HPPC rest after discharge {index}",
                ),
                CurrentSegment(
                    pulse_duration_s,
                    -charge_current_A,
                    time_step_s,
                    label=f"HPPC charge {index}",
                ),
                CurrentSegment(
                    rest_duration_s,
                    0.0,
                    time_step_s,
                    label=f"HPPC rest after charge {index}",
                ),
            )
        )
    return tuple(segments)


def drive_cycle_segment(
    time_s: np.ndarray,
    current_A: np.ndarray,
    *,
    maximum_time_step_s: float,
    label: str = "drive cycle",
) -> CurrentSegment:
    """Create a linearly interpolated arbitrary-current drive-cycle segment."""

    times = np.asarray(time_s, dtype=np.float64)
    currents = np.asarray(current_A, dtype=np.float64)
    if (
        times.ndim != 1
        or currents.shape != times.shape
        or len(times) < 2
        or np.any(np.diff(times) <= 0.0)
        or not np.all(np.isfinite(currents))
    ):
        raise ValueError("drive-cycle time/current arrays are invalid")
    shifted = times - times[0]

    def current(local_time_s: float, state: P2DState) -> float:
        del state
        return float(np.interp(local_time_s, shifted, currents))

    return CurrentSegment(
        duration_s=float(shifted[-1]),
        current_A=current,
        time_step_s=maximum_time_step_s,
        label=label,
    )


def _cycle_segments(
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


def _voltage_control_step(
    model: P2DModel,
    state: P2DState,
    *,
    target_voltage_V: float,
    maximum_charge_current_A: float,
    dt_s: float,
    voltage_tolerance_V: float,
    current_tolerance_A: float,
) -> tuple[float, P2DState]:
    try:
        from scipy.optimize import brentq
    except ImportError as exc:
        raise ImportError("CC-CV control requires SciPy") from exc

    cache: dict[float, P2DState] = {}

    def evaluate(current_A: float) -> float:
        key = float(current_A)
        if key not in cache:
            cache[key] = model.step(state, key, dt_s)
        return cache[key].terminal_voltage_V - target_voltage_V

    lower = -float(maximum_charge_current_A)
    upper = 0.0
    f_lower = evaluate(lower)
    f_upper = evaluate(upper)
    if abs(f_lower) <= voltage_tolerance_V:
        current = lower
    elif abs(f_upper) <= voltage_tolerance_V:
        current = upper
    elif f_lower * f_upper < 0.0:
        current = float(
            brentq(
                evaluate,
                lower,
                upper,
                xtol=current_tolerance_A,
                rtol=4.0 * np.finfo(float).eps,
            )
        )
    else:
        current = lower if abs(f_lower) < abs(f_upper) else upper
    if current not in cache:
        cache[current] = model.step(state, current, dt_s)
    return current, cache[current]


def _trajectory_accounting(states: list[P2DState]) -> dict[str, float]:
    if len(states) < 2:
        return {
            "discharge_capacity_Ah": 0.0,
            "charge_capacity_Ah": 0.0,
            "discharge_energy_Wh": 0.0,
            "charge_energy_Wh": 0.0,
        }
    times = np.asarray([state.time_s for state in states])
    dt = np.diff(times)
    current = np.asarray([state.current_A for state in states[1:]])
    voltage = np.asarray([state.terminal_voltage_V for state in states[1:]])
    discharge = np.maximum(current, 0.0)
    charge = np.maximum(-current, 0.0)
    return {
        "discharge_capacity_Ah": float(np.sum(discharge * dt) / 3600.0),
        "charge_capacity_Ah": float(np.sum(charge * dt) / 3600.0),
        "discharge_energy_Wh": float(np.sum(discharge * voltage * dt) / 3600.0),
        "charge_energy_Wh": float(np.sum(charge * voltage * dt) / 3600.0),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


__all__ = [
    "AdvancedProtocolRunner",
    "CCCVConfig",
    "CCCVResult",
    "CycleSummary",
    "CyclingProtocol",
    "CyclingResult",
    "drive_cycle_segment",
    "gitt_protocol",
    "hppc_protocol",
]
