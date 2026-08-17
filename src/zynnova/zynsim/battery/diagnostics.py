"""Electrochemical diagnostics derived from deterministic battery simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .p2d import P2DModel, P2DState, P2DTrajectory


@dataclass(frozen=True, slots=True)
class EISConfig:
    current_amplitude_A: float
    cycles_per_frequency: int = 4
    discard_cycles: int = 2
    points_per_cycle: int = 24
    bias_current_A: float = 0.0

    def __post_init__(self) -> None:
        integer_values = (
            self.cycles_per_frequency,
            self.discard_cycles,
            self.points_per_cycle,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for value in integer_values
        ):
            raise TypeError("EIS cycle and point counts must be integers")
        if (
            not np.isfinite(self.current_amplitude_A)
            or self.current_amplitude_A <= 0.0
        ):
            raise ValueError("EIS current amplitude must be positive")
        if not np.isfinite(self.bias_current_A):
            raise ValueError("EIS bias current must be finite")
        if self.cycles_per_frequency < 2:
            raise ValueError("EIS requires at least two cycles per frequency")
        if not 0 <= self.discard_cycles < self.cycles_per_frequency:
            raise ValueError("discard_cycles must be below cycles_per_frequency")
        if self.points_per_cycle < 8:
            raise ValueError("EIS needs at least eight points per cycle")


@dataclass(frozen=True, slots=True)
class EISResult:
    frequency_Hz: np.ndarray
    impedance_ohm: np.ndarray
    voltage_amplitude_V: np.ndarray
    current_amplitude_A: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequency_Hz, dtype=np.float64)
        impedance = np.asarray(self.impedance_ohm, dtype=np.complex128)
        voltage = np.asarray(self.voltage_amplitude_V, dtype=np.float64)
        current = np.asarray(self.current_amplitude_A, dtype=np.float64)
        if (
            frequency.ndim != 1
            or impedance.shape != frequency.shape
            or voltage.shape != frequency.shape
            or current.shape != frequency.shape
            or np.any(frequency <= 0.0)
            or not np.all(np.isfinite(impedance))
        ):
            raise ValueError("EIS result arrays are invalid")
        object.__setattr__(self, "frequency_Hz", frequency)
        object.__setattr__(self, "impedance_ohm", impedance)
        object.__setattr__(self, "voltage_amplitude_V", voltage)
        object.__setattr__(self, "current_amplitude_A", current)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def magnitude_ohm(self) -> np.ndarray:
        return np.abs(self.impedance_ohm)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.impedance_ohm, deg=True)


@dataclass(frozen=True, slots=True)
class PulseResistance:
    time_s: float
    delta_current_A: float
    instantaneous_ohm: float
    relaxed_ohm: float | None


def simulate_eis(
    model: P2DModel,
    state: P2DState,
    frequencies_Hz: np.ndarray,
    config: EISConfig,
) -> EISResult:
    """Estimate small-signal impedance by time-domain harmonic simulation.

    Each frequency starts from an identical copied state.  With the ZynSim sign
    convention, positive current is discharge, so impedance is reported as
    ``-V_hat / I_hat`` to retain the conventional positive ohmic direction.
    """

    frequencies = np.asarray(frequencies_Hz, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or len(frequencies) < 1
        or np.any(~np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
    ):
        raise ValueError("frequencies_Hz must contain positive finite values")
    impedance: list[complex] = []
    voltage_amplitudes: list[float] = []
    current_amplitudes: list[float] = []

    for frequency in frequencies:
        local = state.copy()
        dt_s = 1.0 / (frequency * config.points_per_cycle)
        total_steps = config.cycles_per_frequency * config.points_per_cycle
        discard_steps = config.discard_cycles * config.points_per_cycle
        times: list[float] = []
        currents: list[float] = []
        voltages: list[float] = []
        start_time = local.time_s
        for step in range(total_steps):
            local_time = (step + 1) * dt_s
            current = config.bias_current_A + config.current_amplitude_A * np.sin(
                2.0 * np.pi * frequency * local_time
            )
            local = model.step(local, float(current), dt_s)
            if step >= discard_steps:
                times.append(local.time_s - start_time)
                currents.append(current)
                voltages.append(local.terminal_voltage_V)
        z_value, voltage_hat, current_hat = impedance_from_time_series(
            np.asarray(times),
            np.asarray(currents),
            np.asarray(voltages),
            frequency_Hz=float(frequency),
            positive_current_is_discharge=True,
        )
        impedance.append(z_value)
        voltage_amplitudes.append(abs(voltage_hat))
        current_amplitudes.append(abs(current_hat))

    return EISResult(
        frequency_Hz=frequencies,
        impedance_ohm=np.asarray(impedance),
        voltage_amplitude_V=np.asarray(voltage_amplitudes),
        current_amplitude_A=np.asarray(current_amplitudes),
        metadata={
            "method": "time-domain harmonic projection",
            "cycles_per_frequency": config.cycles_per_frequency,
            "discard_cycles": config.discard_cycles,
            "points_per_cycle": config.points_per_cycle,
            "bias_current_A": config.bias_current_A,
            "positive_current_is_discharge": True,
        },
    )


def impedance_from_time_series(
    time_s: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    *,
    frequency_Hz: float,
    positive_current_is_discharge: bool = True,
) -> tuple[complex, complex, complex]:
    """Project current and voltage samples onto one harmonic."""

    time = np.asarray(time_s, dtype=np.float64)
    current = np.asarray(current_A, dtype=np.float64)
    voltage = np.asarray(voltage_V, dtype=np.float64)
    if (
        time.ndim != 1
        or current.shape != time.shape
        or voltage.shape != time.shape
        or len(time) < 8
        or frequency_Hz <= 0.0
    ):
        raise ValueError("harmonic time-series inputs are invalid")
    kernel = np.exp(-2.0j * np.pi * frequency_Hz * time)
    current_hat = 2.0 * np.mean((current - np.mean(current)) * kernel)
    voltage_hat = 2.0 * np.mean((voltage - np.mean(voltage)) * kernel)
    if abs(current_hat) <= np.finfo(float).eps:
        raise ValueError("current harmonic amplitude is zero")
    sign = -1.0 if positive_current_is_discharge else 1.0
    return complex(sign * voltage_hat / current_hat), complex(voltage_hat), complex(
        current_hat
    )


def pulse_resistances(
    trajectory: P2DTrajectory,
    *,
    relaxation_window_s: float | None = None,
    current_threshold_A: float = 1.0e-12,
) -> tuple[PulseResistance, ...]:
    """Estimate instantaneous and optional relaxed resistance at current steps."""

    time = trajectory.time_s
    voltage = trajectory.voltage_V
    current = trajectory.current_A
    results: list[PulseResistance] = []
    for index in range(1, len(time)):
        delta_current = current[index] - current[index - 1]
        if abs(delta_current) <= current_threshold_A:
            continue
        instantaneous = -float((voltage[index] - voltage[index - 1]) / delta_current)
        relaxed: float | None = None
        if relaxation_window_s is not None and relaxation_window_s > 0.0:
            target = time[index] + relaxation_window_s
            relaxed_index = int(np.searchsorted(time, target, side="left"))
            if relaxed_index < len(time):
                relaxed = -float(
                    (voltage[relaxed_index] - voltage[index - 1]) / delta_current
                )
        results.append(
            PulseResistance(
                time_s=float(time[index]),
                delta_current_A=float(delta_current),
                instantaneous_ohm=instantaneous,
                relaxed_ohm=relaxed,
            )
        )
    return tuple(results)


__all__ = [
    "EISConfig",
    "EISResult",
    "PulseResistance",
    "impedance_from_time_series",
    "pulse_resistances",
    "simulate_eis",
]
