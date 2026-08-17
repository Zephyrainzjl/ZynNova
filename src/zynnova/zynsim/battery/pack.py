"""Pack-scale electrothermal networks and a lightweight ECM state observer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


OCVFunction = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class CellECMParameters:
    capacity_Ah: float
    ohmic_resistance_ohm: float
    polarization_resistance_ohm: float
    polarization_capacitance_F: float
    thermal_capacity_J_K: float
    cooling_conductance_W_K: float
    ocv_V: OCVFunction

    def __post_init__(self) -> None:
        values = (
            self.capacity_Ah,
            self.ohmic_resistance_ohm,
            self.polarization_resistance_ohm,
            self.polarization_capacitance_F,
            self.thermal_capacity_J_K,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("ECM electrical, capacity, and thermal values must be positive")
        if (
            not np.isfinite(self.cooling_conductance_W_K)
            or self.cooling_conductance_W_K < 0.0
            or not callable(self.ocv_V)
        ):
            raise ValueError("ECM cooling must be non-negative and OCV must be callable")


@dataclass(slots=True)
class CellECMState:
    soc: float
    polarization_voltage_V: float
    temperature_K: float

    def copy(self) -> CellECMState:
        return CellECMState(
            float(self.soc),
            float(self.polarization_voltage_V),
            float(self.temperature_K),
        )


@dataclass(slots=True)
class PackState:
    time_s: float
    cells: list[CellECMState]
    pack_current_A: float = 0.0
    pack_voltage_V: float = 0.0
    cell_current_A: np.ndarray = field(default_factory=lambda: np.empty(0))
    group_voltage_V: np.ndarray = field(default_factory=lambda: np.empty(0))

    def copy(self) -> PackState:
        return PackState(
            time_s=float(self.time_s),
            cells=[state.copy() for state in self.cells],
            pack_current_A=float(self.pack_current_A),
            pack_voltage_V=float(self.pack_voltage_V),
            cell_current_A=np.asarray(self.cell_current_A).copy(),
            group_voltage_V=np.asarray(self.group_voltage_V).copy(),
        )


@dataclass(slots=True)
class PackTrajectory:
    states: list[PackState]

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray([state.time_s for state in self.states])

    @property
    def voltage_V(self) -> np.ndarray:
        return np.asarray([state.pack_voltage_V for state in self.states])

    @property
    def current_A(self) -> np.ndarray:
        return np.asarray([state.pack_current_A for state in self.states])

    @property
    def soc(self) -> np.ndarray:
        return np.asarray([[cell.soc for cell in state.cells] for state in self.states])

    @property
    def temperature_K(self) -> np.ndarray:
        return np.asarray(
            [[cell.temperature_K for cell in state.cells] for state in self.states]
        )


class ElectrothermalPack:
    """Series groups of parallel ECM cells with pairwise thermal coupling."""

    def __init__(
        self,
        parameters: list[CellECMParameters] | tuple[CellECMParameters, ...],
        *,
        series_count: int,
        parallel_count: int,
        thermal_conductance_W_K: np.ndarray | None = None,
        ambient_temperature_K: float = 298.15,
    ) -> None:
        self.parameters = tuple(parameters)
        if (
            isinstance(series_count, bool)
            or not isinstance(series_count, (int, np.integer))
            or isinstance(parallel_count, bool)
            or not isinstance(parallel_count, (int, np.integer))
        ):
            raise TypeError("series_count and parallel_count must be integers")
        self.series_count = int(series_count)
        self.parallel_count = int(parallel_count)
        expected = self.series_count * self.parallel_count
        if self.series_count < 1 or self.parallel_count < 1 or len(parameters) != expected:
            raise ValueError("pack parameter count must equal series_count*parallel_count")
        if not np.isfinite(ambient_temperature_K) or ambient_temperature_K <= 0.0:
            raise ValueError("ambient temperature must be positive")
        self.ambient_temperature_K = float(ambient_temperature_K)
        if thermal_conductance_W_K is None:
            conductance = np.zeros((expected, expected), dtype=np.float64)
        else:
            conductance = np.asarray(thermal_conductance_W_K, dtype=np.float64)
            if (
                conductance.shape != (expected, expected)
                or np.any(conductance < 0.0)
                or not np.allclose(conductance, conductance.T)
            ):
                raise ValueError(
                    "thermal conductance must be a symmetric non-negative square matrix"
                )
            if not np.allclose(np.diag(conductance), 0.0):
                raise ValueError("thermal conductance diagonal must be zero")
        self.thermal_conductance_W_K = conductance

    def initialize(
        self,
        soc: float | np.ndarray = 1.0,
        *,
        temperature_K: float | np.ndarray | None = None,
    ) -> PackState:
        count = len(self.parameters)
        soc_values = np.broadcast_to(np.asarray(soc, dtype=float), (count,))
        temperatures = np.broadcast_to(
            np.asarray(
                self.ambient_temperature_K
                if temperature_K is None
                else temperature_K,
                dtype=float,
            ),
            (count,),
        )
        if np.any((soc_values < 0.0) | (soc_values > 1.0)) or np.any(temperatures <= 0.0):
            raise ValueError("initial pack SOC or temperature is invalid")
        state = PackState(
            time_s=0.0,
            cells=[
                CellECMState(float(soc_values[i]), 0.0, float(temperatures[i]))
                for i in range(count)
            ],
        )
        return self._with_outputs(state, 0.0)

    def step(
        self,
        state: PackState,
        pack_current_A: float,
        dt_s: float,
    ) -> PackState:
        if dt_s <= 0.0 or not np.isfinite(pack_current_A):
            raise ValueError("pack step requires positive dt and finite current")
        if len(state.cells) != len(self.parameters):
            raise ValueError("pack state cell count is inconsistent")
        cell_current, _ = self._current_distribution(state, pack_current_A)
        temperatures = np.asarray([cell.temperature_K for cell in state.cells])
        thermal_exchange = self.thermal_conductance_W_K @ temperatures - (
            np.sum(self.thermal_conductance_W_K, axis=1) * temperatures
        )
        updated: list[CellECMState] = []
        for index, (cell, parameters) in enumerate(zip(state.cells, self.parameters, strict=True)):
            current = cell_current[index]
            soc = cell.soc - current * dt_s / (3600.0 * parameters.capacity_Ah)
            if not 0.0 <= soc <= 1.0:
                raise ValueError(f"cell {index} SOC left [0, 1]")
            tau = (
                parameters.polarization_resistance_ohm
                * parameters.polarization_capacitance_F
            )
            decay = np.exp(-dt_s / tau)
            polarization = (
                decay * cell.polarization_voltage_V
                + parameters.polarization_resistance_ohm
                * current
                * (1.0 - decay)
            )
            irreversible_heat = (
                current**2 * parameters.ohmic_resistance_ohm
                + polarization**2 / parameters.polarization_resistance_ohm
            )
            cooling = parameters.cooling_conductance_W_K * (
                self.ambient_temperature_K - cell.temperature_K
            )
            temperature = cell.temperature_K + dt_s * (
                irreversible_heat + cooling + thermal_exchange[index]
            ) / parameters.thermal_capacity_J_K
            if temperature <= 0.0 or not np.isfinite(temperature):
                raise ValueError(f"cell {index} temperature became invalid")
            updated.append(CellECMState(float(soc), float(polarization), float(temperature)))
        result = PackState(time_s=state.time_s + dt_s, cells=updated)
        return self._with_outputs(result, pack_current_A)

    def run(
        self,
        initial_state: PackState,
        time_s: np.ndarray,
        current_A: np.ndarray,
        *,
        maximum_time_step_s: float,
    ) -> PackTrajectory:
        times = np.asarray(time_s, dtype=np.float64)
        currents = np.asarray(current_A, dtype=np.float64)
        if (
            times.ndim != 1
            or currents.shape != times.shape
            or len(times) < 2
            or np.any(np.diff(times) <= 0.0)
            or maximum_time_step_s <= 0.0
        ):
            raise ValueError("pack drive-cycle arrays or time step are invalid")
        state = initial_state.copy()
        states = [state.copy()]
        for interval in range(len(times) - 1):
            duration = times[interval + 1] - times[interval]
            elapsed = 0.0
            while elapsed < duration - 1.0e-14:
                dt_s = min(maximum_time_step_s, duration - elapsed)
                local_fraction = (elapsed + 0.5 * dt_s) / duration
                current = (1.0 - local_fraction) * currents[interval] + (
                    local_fraction * currents[interval + 1]
                )
                state = self.step(state, float(current), dt_s)
                states.append(state.copy())
                elapsed += dt_s
        return PackTrajectory(states)

    def _current_distribution(
        self,
        state: PackState,
        pack_current_A: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cell_currents = np.empty(len(self.parameters), dtype=np.float64)
        group_voltages = np.empty(self.series_count, dtype=np.float64)
        for group in range(self.series_count):
            start = group * self.parallel_count
            stop = start + self.parallel_count
            parameters = self.parameters[start:stop]
            cells = state.cells[start:stop]
            conductance = np.asarray(
                [1.0 / item.ohmic_resistance_ohm for item in parameters]
            )
            source_voltage = np.asarray(
                [
                    item.ocv_V(cell.soc, cell.temperature_K)
                    - cell.polarization_voltage_V
                    for item, cell in zip(parameters, cells, strict=True)
                ]
            )
            if not np.isfinite(source_voltage).all():
                raise ValueError("cell OCV returned a non-finite voltage")
            group_voltage = (
                np.dot(conductance, source_voltage) - pack_current_A
            ) / np.sum(conductance)
            group_current = conductance * (source_voltage - group_voltage)
            cell_currents[start:stop] = group_current
            group_voltages[group] = group_voltage
        return cell_currents, group_voltages

    def _with_outputs(self, state: PackState, pack_current_A: float) -> PackState:
        currents, voltages = self._current_distribution(state, pack_current_A)
        state.pack_current_A = float(pack_current_A)
        state.cell_current_A = currents
        state.group_voltage_V = voltages
        state.pack_voltage_V = float(np.sum(voltages))
        return state


@dataclass(slots=True)
class CellECMObserver:
    """Extended Kalman observer for SOC, polarization voltage, and temperature."""

    parameters: CellECMParameters
    state: CellECMState
    covariance: np.ndarray
    process_covariance: np.ndarray
    voltage_variance: float

    def __post_init__(self) -> None:
        self.covariance = _covariance(self.covariance, "covariance")
        self.process_covariance = _covariance(
            self.process_covariance, "process_covariance"
        )
        if self.voltage_variance <= 0.0:
            raise ValueError("voltage_variance must be positive")

    def update(
        self,
        *,
        current_A: float,
        measured_voltage_V: float,
        dt_s: float,
        ambient_temperature_K: float = 298.15,
    ) -> CellECMState:
        if dt_s <= 0.0:
            raise ValueError("observer time step must be positive")
        p = self.parameters
        tau = p.polarization_resistance_ohm * p.polarization_capacitance_F
        decay = np.exp(-dt_s / tau)
        predicted = np.asarray(
            [
                self.state.soc - current_A * dt_s / (3600.0 * p.capacity_Ah),
                decay * self.state.polarization_voltage_V
                + p.polarization_resistance_ohm * current_A * (1.0 - decay),
                self.state.temperature_K
                + dt_s
                * (
                    current_A**2 * p.ohmic_resistance_ohm
                    + p.cooling_conductance_W_K
                    * (ambient_temperature_K - self.state.temperature_K)
                )
                / p.thermal_capacity_J_K,
            ],
            dtype=np.float64,
        )
        transition = np.diag(
            (
                1.0,
                decay,
                1.0
                - dt_s * p.cooling_conductance_W_K / p.thermal_capacity_J_K,
            )
        )
        predicted_covariance = (
            transition @ self.covariance @ transition.T + self.process_covariance
        )
        measurement = (
            p.ocv_V(predicted[0], predicted[2])
            - predicted[1]
            - current_A * p.ohmic_resistance_ohm
        )
        derivative_soc = _central_derivative(
            lambda value: p.ocv_V(value, predicted[2]),
            predicted[0],
            lower=0.0,
            upper=1.0,
        )
        derivative_temperature = _central_derivative(
            lambda value: p.ocv_V(predicted[0], value),
            predicted[2],
            lower=1.0,
            upper=float("inf"),
        )
        measurement_jacobian = np.asarray(
            [[derivative_soc, -1.0, derivative_temperature]]
        )
        innovation_variance = float(
            (
                measurement_jacobian
                @ predicted_covariance
                @ measurement_jacobian.T
            ).item()
            + self.voltage_variance
        )
        gain = (
            predicted_covariance @ measurement_jacobian.T / innovation_variance
        )
        corrected = predicted + gain[:, 0] * (measured_voltage_V - measurement)
        corrected[0] = np.clip(corrected[0], 0.0, 1.0)
        corrected[2] = max(corrected[2], 1.0)
        identity = np.eye(3)
        self.covariance = (
            identity - gain @ measurement_jacobian
        ) @ predicted_covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.state = CellECMState(*map(float, corrected))
        return self.state.copy()


def linear_ocv(
    minimum_voltage_V: float = 3.0,
    maximum_voltage_V: float = 4.2,
) -> OCVFunction:
    if minimum_voltage_V >= maximum_voltage_V:
        raise ValueError("minimum OCV must be below maximum OCV")

    def evaluate(soc: float, temperature_K: float) -> float:
        del temperature_K
        return float(
            minimum_voltage_V
            + np.clip(soc, 0.0, 1.0)
            * (maximum_voltage_V - minimum_voltage_V)
        )

    return evaluate


def _covariance(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (3, 3)
        or not np.allclose(array, array.T)
        or np.linalg.eigvalsh(array).min() < -1.0e-14
    ):
        raise ValueError(f"{name} must be a symmetric positive-semidefinite 3x3 matrix")
    return array


def _central_derivative(
    function: Callable[[float], float],
    value: float,
    *,
    lower: float,
    upper: float,
) -> float:
    step = 1.0e-5 * max(abs(value), 1.0)
    left = max(value - step, lower)
    right = min(value + step, upper)
    if right <= left:
        return 0.0
    return float((function(right) - function(left)) / (right - left))


__all__ = [
    "CellECMObserver",
    "CellECMParameters",
    "CellECMState",
    "ElectrothermalPack",
    "OCVFunction",
    "PackState",
    "PackTrajectory",
    "linear_ocv",
]
