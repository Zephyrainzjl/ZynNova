from __future__ import annotations

import json
from typing import Any

import numpy as np

from .constants import (
    BOLTZMANN_EV_PER_K,
    FORCE_TO_ACCELERATION,
    KINETIC_ENERGY_FACTOR,
)
from .native import NativeBackend, native_module, resolve_native_backend


def _system_arrays(
    masses: np.ndarray | list[float],
    mobile: np.ndarray | list[bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    mass_array = np.ascontiguousarray(masses, dtype=np.float64)
    if mass_array.ndim != 1 or len(mass_array) == 0:
        raise ValueError("masses must have shape [N] with N > 0")
    if not np.all(np.isfinite(mass_array)) or np.any(mass_array <= 0):
        raise ValueError("all masses must be finite and positive")
    mobile_array = (
        np.ones(len(mass_array), dtype=bool)
        if mobile is None
        else np.ascontiguousarray(mobile, dtype=bool)
    )
    if mobile_array.shape != mass_array.shape or not np.any(mobile_array):
        raise ValueError("mobile must have shape [N] and contain at least one True value")
    return mass_array, mobile_array


def _phase_array(values: Any, atom_count: int, name: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.shape != (atom_count, 3):
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _python_kinetic_energy(
    masses: np.ndarray,
    velocities: np.ndarray,
    mobile: np.ndarray,
) -> float:
    squared_speed = np.sum(velocities[mobile] ** 2, axis=1)
    return float(0.5 * KINETIC_ENERGY_FACTOR * np.sum(masses[mobile] * squared_speed))


def _python_temperature(
    masses: np.ndarray,
    velocities: np.ndarray,
    mobile: np.ndarray,
    remove_center_of_mass_dof: bool,
) -> float:
    count = int(np.count_nonzero(mobile))
    degrees_of_freedom = 3 * count - (3 if remove_center_of_mass_dof and count > 1 else 0)
    if degrees_of_freedom == 0:
        return 0.0
    energy = _python_kinetic_energy(masses, velocities, mobile)
    return 2.0 * energy / (degrees_of_freedom * BOLTZMANN_EV_PER_K)


def maxwell_boltzmann_velocities(
    masses: np.ndarray | list[float],
    temperature_K: float,
    *,
    mobile: np.ndarray | list[bool] | None = None,
    seed: int = 0,
    remove_center_of_mass: bool = True,
    exact_temperature: bool = True,
    backend: NativeBackend = "auto",
) -> np.ndarray:
    mass_array, mobile_array = _system_arrays(masses, mobile)
    if temperature_K <= 0 or not np.isfinite(temperature_K):
        raise ValueError("temperature_K must be finite and positive")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    selected = resolve_native_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        return np.asarray(
            native.maxwell_boltzmann_velocities(
                mass_array,
                mobile_array,
                float(temperature_K),
                int(seed),
                bool(remove_center_of_mass),
                bool(exact_temperature),
            ),
            dtype=np.float64,
        )

    rng = np.random.default_rng(seed)
    velocities = np.zeros((len(mass_array), 3), dtype=np.float64)
    standard_deviation = np.sqrt(
        BOLTZMANN_EV_PER_K * temperature_K / (mass_array[mobile_array] * KINETIC_ENERGY_FACTOR)
    )
    velocities[mobile_array] = (
        rng.normal(size=(int(np.count_nonzero(mobile_array)), 3)) * standard_deviation[:, None]
    )
    remove_com = remove_center_of_mass and np.count_nonzero(mobile_array) > 1
    if remove_com:
        center_velocity = np.average(
            velocities[mobile_array],
            axis=0,
            weights=mass_array[mobile_array],
        )
        velocities[mobile_array] -= center_velocity
    if exact_temperature:
        current = _python_temperature(mass_array, velocities, mobile_array, bool(remove_com))
        if current <= 0 or not np.isfinite(current):
            raise RuntimeError("cannot rescale a zero-temperature velocity distribution")
        velocities[mobile_array] *= np.sqrt(temperature_K / current)
    return velocities


def kinetic_energy_eV(
    masses: np.ndarray | list[float],
    velocities: np.ndarray,
    *,
    mobile: np.ndarray | list[bool] | None = None,
    backend: NativeBackend = "auto",
) -> float:
    mass_array, mobile_array = _system_arrays(masses, mobile)
    velocity_array = _phase_array(velocities, len(mass_array), "velocities")
    selected = resolve_native_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        return float(native.kinetic_energy_eV(mass_array, velocity_array, mobile_array))
    return _python_kinetic_energy(mass_array, velocity_array, mobile_array)


def instantaneous_temperature_K(
    masses: np.ndarray | list[float],
    velocities: np.ndarray,
    *,
    mobile: np.ndarray | list[bool] | None = None,
    remove_center_of_mass_dof: bool = True,
    backend: NativeBackend = "auto",
) -> float:
    mass_array, mobile_array = _system_arrays(masses, mobile)
    velocity_array = _phase_array(velocities, len(mass_array), "velocities")
    selected = resolve_native_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        return float(
            native.instantaneous_temperature_K(
                mass_array,
                velocity_array,
                mobile_array,
                bool(remove_center_of_mass_dof),
            )
        )
    return _python_temperature(mass_array, velocity_array, mobile_array, remove_center_of_mass_dof)


class _PythonIntegrator:
    def __init__(
        self,
        masses: np.ndarray,
        mobile: np.ndarray,
        timestep_fs: float,
        ensemble: str,
        temperature_K: float,
        friction_per_fs: float,
        seed: int,
    ) -> None:
        if ensemble not in {"nve", "nvt_langevin"}:
            raise ValueError("ensemble must be 'nve' or 'nvt_langevin'")
        if timestep_fs <= 0 or friction_per_fs < 0:
            raise ValueError("timestep_fs must be positive and friction non-negative")
        if ensemble == "nvt_langevin" and temperature_K <= 0:
            raise ValueError("nvt_langevin requires temperature_K > 0")
        self.masses = masses
        self.mobile = mobile
        self.timestep_fs = float(timestep_fs)
        self.ensemble = ensemble
        self.temperature_target_K = float(temperature_K)
        self.friction_per_fs = float(friction_per_fs)
        self._rng = np.random.default_rng(seed)
        self._velocities = np.zeros((len(masses), 3), dtype=np.float64)
        self.step_index = 0
        self.awaiting_forces = False

    def set_velocities(self, velocities: np.ndarray) -> None:
        if self.awaiting_forces:
            raise RuntimeError("cannot replace velocities in the middle of a step")
        self._velocities = _phase_array(velocities, len(self.masses), "velocities").copy()
        self._velocities[~self.mobile] = 0.0

    def velocities(self) -> np.ndarray:
        return self._velocities.copy()

    def _half_kick(self, forces: np.ndarray) -> None:
        self._velocities[self.mobile] += (
            0.5
            * self.timestep_fs
            * FORCE_TO_ACCELERATION
            * forces[self.mobile]
            / self.masses[self.mobile, None]
        )

    def begin_step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        if self.awaiting_forces:
            raise RuntimeError("end_step must be called before beginning another step")
        position_array = _phase_array(positions, len(self.masses), "positions").copy()
        force_array = _phase_array(forces, len(self.masses), "forces")
        self._half_kick(force_array)
        if self.ensemble == "nve":
            position_array[self.mobile] += self.timestep_fs * self._velocities[self.mobile]
        else:
            half_timestep = 0.5 * self.timestep_fs
            position_array[self.mobile] += half_timestep * self._velocities[self.mobile]
            damping = np.exp(-self.friction_per_fs * self.timestep_fs)
            sigma = np.sqrt(
                (1.0 - damping**2)
                * BOLTZMANN_EV_PER_K
                * self.temperature_target_K
                / (self.masses[self.mobile] * KINETIC_ENERGY_FACTOR)
            )
            noise = self._rng.normal(size=(int(np.count_nonzero(self.mobile)), 3))
            self._velocities[self.mobile] = (
                damping * self._velocities[self.mobile] + sigma[:, None] * noise
            )
            position_array[self.mobile] += half_timestep * self._velocities[self.mobile]
        self.awaiting_forces = True
        return position_array

    def end_step(self, forces: np.ndarray) -> np.ndarray:
        if not self.awaiting_forces:
            raise RuntimeError("begin_step must be called before end_step")
        self._half_kick(_phase_array(forces, len(self.masses), "forces"))
        self.awaiting_forces = False
        self.step_index += 1
        return self.velocities()

    def kinetic_energy_eV(self) -> float:
        return _python_kinetic_energy(self.masses, self._velocities, self.mobile)

    def temperature_K(self, remove_center_of_mass_dof: bool = True) -> float:
        return _python_temperature(
            self.masses,
            self._velocities,
            self.mobile,
            remove_center_of_mass_dof,
        )

    @property
    def rng_state(self) -> str:
        return json.dumps(self._rng.bit_generator.state, default=int)

    @rng_state.setter
    def rng_state(self, state: str) -> None:
        if self.awaiting_forces:
            raise RuntimeError("cannot restore RNG state in the middle of a step")
        self._rng.bit_generator.state = json.loads(state)


class AIMDIntegrator:
    """Backend-neutral NVE velocity-Verlet / NVT BAOAB integrator."""

    def __init__(
        self,
        masses: np.ndarray | list[float],
        *,
        mobile: np.ndarray | list[bool] | None = None,
        timestep_fs: float = 0.5,
        ensemble: str = "nve",
        temperature_K: float = 300.0,
        friction_per_fs: float = 0.01,
        seed: int = 0,
        backend: NativeBackend = "auto",
    ) -> None:
        self.masses, self.mobile = _system_arrays(masses, mobile)
        if int(seed) != seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.backend = resolve_native_backend(backend)
        if self.backend == "cpp":
            native = native_module()
            assert native is not None
            self._implementation = native.AIMDIntegrator(
                self.masses,
                self.mobile,
                float(timestep_fs),
                ensemble,
                float(temperature_K),
                float(friction_per_fs),
                int(seed),
            )
        else:
            self._implementation = _PythonIntegrator(
                self.masses,
                self.mobile,
                float(timestep_fs),
                ensemble,
                float(temperature_K),
                float(friction_per_fs),
                int(seed),
            )

    def set_velocities(self, velocities: np.ndarray) -> None:
        self._implementation.set_velocities(
            _phase_array(velocities, len(self.masses), "velocities")
        )

    def velocities(self) -> np.ndarray:
        return np.asarray(self._implementation.velocities(), dtype=np.float64)

    def begin_step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._implementation.begin_step(
                _phase_array(positions, len(self.masses), "positions"),
                _phase_array(forces, len(self.masses), "forces"),
            ),
            dtype=np.float64,
        )

    def end_step(self, forces: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._implementation.end_step(_phase_array(forces, len(self.masses), "forces")),
            dtype=np.float64,
        )

    def kinetic_energy_eV(self) -> float:
        return float(self._implementation.kinetic_energy_eV())

    def temperature_K(self, remove_center_of_mass_dof: bool = True) -> float:
        return float(self._implementation.temperature_K(remove_center_of_mass_dof))

    @property
    def step_index(self) -> int:
        return int(self._implementation.step_index)

    @step_index.setter
    def step_index(self, value: int) -> None:
        self._implementation.step_index = int(value)

    @property
    def rng_state(self) -> str:
        return str(self._implementation.rng_state)

    @rng_state.setter
    def rng_state(self, state: str) -> None:
        self._implementation.rng_state = state

    @property
    def awaiting_forces(self) -> bool:
        return bool(self._implementation.awaiting_forces)
