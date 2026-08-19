from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zynnova.structure import StructureData


def _trapezoidal(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(grid)))


@dataclass(slots=True)
class StationaryStates:
    grid: np.ndarray
    potential: np.ndarray
    energies: np.ndarray
    wavefunctions: np.ndarray
    residual_norms: np.ndarray
    units: str
    backend: str
    mass: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grid = np.ascontiguousarray(self.grid, dtype=np.float64)
        self.potential = np.ascontiguousarray(self.potential, dtype=np.float64)
        self.energies = np.ascontiguousarray(self.energies, dtype=np.float64)
        self.wavefunctions = np.ascontiguousarray(self.wavefunctions, dtype=np.float64)
        self.residual_norms = np.ascontiguousarray(self.residual_norms, dtype=np.float64)
        if self.grid.ndim != 1 or self.potential.shape != self.grid.shape:
            raise ValueError("grid and potential must have matching one-dimensional shapes")
        if self.wavefunctions.shape != (len(self.energies), len(self.grid)):
            raise ValueError("wavefunctions must have shape [num_states, num_points]")
        if self.residual_norms.shape != self.energies.shape:
            raise ValueError("residual_norms must have shape [num_states]")

    def state(self, index: int) -> np.ndarray:
        return self.wavefunctions[index].copy()

    def probability_density(self, index: int) -> np.ndarray:
        return np.abs(self.wavefunctions[index]) ** 2

    def expectation_position(self, index: int) -> float:
        density = self.probability_density(index)
        return _trapezoidal(self.grid * density, self.grid)

    def norm(self, index: int) -> float:
        return _trapezoidal(self.probability_density(index), self.grid)


@dataclass(slots=True)
class WavefunctionTrajectory:
    grid: np.ndarray
    potential: np.ndarray
    times: np.ndarray
    wavefunctions: np.ndarray
    norms: np.ndarray
    units: str
    backend: str
    mass: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grid = np.ascontiguousarray(self.grid, dtype=np.float64)
        self.potential = np.ascontiguousarray(self.potential, dtype=np.float64)
        self.times = np.ascontiguousarray(self.times, dtype=np.float64)
        self.wavefunctions = np.ascontiguousarray(self.wavefunctions, dtype=np.complex128)
        self.norms = np.ascontiguousarray(self.norms, dtype=np.float64)
        if self.wavefunctions.shape != (len(self.times), len(self.grid)):
            raise ValueError("wavefunctions must have shape [num_frames, num_points]")
        if self.norms.shape != self.times.shape:
            raise ValueError("norms must have shape [num_frames]")

    def probability_density(self, frame: int = -1) -> np.ndarray:
        return np.abs(self.wavefunctions[frame]) ** 2

    def expectation_position(self, frame: int = -1) -> float:
        density = self.probability_density(frame)
        return _trapezoidal(self.grid * density, self.grid)


@dataclass(slots=True)
class ElectronicStructureResult:
    structure: StructureData
    energy_eV: float
    forces_eV_per_A: np.ndarray
    stress_eV_per_A3: np.ndarray | None = None
    dipole_eA: np.ndarray | None = None
    charges_e: np.ndarray | None = None
    magnetic_moments: np.ndarray | None = None
    wall_time_s: float = 0.0
    backend: str = "external"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AIMDThermoSeries:
    step: list[int] = field(default_factory=list)
    time_fs: list[float] = field(default_factory=list)
    potential_energy_eV: list[float] = field(default_factory=list)
    kinetic_energy_eV: list[float] = field(default_factory=list)
    total_energy_eV: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    max_force_eV_per_A: list[float] = field(default_factory=list)
    electronic_wall_time_s: list[float] = field(default_factory=list)

    def append(self, sample: dict[str, float | int]) -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).append(sample[name])

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def total_energy_drift_eV(self) -> float | None:
        if len(self.total_energy_eV) < 2:
            return None
        return float(self.total_energy_eV[-1] - self.total_energy_eV[0])


@dataclass(slots=True)
class AIMDResult:
    initial_structure: StructureData
    final_structure: StructureData
    completed_steps: int
    requested_steps: int
    wall_time_s: float
    thermo: AIMDThermoSeries
    trajectory_path: Path | None = None
    thermo_path: Path | None = None
    checkpoint_path: Path | None = None
    metadata_path: Path | None = None
    status: str = "completed"
    backend: str = "external"
    metadata: dict[str, Any] = field(default_factory=dict)
