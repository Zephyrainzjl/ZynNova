from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zynnova.structure import StructureData


@dataclass(slots=True)
class ThermoSeries:
    step: list[int] = field(default_factory=list)
    time_fs: list[float] = field(default_factory=list)
    potential_energy_eV: list[float] = field(default_factory=list)
    kinetic_energy_eV: list[float] = field(default_factory=list)
    total_energy_eV: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    volume_A3: list[float] = field(default_factory=list)
    pressure_GPa: list[float] = field(default_factory=list)
    max_force_eV_per_A: list[float] = field(default_factory=list)

    def append(self, sample: dict[str, float | int]) -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).append(sample[name])

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def energy_drift_eV_per_atom(self) -> float | None:
        if len(self.total_energy_eV) < 2:
            return None
        return float(self.total_energy_eV[-1] - self.total_energy_eV[0])


@dataclass(slots=True)
class RelaxationResult:
    initial_structure: StructureData
    final_structure: StructureData
    converged: bool
    steps: int
    initial_energy_eV: float
    final_energy_eV: float
    final_fmax_eV_per_A: float
    final_smax_eV_per_A3: float | None
    wall_time_s: float
    trajectory_path: Path | None = None
    logfile_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    initial_structure: StructureData
    final_structure: StructureData
    completed_steps: int
    requested_steps: int
    wall_time_s: float
    thermo: ThermoSeries
    trajectory_path: Path | None = None
    thermo_path: Path | None = None
    checkpoint_path: Path | None = None
    metadata_path: Path | None = None
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)
