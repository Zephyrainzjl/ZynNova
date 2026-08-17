from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import SafetyConfig
from .exceptions import SimulationDivergedError
from .results import ThermoSeries


THERMO_FIELDS = (
    "step",
    "time_fs",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
    "temperature_K",
    "volume_A3",
    "pressure_GPa",
    "max_force_eV_per_A",
)


def thermo_sample(atoms: Any, dynamics: Any) -> dict[str, float | int]:
    try:
        from ase import units
    except ImportError:
        units = None
    potential = float(atoms.get_potential_energy())
    kinetic = float(atoms.get_kinetic_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    volume = float(atoms.get_volume()) if np.any(atoms.pbc) else 0.0
    pressure = math.nan
    if np.any(atoms.pbc):
        try:
            stress = np.asarray(
                atoms.get_stress(include_ideal_gas=True, voigt=False), dtype=float
            )
            pressure_au = -float(np.trace(stress) / 3.0)
            pressure = pressure_au / units.GPa if units is not None else pressure_au
        except Exception:
            pressure = math.nan
    time_fs = float(dynamics.get_time() / units.fs) if units is not None else 0.0
    return {
        "step": int(dynamics.nsteps),
        "time_fs": time_fs,
        "potential_energy_eV": potential,
        "kinetic_energy_eV": kinetic,
        "total_energy_eV": potential + kinetic,
        "temperature_K": float(atoms.get_temperature()),
        "volume_A3": volume,
        "pressure_GPa": pressure,
        "max_force_eV_per_A": fmax,
    }


class ThermoRecorder:
    def __init__(
        self,
        atoms: Any,
        dynamics: Any,
        path: Path | None,
        *,
        append: bool = False,
        store_in_memory: bool = True,
    ) -> None:
        self.atoms = atoms
        self.dynamics = dynamics
        self.path = path
        self.store_in_memory = store_in_memory
        self.series = ThermoSeries()
        self._file = None
        self._writer = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.stat().st_size > 0
            self._file = path.open("a" if append else "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=THERMO_FIELDS)
            if not append or not exists:
                self._writer.writeheader()

    def __call__(self) -> None:
        sample = thermo_sample(self.atoms, self.dynamics)
        if self.store_in_memory:
            self.series.append(sample)
        if self._writer is not None:
            self._writer.writerow(sample)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class SafetyMonitor:
    def __init__(self, atoms: Any, dynamics: Any, config: SafetyConfig) -> None:
        self.atoms = atoms
        self.dynamics = dynamics
        self.config = config
        self._last_energy_per_atom: float | None = None

    def _fail(self, message: str) -> None:
        raise SimulationDivergedError(
            f"Safety check failed at step {self.dynamics.nsteps}: {message}"
        )

    def __call__(self) -> None:
        cfg = self.config
        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        forces = np.asarray(self.atoms.get_forces(), dtype=float)
        energy = float(self.atoms.get_potential_energy())
        temperature = float(self.atoms.get_temperature())
        if cfg.stop_on_nonfinite:
            values = np.concatenate([positions.ravel(), forces.ravel(), [energy, temperature]])
            if not np.all(np.isfinite(values)):
                self._fail("non-finite position, force, energy, or temperature")
        fmax = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
        if cfg.max_force_eV_per_A is not None and fmax > cfg.max_force_eV_per_A:
            self._fail(f"maximum force {fmax:.6g} eV/Å exceeds limit")
        if cfg.max_temperature_K is not None and temperature > cfg.max_temperature_K:
            self._fail(f"temperature {temperature:.6g} K exceeds limit")
        if cfg.min_temperature_K is not None and temperature < cfg.min_temperature_K:
            self._fail(f"temperature {temperature:.6g} K is below limit")
        per_atom = energy / max(len(self.atoms), 1)
        if cfg.max_energy_per_atom_eV is not None and abs(per_atom) > cfg.max_energy_per_atom_eV:
            self._fail(f"|potential energy/atom| {abs(per_atom):.6g} eV exceeds limit")
        if self._last_energy_per_atom is not None and cfg.max_energy_jump_per_atom_eV is not None:
            jump = abs(per_atom - self._last_energy_per_atom)
            if jump > cfg.max_energy_jump_per_atom_eV:
                self._fail(f"potential-energy jump/atom {jump:.6g} eV exceeds limit")
        self._last_energy_per_atom = per_atom
        if cfg.minimum_distance_A is not None and len(self.atoms) > 1:
            try:
                from ase.neighborlist import neighbor_list

                distances = neighbor_list("d", self.atoms, cfg.minimum_distance_A)
                if len(distances):
                    minimum = float(np.min(distances))
                    self._fail(
                        f"minimum interatomic distance {minimum:.6g} Å is below limit"
                    )
            except ImportError:
                distances = self.atoms.get_all_distances(
                    mic=bool(np.any(self.atoms.pbc))
                )
                distances[distances == 0] = np.inf
                minimum = float(distances.min())
                if minimum < cfg.minimum_distance_A:
                    self._fail(
                        f"minimum interatomic distance {minimum:.6g} Å is below limit"
                    )
        if np.any(self.atoms.pbc):
            volume = float(self.atoms.get_volume())
            if cfg.min_volume_A3 is not None and volume < cfg.min_volume_A3:
                self._fail(f"cell volume {volume:.6g} Å³ is below limit")
        if cfg.wrap_positions and np.any(self.atoms.pbc):
            self.atoms.wrap()


class CheckpointWriter:
    def __init__(
        self,
        atoms: Any,
        dynamics: Any,
        trajectory_path: Path,
        state_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.atoms = atoms
        self.dynamics = dynamics
        self.trajectory_path = trajectory_path
        self.state_path = state_path
        self.metadata = metadata

    def __call__(self) -> None:
        from ase.io import write

        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.trajectory_path.with_suffix(self.trajectory_path.suffix + ".tmp")
        write(temporary, self.atoms, format="traj")
        temporary.replace(self.trajectory_path)
        state = {
            **self.metadata,
            "completed_steps": int(self.dynamics.nsteps),
            "time": float(self.dynamics.get_time()),
        }
        temporary_state = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary_state.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary_state.replace(self.state_path)


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
