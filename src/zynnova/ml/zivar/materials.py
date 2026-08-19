"""Materials workflows driven by the single ZIVAR energy surface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calculator import zivar_calculator


@dataclass(frozen=True, slots=True)
class RelaxationResult:
    final_atoms: Any
    converged: bool
    steps: int
    trajectory: Path
    logfile: Path


def relax_structure(
    structure: Any,
    potential: Any,
    *,
    fmax_eV_per_A: float = 0.03,
    steps: int = 500,
    relax_cell: bool = True,
    pressure_eV_per_A3: float = 0.0,
    output_directory: str | Path = "zivar-relax",
    device: str = "cpu",
    dtype: str | None = None,
) -> RelaxationResult:
    from ase.filters import FrechetCellFilter
    from ase.io import Trajectory
    from ase.optimize import FIRE

    if fmax_eV_per_A <= 0 or steps < 1:
        raise ValueError("invalid relaxation settings")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atoms = structure.copy()
    atoms.calc = zivar_calculator(potential, device=device, dtype=dtype)
    target = (
        FrechetCellFilter(atoms, scalar_pressure=pressure_eV_per_A3)
        if relax_cell and bool(atoms.pbc.all())
        else atoms
    )
    trajectory_path, logfile = output / "relaxation.traj", output / "relaxation.log"
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    optimizer = FIRE(target, logfile=str(logfile))
    optimizer.attach(trajectory.write, interval=1)
    converged = bool(optimizer.run(fmax=fmax_eV_per_A, steps=steps))
    trajectory.close()
    return RelaxationResult(
        atoms.copy(), converged, int(optimizer.nsteps), trajectory_path, logfile
    )


@dataclass(frozen=True, slots=True)
class MolecularDynamicsResult:
    final_atoms: Any
    ensemble: str
    steps: int
    timestep_fs: float
    temperature_K: float
    wall_time_s: float
    trajectory: Path
    logfile: Path


def run_molecular_dynamics(
    structure: Any,
    potential: Any,
    *,
    ensemble: str = "nvt",
    temperature_K: float = 800.0,
    timestep_fs: float = 1.0,
    steps: int = 20_000,
    pressure_bar: float = 1.0,
    friction_per_fs: float = 0.01,
    trajectory_interval: int = 10,
    output_directory: str | Path = "zivar-md",
    device: str = "cpu",
    dtype: str | None = None,
    seed: int = 42,
) -> MolecularDynamicsResult:
    from ase import units
    from ase.io import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet

    if ensemble not in {"nve", "nvt", "npt"}:
        raise ValueError("ensemble must be nve, nvt, or npt")
    if temperature_K < 0 or timestep_fs <= 0 or steps < 1 or trajectory_interval < 1:
        raise ValueError("invalid MD settings")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path, logfile = output / "md.traj", output / "md.log"
    atoms = structure.copy()
    atoms.calc = zivar_calculator(potential, device=device, dtype=dtype)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed)
    )
    Stationary(atoms)
    timestep = timestep_fs * units.fs
    if ensemble == "nve":
        dynamics = VelocityVerlet(atoms, timestep=timestep)
    elif ensemble == "nvt":
        dynamics = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            friction=friction_per_fs / units.fs,
        )
    else:
        if not bool(atoms.pbc.all()):
            raise ValueError("NPT requires full periodicity")
        dynamics = NPTBerendsen(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            taut=100.0 * units.fs,
            pressure_au=pressure_bar * units.bar,
            taup=1000.0 * units.fs,
            compressibility_au=4.57e-5 / units.bar,
        )
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    dynamics.attach(trajectory.write, interval=trajectory_interval)
    def write_log() -> None:
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{dynamics.nsteps} {atoms.get_temperature():.8g} "
                f"{atoms.get_potential_energy():.12g}\n"
            )

    dynamics.attach(write_log, interval=trajectory_interval)
    started = time.perf_counter()
    dynamics.run(steps)
    elapsed = time.perf_counter() - started
    trajectory.close()
    return MolecularDynamicsResult(
        atoms.copy(), ensemble, steps, timestep_fs, temperature_K, elapsed,
        trajectory_path, logfile
    )


@dataclass(frozen=True, slots=True)
class EquationOfStateResult:
    volumes_A3: np.ndarray
    energies_eV: np.ndarray
    equilibrium_volume_A3: float
    equilibrium_energy_eV: float
    bulk_modulus_GPa: float


def equation_of_state(
    structure: Any,
    potential: Any,
    *,
    linear_scales: Any = None,
    eos: str = "birchmurnaghan",
    device: str = "cpu",
    dtype: str | None = None,
) -> EquationOfStateResult:
    from ase.eos import EquationOfState

    if not bool(structure.pbc.all()):
        raise ValueError("EOS requires full periodicity")
    scales = np.asarray(
        np.linspace(0.94, 1.06, 7) if linear_scales is None else linear_scales,
        dtype=float,
    )
    if scales.ndim != 1 or scales.size < 4 or np.any(scales <= 0):
        raise ValueError("linear_scales requires at least four positive values")
    calculator = zivar_calculator(potential, device=device, dtype=dtype)
    volumes, energies = [], []
    for scale in scales:
        atoms = structure.copy()
        atoms.set_cell(np.asarray(structure.cell.array) * scale, scale_atoms=True)
        atoms.calc = calculator
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())
    volume, energy, bulk = EquationOfState(volumes, energies, eos=eos).fit()
    return EquationOfStateResult(
        np.asarray(volumes), np.asarray(energies), float(volume), float(energy),
        float(bulk) * 160.21766208
    )


@dataclass(frozen=True, slots=True)
class ElasticTensorResult:
    stiffness_GPa: np.ndarray
    compliance_per_GPa: np.ndarray


def _voigt_strain(index: int, magnitude: float) -> np.ndarray:
    result = np.zeros((3, 3))
    if index < 3:
        result[index, index] = magnitude
    else:
        row, column = ((1, 2), (0, 2), (0, 1))[index - 3]
        result[row, column] = result[column, row] = 0.5 * magnitude
    return result


def calculate_elastic_tensor(
    structure: Any,
    potential: Any,
    *,
    strain: float = 5.0e-3,
    device: str = "cpu",
    dtype: str | None = None,
) -> ElasticTensorResult:
    if not bool(structure.pbc.all()) or not 1.0e-5 <= strain <= 5.0e-2:
        raise ValueError("elastic tensor requires periodicity and valid strain")
    calculator = zivar_calculator(potential, device=device, dtype=dtype)
    reference = np.asarray(structure.cell.array)
    stiffness = np.empty((6, 6))
    for column in range(6):
        sampled = []
        for sign in (-1.0, 1.0):
            atoms = structure.copy()
            atoms.set_cell(
                reference @ (np.eye(3) + _voigt_strain(column, sign * strain)),
                scale_atoms=True,
            )
            atoms.calc = calculator
            sampled.append(atoms.get_stress(voigt=True))
        stiffness[:, column] = (np.asarray(sampled[1]) - sampled[0]) / (2.0 * strain)
    stiffness = 0.5 * (stiffness + stiffness.T) * 160.21766208
    return ElasticTensorResult(stiffness, np.linalg.pinv(stiffness, rcond=1.0e-12))


@dataclass(frozen=True, slots=True)
class PhononResult:
    gamma_frequencies_eV: np.ndarray
    force_constants: np.ndarray
    working_directory: Path


def calculate_phonons(
    structure: Any,
    potential: Any,
    *,
    supercell: tuple[int, int, int] = (2, 2, 2),
    displacement_A: float = 0.01,
    output_directory: str | Path = "zivar-phonons",
    device: str = "cpu",
    dtype: str | None = None,
) -> PhononResult:
    from ase.phonons import Phonons

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    phonons = Phonons(
        structure.copy(),
        zivar_calculator(potential, device=device, dtype=dtype),
        supercell=supercell,
        delta=displacement_A,
        name=str(output / "displacements"),
    )
    phonons.run()
    phonons.read(acoustic=True)
    frequencies = np.asarray(phonons.band_structure([[0.0, 0.0, 0.0]]))[0]
    return PhononResult(frequencies, np.asarray(phonons.get_force_constant()), output)


@dataclass(frozen=True, slots=True)
class NEBResult:
    images: tuple[Any, ...]
    converged: bool
    energies_eV: np.ndarray
    barrier_eV: float


def run_neb(
    initial: Any,
    final: Any,
    potential: Any,
    *,
    image_count: int = 7,
    fmax_eV_per_A: float = 0.05,
    steps: int = 500,
    climb: bool = True,
    output_directory: str | Path = "zivar-neb",
    device: str = "cpu",
    dtype: str | None = None,
) -> NEBResult:
    from ase.mep import NEB
    from ase.optimize import FIRE

    if image_count < 3:
        raise ValueError("NEB requires at least three images")
    images = [initial.copy()] + [initial.copy() for _ in range(image_count - 2)] + [final.copy()]
    neb = NEB(images, climb=climb)
    neb.interpolate(method="idpp")
    for image in images:
        image.calc = zivar_calculator(potential, device=device, dtype=dtype)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    optimizer = FIRE(neb, logfile=str(output / "neb.log"), trajectory=str(output / "neb.traj"))
    converged = bool(optimizer.run(fmax=fmax_eV_per_A, steps=steps))
    energies = np.asarray([image.get_potential_energy() for image in images])
    return NEBResult(
        tuple(image.copy() for image in images),
        converged,
        energies,
        float(energies.max() - energies[0]),
    )


__all__ = [
    "ElasticTensorResult", "EquationOfStateResult", "MolecularDynamicsResult",
    "NEBResult", "PhononResult", "RelaxationResult", "calculate_elastic_tensor",
    "calculate_phonons", "equation_of_state", "relax_structure",
    "run_molecular_dynamics", "run_neb",
]
