from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ....dynamics.adapters import to_ase_atoms
from .materials import _resolve_calculator


def _optimizer_class(name: str) -> Any:
    try:
        from ase.optimize import BFGS, FIRE, LBFGS, MDMin
    except ImportError as exc:
        raise ImportError("rare-event workflows require ASE; install zynnova[mlff]") from exc
    optimizers = {
        "bfgs": BFGS,
        "fire": FIRE,
        "lbfgs": LBFGS,
        "mdmin": MDMin,
    }
    try:
        return optimizers[name.lower()]
    except KeyError as exc:
        raise ValueError(f"optimizer must be one of {sorted(optimizers)}") from exc


@dataclass(slots=True)
class NEBConfig:
    n_images: int = 9
    spring_eV_per_A: float = 0.10
    interpolation: Literal["linear", "idpp"] = "idpp"
    method: str = "improvedtangent"
    mic: bool = True
    climb: bool = True
    preclimb_steps: int = 100
    fmax_eV_per_A: float = 0.03
    max_steps: int = 600
    optimizer: str = "fire"
    trajectory_filename: str | None = "neb.traj"
    logfile: str | None = "neb.log"

    def __post_init__(self) -> None:
        if self.n_images < 3:
            raise ValueError("NEB requires at least three images")
        if self.spring_eV_per_A <= 0:
            raise ValueError("NEB spring constant must be positive")
        if self.interpolation not in {"linear", "idpp"}:
            raise ValueError("interpolation must be 'linear' or 'idpp'")
        if self.preclimb_steps < 0 or self.max_steps < 1:
            raise ValueError("NEB step counts must be non-negative")
        if self.preclimb_steps > self.max_steps:
            raise ValueError("preclimb_steps cannot exceed max_steps")
        if self.fmax_eV_per_A <= 0:
            raise ValueError("fmax_eV_per_A must be positive")


@dataclass(slots=True)
class NEBResult:
    images: tuple[Any, ...]
    energies_eV: np.ndarray
    relative_energies_eV: np.ndarray
    path_coordinate_A: np.ndarray
    forward_barrier_eV: float
    reverse_barrier_eV: float
    reaction_energy_eV: float
    saddle_image_index: int
    converged: bool
    climb: bool
    output_directory: Path

    @property
    def saddle_structure(self) -> Any:
        return self.images[self.saddle_image_index]

    def plot(self, *, ax: Any | None = None) -> Any:
        try:
            from ase.mep import NEBTools
        except ImportError as exc:
            raise ImportError("NEB plotting requires ASE") from exc
        return NEBTools(list(self.images)).plot_band(ax=ax)


class JouleWeaveNEB:
    """Two-stage ASE NEB/CI-NEB with IDPP and explicit barrier accounting."""

    def __init__(
        self,
        potential: Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
    ) -> None:
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.compile_model = bool(compile_model)

    def _calculator(self) -> Any:
        return _resolve_calculator(
            self.potential,
            require_stress=False,
            device=self.device,
            dtype=self.dtype,
            compile_model=self.compile_model,
        )

    @staticmethod
    def _path_coordinate(images: Sequence[Any], mic: bool) -> np.ndarray:
        coordinate = [0.0]
        for previous, current in zip(images[:-1], images[1:], strict=True):
            delta = np.asarray(current.positions) - np.asarray(previous.positions)
            if mic and bool(np.any(current.pbc)):
                from ase.geometry import find_mic

                delta, _lengths = find_mic(delta, current.cell, current.pbc)
            rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
            coordinate.append(coordinate[-1] + rms)
        return np.asarray(coordinate, dtype=float)

    def run(
        self,
        initial: Any,
        final: Any,
        *,
        config: NEBConfig | None = None,
        output_directory: str | Path = "jouleweave-neb",
    ) -> NEBResult:
        config = config or NEBConfig()
        initial_atoms = to_ase_atoms(initial).copy()
        final_atoms = to_ase_atoms(final).copy()
        if len(initial_atoms) != len(final_atoms):
            raise ValueError("NEB endpoints must contain the same number of atoms")
        if not np.array_equal(
            initial_atoms.get_atomic_numbers(),
            final_atoms.get_atomic_numbers(),
        ):
            raise ValueError("NEB endpoint atom ordering and species must match")
        if not np.array_equal(initial_atoms.pbc, final_atoms.pbc):
            raise ValueError("NEB endpoint PBC flags must match")
        if not np.allclose(initial_atoms.cell.array, final_atoms.cell.array):
            raise ValueError("standard NEB requires identical endpoint cells")

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        images = [initial_atoms]
        images.extend(initial_atoms.copy() for _ in range(config.n_images - 2))
        images.append(final_atoms)
        # ASE evaluates this serial band one image at a time. Sharing one
        # calculator avoids loading n_images copies of a checkpoint into GPU
        # memory and is explicitly enabled on the NEB object below.
        calculator = self._calculator()
        for image in images:
            image.calc = calculator
        shared_calculator = True
        try:
            from ase.mep import NEB
        except ImportError as exc:
            raise ImportError("NEB requires ASE; install zynnova[mlff]") from exc
        start_with_climb = bool(config.climb and config.preclimb_steps == 0)
        neb = NEB(
            images,
            k=config.spring_eV_per_A,
            climb=start_with_climb,
            method=config.method,
            allow_shared_calculator=shared_calculator,
        )
        neb.interpolate(
            method=config.interpolation,
            mic=config.mic,
            apply_constraint=False,
        )
        optimizer_cls = _optimizer_class(config.optimizer)
        converged = False
        if config.climb and config.preclimb_steps:
            warmup = optimizer_cls(
                neb,
                trajectory=str(output / "neb-preclimb.traj"),
                logfile=str(output / "neb-preclimb.log"),
            )
            warmup.run(
                fmax=config.fmax_eV_per_A,
                steps=config.preclimb_steps,
            )
            neb.climb = True
        remaining_steps = config.max_steps - (config.preclimb_steps if config.climb else 0)
        if remaining_steps:
            trajectory = (
                None
                if config.trajectory_filename is None
                else str(output / config.trajectory_filename)
            )
            logfile = None if config.logfile is None else str(output / config.logfile)
            optimizer = optimizer_cls(
                neb,
                trajectory=trajectory,
                logfile=logfile,
            )
            converged = bool(
                optimizer.run(
                    fmax=config.fmax_eV_per_A,
                    steps=remaining_steps,
                )
            )

        energies = np.asarray(
            [float(image.get_potential_energy()) for image in images],
            dtype=float,
        )
        relative = energies - energies[0]
        saddle_index = int(np.argmax(energies))
        saddle_energy = float(energies[saddle_index])
        return NEBResult(
            images=tuple(images),
            energies_eV=energies,
            relative_energies_eV=relative,
            path_coordinate_A=self._path_coordinate(images, config.mic),
            forward_barrier_eV=saddle_energy - float(energies[0]),
            reverse_barrier_eV=saddle_energy - float(energies[-1]),
            reaction_energy_eV=float(energies[-1] - energies[0]),
            saddle_image_index=saddle_index,
            converged=converged,
            climb=bool(config.climb),
            output_directory=output,
        )


@dataclass(slots=True)
class DimerConfig:
    fmax_eV_per_A: float = 0.03
    max_steps: int = 600
    displacement_A: float = 0.05
    dimer_separation_A: float = 0.01
    maximum_translation_A: float = 0.10
    seed: int = 42
    trajectory_filename: str | None = "dimer.traj"
    logfile: str | None = "dimer.log"
    eigenmode_logfile: str | None = "dimer-eigenmode.log"

    def __post_init__(self) -> None:
        if self.fmax_eV_per_A <= 0 or self.max_steps < 1:
            raise ValueError("invalid dimer convergence controls")
        if self.displacement_A <= 0 or self.dimer_separation_A <= 0:
            raise ValueError("dimer displacement and separation must be positive")
        if self.maximum_translation_A <= 0:
            raise ValueError("maximum_translation_A must be positive")


@dataclass(slots=True)
class DimerResult:
    initial_structure: Any
    saddle_structure: Any
    initial_energy_eV: float
    saddle_energy_eV: float
    barrier_eV: float
    curvature_eV_per_A2: float | None
    converged: bool
    output_directory: Path


class JouleWeaveDimer:
    """ASE minimum-mode dimer saddle search from one local minimum."""

    def __init__(
        self,
        potential: Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
    ) -> None:
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.compile_model = bool(compile_model)

    def run(
        self,
        structure: Any,
        *,
        migrating_index: int | None = None,
        direction: Sequence[float] | None = None,
        displacement_vector: Any | None = None,
        movable_mask: Sequence[bool] | None = None,
        config: DimerConfig | None = None,
        output_directory: str | Path = "jouleweave-dimer",
    ) -> DimerResult:
        config = config or DimerConfig()
        atoms = to_ase_atoms(structure).copy()
        atoms.calc = _resolve_calculator(
            self.potential,
            require_stress=False,
            device=self.device,
            dtype=self.dtype,
            compile_model=self.compile_model,
        )
        initial = atoms.copy()
        initial.calc = atoms.calc
        initial_energy = float(atoms.get_potential_energy())
        if displacement_vector is None:
            if migrating_index is None or direction is None:
                raise ValueError("provide displacement_vector or migrating_index plus direction")
            if not 0 <= int(migrating_index) < len(atoms):
                raise IndexError("migrating_index is out of range")
            unit = np.asarray(direction, dtype=float).reshape(3)
            norm = float(np.linalg.norm(unit))
            if norm <= 0:
                raise ValueError("dimer direction cannot be zero")
            displacement = np.zeros((len(atoms), 3), dtype=float)
            displacement[int(migrating_index)] = config.displacement_A * unit / norm
        else:
            displacement = np.asarray(displacement_vector, dtype=float)
            if displacement.shape != (len(atoms), 3):
                raise ValueError("displacement_vector must have shape [atoms, 3]")
        if movable_mask is None:
            movable = np.linalg.norm(displacement, axis=1) > 0
        else:
            movable = np.asarray(movable_mask, dtype=bool).reshape(-1)
            if movable.shape != (len(atoms),):
                raise ValueError("movable_mask must contain one flag per atom")

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        try:
            from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
        except ImportError as exc:
            raise ImportError("dimer search requires ASE; install zynnova[mlff]") from exc
        control_log = None if config.logfile is None else str(output / "dimer-control.log")
        eigenmode_log = (
            None if config.eigenmode_logfile is None else str(output / config.eigenmode_logfile)
        )
        with DimerControl(
            initial_eigenmode_method="displacement",
            displacement_method="vector",
            dimer_separation=config.dimer_separation_A,
            maximum_translation=config.maximum_translation_A,
            mask=movable.tolist(),
            logfile=control_log,
            eigenmode_logfile=eigenmode_log,
        ) as control:
            dimer_atoms = MinModeAtoms(
                atoms,
                control,
                random_seed=config.seed,
            )
            dimer_atoms.displace(displacement_vector=displacement)
            trajectory = (
                None
                if config.trajectory_filename is None
                else str(output / config.trajectory_filename)
            )
            logfile = None if config.logfile is None else str(output / config.logfile)
            with MinModeTranslate(
                dimer_atoms,
                trajectory=trajectory,
                logfile=logfile,
            ) as optimizer:
                converged = bool(
                    optimizer.run(
                        fmax=config.fmax_eV_per_A,
                        steps=config.max_steps,
                    )
                )
            saddle_energy = float(atoms.get_potential_energy())
            curvature_method = getattr(dimer_atoms, "get_curvature", None)
            curvature = float(curvature_method()) if callable(curvature_method) else None
        return DimerResult(
            initial_structure=initial,
            saddle_structure=atoms.copy(),
            initial_energy_eV=initial_energy,
            saddle_energy_eV=saddle_energy,
            barrier_eV=saddle_energy - initial_energy,
            curvature_eV_per_A2=curvature,
            converged=converged,
            output_directory=output,
        )


def well_tempered_metadynamics_input(
    collective_variable_actions: Sequence[str],
    *,
    arguments: Sequence[str],
    sigma: Sequence[float],
    height_eV: float,
    pace: int,
    temperature_K: float,
    bias_factor: float,
    print_stride: int = 10,
    hills_file: str = "HILLS",
    colvar_file: str = "COLVAR",
) -> tuple[str, ...]:
    """Build a PLUMED well-tempered METAD block in Å, fs, eV, and kelvin."""

    if not arguments or len(arguments) != len(sigma):
        raise ValueError("arguments and sigma must have the same non-zero length")
    if height_eV <= 0 or pace < 1 or temperature_K <= 0 or bias_factor <= 1:
        raise ValueError("invalid well-tempered metadynamics parameters")
    argument_text = ",".join(arguments)
    sigma_text = ",".join(f"{float(value):.12g}" for value in sigma)
    actions = ["UNITS LENGTH=A TIME=fs ENERGY=eV"]
    actions.extend(str(value) for value in collective_variable_actions)
    actions.append(
        "jw_bias: METAD "
        f"ARG={argument_text} SIGMA={sigma_text} HEIGHT={float(height_eV):.12g} "
        f"PACE={int(pace)} BIASFACTOR={float(bias_factor):.12g} "
        f"TEMP={float(temperature_K):.12g} FILE={hills_file}"
    )
    actions.append(
        f"PRINT ARG={argument_text},jw_bias.bias STRIDE={int(print_stride)} FILE={colvar_file}"
    )
    return tuple(actions)


@dataclass(slots=True)
class MetadynamicsConfig:
    plumed_input: tuple[str, ...]
    steps: int = 100_000
    timestep_fs: float = 1.0
    temperature_K: float = 600.0
    ensemble: Literal["nvt", "nve"] = "nvt"
    friction_per_fs: float = 0.01
    initialize_velocities: bool = True
    seed: int = 42
    trajectory_interval: int = 100
    trajectory_filename: str = "metadynamics.traj"
    plumed_log: str = "plumed.log"
    restart: bool = False
    use_charge: bool = False
    update_charge: bool = False

    def __post_init__(self) -> None:
        self.plumed_input = tuple(str(value) for value in self.plumed_input)
        if not self.plumed_input:
            raise ValueError("plumed_input cannot be empty")
        if self.steps < 1 or self.timestep_fs <= 0 or self.temperature_K <= 0:
            raise ValueError("invalid metadynamics run controls")
        if self.ensemble not in {"nvt", "nve"}:
            raise ValueError("ensemble must be 'nvt' or 'nve'")
        if self.friction_per_fs < 0 or self.trajectory_interval < 1:
            raise ValueError("invalid friction or trajectory interval")
        if self.update_charge and not self.use_charge:
            raise ValueError("update_charge=True requires use_charge=True")


@dataclass(slots=True)
class MetadynamicsResult:
    final_structure: Any
    steps: int
    output_directory: Path
    trajectory_path: Path
    hills_path: Path | None
    colvar_path: Path | None
    plumed_input: tuple[str, ...] = field(repr=False)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class JouleWeaveMetadynamics:
    """ASE-PLUMED enhanced sampling driven by a JouleWeave calculator."""

    def __init__(
        self,
        potential: Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
    ) -> None:
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.compile_model = bool(compile_model)

    def run(
        self,
        structure: Any,
        *,
        config: MetadynamicsConfig,
        output_directory: str | Path = "jouleweave-metadynamics",
    ) -> MetadynamicsResult:
        atoms = to_ase_atoms(structure).copy()
        base_calculator = _resolve_calculator(
            self.potential,
            require_stress=False,
            device=self.device,
            dtype=self.dtype,
            compile_model=self.compile_model,
        )
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        try:
            from ase import units
            from ase.calculators.plumed import Plumed
            from ase.io.trajectory import Trajectory
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            from ase.md.verlet import VelocityVerlet
        except ImportError as exc:
            raise ImportError(
                "metadynamics requires ASE and the PLUMED Python wrapper "
                "(for example conda install -c conda-forge py-plumed)"
            ) from exc
        timestep = config.timestep_fs * units.fs
        atoms.calc = Plumed(
            calc=base_calculator,
            input=list(config.plumed_input),
            timestep=timestep,
            atoms=atoms,
            kT=units.kB * config.temperature_K,
            log=str(output / config.plumed_log),
            restart=config.restart,
            use_charge=config.use_charge,
            update_charge=config.update_charge,
        )
        if config.initialize_velocities and not config.restart:
            rng = np.random.RandomState(config.seed)
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=config.temperature_K,
                rng=rng,
            )
        if config.ensemble == "nvt":
            dynamics = Langevin(
                atoms,
                timestep,
                temperature_K=config.temperature_K,
                friction=config.friction_per_fs / units.fs,
            )
        else:
            dynamics = VelocityVerlet(atoms, timestep)
        trajectory_path = output / config.trajectory_filename
        trajectory = Trajectory(str(trajectory_path), "a" if config.restart else "w", atoms)
        dynamics.attach(trajectory.write, interval=config.trajectory_interval)
        with _working_directory(output):
            dynamics.run(config.steps)
        trajectory.close()
        hills = output / "HILLS"
        colvar = output / "COLVAR"
        return MetadynamicsResult(
            final_structure=atoms.copy(),
            steps=config.steps,
            output_directory=output,
            trajectory_path=trajectory_path,
            hills_path=hills if hills.is_file() else None,
            colvar_path=colvar if colvar.is_file() else None,
            plumed_input=config.plumed_input,
        )


__all__ = [
    "DimerConfig",
    "DimerResult",
    "JouleWeaveDimer",
    "JouleWeaveMetadynamics",
    "JouleWeaveNEB",
    "MetadynamicsConfig",
    "MetadynamicsResult",
    "NEBConfig",
    "NEBResult",
    "well_tempered_metadynamics_input",
]
