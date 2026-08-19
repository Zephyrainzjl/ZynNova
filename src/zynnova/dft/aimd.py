from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import to_ase_atoms, to_structure_data
from .backends import calculator_from_config, create_dft_calculator
from .config import AIMDConfig, ElectronicConfig
from .exceptions import (
    AIMDDivergedError,
    AIMDRestartError,
    DFTConfigurationError,
)
from .integrators import AIMDIntegrator, maxwell_boltzmann_velocities
from .results import AIMDResult, AIMDThermoSeries

_THERMO_FIELDS = (
    "step",
    "time_fs",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
    "temperature_K",
    "max_force_eV_per_A",
    "electronic_wall_time_s",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


class _ThermoWriter:
    def __init__(
        self,
        path: Path | None,
        *,
        append: bool,
        store_in_memory: bool,
    ) -> None:
        self.path = path
        self.series = AIMDThermoSeries()
        self.store_in_memory = store_in_memory
        self._file = None
        self._writer = None
        self.had_existing_data = False
        if path is not None:
            self.had_existing_data = path.exists() and path.stat().st_size > 0
            self._file = path.open("a" if append else "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=_THERMO_FIELDS)
            if not append or not self.had_existing_data:
                self._writer.writeheader()

    def write(
        self,
        sample: dict[str, float | int],
        *,
        write_file: bool = True,
    ) -> None:
        if self.store_in_memory:
            self.series.append(sample)
        if write_file and self._writer is not None:
            self._writer.writerow(sample)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _mobile_mask(atoms: Any, fixed_indices: tuple[int, ...]) -> np.ndarray:
    mobile = np.ones(len(atoms), dtype=bool)
    for index in fixed_indices:
        if index >= len(atoms):
            raise DFTConfigurationError(
                f"fixed atom index {index} is out of range for {len(atoms)} atoms"
            )
        mobile[index] = False
    for constraint in getattr(atoms, "constraints", ()):
        if type(constraint).__name__ != "FixAtoms":
            raise DFTConfigurationError(
                "The native AIMD integrator currently supports FixAtoms only; "
                f"received {type(constraint).__name__}"
            )
        for index in np.asarray(constraint.get_indices(), dtype=int):
            mobile[int(index)] = False
    if not np.any(mobile):
        raise DFTConfigurationError("at least one atom must remain mobile")
    return mobile


def _set_ase_velocities(atoms: Any, velocities_A_per_fs: np.ndarray) -> None:
    from ase import units

    atoms.set_velocities(np.asarray(velocities_A_per_fs) / units.fs)


def _get_ase_velocities(atoms: Any) -> np.ndarray | None:
    from ase import units

    velocities = atoms.get_velocities()
    if velocities is None:
        return None
    return np.asarray(velocities, dtype=np.float64) * units.fs


def _electronic_evaluation(atoms: Any) -> tuple[float, np.ndarray, float]:
    started = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    elapsed = time.perf_counter() - started
    if forces.shape != (len(atoms), 3):
        raise RuntimeError("electronic calculator returned forces with an invalid shape")
    return energy, forces, elapsed


def _sample(
    step: int,
    timestep_fs: float,
    energy: float,
    forces: np.ndarray,
    electronic_wall_time: float,
    integrator: AIMDIntegrator,
    remove_center_of_mass_dof: bool,
) -> dict[str, float | int]:
    kinetic = integrator.kinetic_energy_eV()
    temperature = integrator.temperature_K(remove_center_of_mass_dof)
    max_force = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    return {
        "step": int(step),
        "time_fs": float(step * timestep_fs),
        "potential_energy_eV": float(energy),
        "kinetic_energy_eV": kinetic,
        "total_energy_eV": float(energy + kinetic),
        "temperature_K": temperature,
        "max_force_eV_per_A": max_force,
        "electronic_wall_time_s": float(electronic_wall_time),
    }


def _check_safety(
    atoms: Any,
    sample: dict[str, float | int],
    forces: np.ndarray,
    config: Any,
    previous_energy_per_atom: float | None,
) -> float:
    step = int(sample["step"])

    def fail(message: str) -> None:
        raise AIMDDivergedError(f"AIMD safety check failed at step {step}: {message}")

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    energy = float(sample["potential_energy_eV"])
    temperature = float(sample["temperature_K"])
    if config.stop_on_nonfinite:
        flattened = np.concatenate([positions.ravel(), forces.ravel(), [energy, temperature]])
        if not np.all(np.isfinite(flattened)):
            fail("non-finite position, force, energy, or temperature")
    max_force = float(sample["max_force_eV_per_A"])
    if config.max_force_eV_per_A is not None and max_force > config.max_force_eV_per_A:
        fail(f"maximum force {max_force:.6g} eV/Å exceeds the configured limit")
    if config.max_temperature_K is not None and temperature > config.max_temperature_K:
        fail(f"temperature {temperature:.6g} K exceeds the configured limit")
    energy_per_atom = energy / max(len(atoms), 1)
    if previous_energy_per_atom is not None and config.max_energy_jump_per_atom_eV is not None:
        jump = abs(energy_per_atom - previous_energy_per_atom)
        if jump > config.max_energy_jump_per_atom_eV:
            fail(f"potential-energy jump {jump:.6g} eV/atom exceeds the limit")
    if config.min_distance_A is not None and len(atoms) > 1:
        distances = np.asarray(
            atoms.get_all_distances(mic=bool(np.any(atoms.pbc))),
            dtype=np.float64,
        )
        np.fill_diagonal(distances, np.inf)
        minimum = float(np.min(distances))
        if minimum < config.min_distance_A:
            fail(f"minimum interatomic distance {minimum:.6g} Å is below the limit")
    return energy_per_atom


def _write_checkpoint(
    path: Path,
    atoms: Any,
    integrator: AIMDIntegrator,
    mobile: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> None:
    if integrator.awaiting_forces:
        raise RuntimeError("refusing to checkpoint a half-completed AIMD step")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
            positions=np.asarray(atoms.get_positions(), dtype=np.float64),
            cell=np.asarray(atoms.cell.array, dtype=np.float64),
            pbc=np.asarray(atoms.pbc, dtype=bool),
            velocities_A_per_fs=integrator.velocities(),
            mobile=np.asarray(mobile, dtype=bool),
            step=np.asarray(integrator.step_index, dtype=np.int64),
            rng_state=np.asarray(integrator.rng_state),
            potential_energy_eV=np.asarray(energy, dtype=np.float64),
            forces_eV_per_A=np.asarray(forces, dtype=np.float64),
        )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    atoms: Any,
    integrator: AIMDIntegrator,
    expected_mobile: np.ndarray,
) -> int:
    if not path.exists():
        raise AIMDRestartError(f"AIMD checkpoint does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as state:
            atomic_numbers = np.asarray(state["atomic_numbers"], dtype=np.int64)
            if not np.array_equal(atomic_numbers, atoms.get_atomic_numbers()):
                raise AIMDRestartError(
                    "checkpoint atomic numbers do not match the requested structure"
                )
            mobile = np.asarray(state["mobile"], dtype=bool)
            if not np.array_equal(mobile, expected_mobile):
                raise AIMDRestartError(
                    "checkpoint mobile/fixed atoms do not match this configuration"
                )
            atoms.set_cell(np.asarray(state["cell"], dtype=np.float64))
            atoms.set_pbc(np.asarray(state["pbc"], dtype=bool))
            atoms.set_positions(
                np.asarray(state["positions"], dtype=np.float64),
                apply_constraint=False,
            )
            integrator.set_velocities(np.asarray(state["velocities_A_per_fs"], dtype=np.float64))
            integrator.step_index = int(state["step"])
            integrator.rng_state = str(state["rng_state"].item())
    except AIMDRestartError:
        raise
    except Exception as exc:
        raise AIMDRestartError(f"invalid AIMD checkpoint: {path}") from exc
    return integrator.step_index


class AIMDSession:
    """Restartable Born–Oppenheimer AIMD session with a native integrator."""

    def __init__(
        self,
        structure: Any,
        calculator: Any | None = None,
        config: AIMDConfig | None = None,
        *,
        electronic: ElectronicConfig | None = None,
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.structure = structure
        self.calculator = calculator
        self.config = config or AIMDConfig()
        self.electronic = electronic
        self.backend = backend
        self.backend_kwargs = dict(backend_kwargs or {})
        self.resume = resume

    @property
    def directory(self) -> Path:
        return Path(self.config.output.directory)

    @property
    def trajectory_path(self) -> Path:
        return self.directory / self.config.output.trajectory_filename

    @property
    def thermo_path(self) -> Path:
        return self.directory / self.config.output.thermo_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / self.config.output.checkpoint_filename

    @property
    def metadata_path(self) -> Path:
        return self.directory / self.config.output.metadata_filename

    def _prepare_directory(self) -> None:
        output = self.config.output
        if self.directory.exists() and not self.resume:
            has_content = any(self.directory.iterdir())
            if has_content and output.overwrite:
                shutil.rmtree(self.directory)
            elif has_content and not output.append:
                raise FileExistsError(
                    f"Output directory is not empty: {self.directory}. "
                    "Set output.overwrite=True or output.append=True."
                )
        self.directory.mkdir(parents=True, exist_ok=True)

    def _build_calculator(self, atoms: Any):
        if self.calculator is not None:
            return self.calculator, type(self.calculator).__name__
        if self.electronic is not None and self.backend is not None:
            raise ValueError("provide either electronic= or backend=, not both")
        if self.electronic is not None:
            return (
                calculator_from_config(self.electronic, structure=atoms),
                self.electronic.backend,
            )
        backend_name = self.backend or "pyscf"
        return (
            create_dft_calculator(backend_name, **self.backend_kwargs),
            backend_name,
        )

    def run(self) -> AIMDResult:
        self.config.validate()
        self._prepare_directory()
        atoms = to_ase_atoms(self.structure)
        initial_structure = to_structure_data(atoms)
        mobile = _mobile_mask(atoms, self.config.fixed_indices)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        integrator = AIMDIntegrator(
            masses,
            mobile=mobile,
            timestep_fs=self.config.timestep_fs,
            ensemble=self.config.ensemble.value,
            temperature_K=self.config.temperature_K,
            friction_per_fs=self.config.friction_per_fs,
            seed=self.config.random_seed,
            backend=self.config.integrator_backend,
        )
        calculator, electronic_backend = self._build_calculator(atoms)
        atoms.calc = calculator

        completed_before = 0
        if self.resume:
            completed_before = _load_checkpoint(self.checkpoint_path, atoms, integrator, mobile)
        else:
            existing_velocities = _get_ase_velocities(atoms)
            if self.config.initialize_velocities or existing_velocities is None:
                velocities = maxwell_boltzmann_velocities(
                    masses,
                    self.config.initial_temperature_K or self.config.temperature_K,
                    mobile=mobile,
                    seed=self.config.random_seed,
                    remove_center_of_mass=self.config.remove_center_of_mass,
                    exact_temperature=self.config.exact_initial_temperature,
                    backend=self.config.integrator_backend,
                )
            else:
                velocities = existing_velocities
                velocities[~mobile] = 0.0
            integrator.set_velocities(velocities)
        _set_ase_velocities(atoms, integrator.velocities())

        output = self.config.output
        append_outputs = self.resume or output.append
        thermo_writer = _ThermoWriter(
            self.thermo_path if output.log_interval > 0 else None,
            append=append_outputs,
            store_in_memory=output.store_in_memory,
        )
        trajectory = None
        if output.trajectory_interval > 0:
            from ase.io.trajectory import Trajectory

            trajectory = Trajectory(
                self.trajectory_path,
                "a" if append_outputs else "w",
                atoms,
            )

        status = "running"
        error: str | None = None
        started_at = time.time()
        run_started = time.perf_counter()
        completed = completed_before
        energy = float("nan")
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        previous_energy_per_atom: float | None = None
        metadata = {
            "status": status,
            "started_at_unix": started_at,
            "configuration": self.config.to_dict(),
            "electronic_backend": electronic_backend,
            "calculator": type(calculator).__qualname__,
            "integrator_backend": integrator.backend,
            "completed_before_restart": completed_before,
        }
        _write_json(self.metadata_path, metadata)

        try:
            energy, forces, electronic_time = _electronic_evaluation(atoms)
            initial_sample = _sample(
                completed,
                self.config.timestep_fs,
                energy,
                forces,
                electronic_time,
                integrator,
                self.config.remove_center_of_mass,
            )
            previous_energy_per_atom = _check_safety(
                atoms,
                initial_sample,
                forces,
                self.config.safety,
                None,
            )
            thermo_writer.write(
                initial_sample,
                write_file=not (self.resume and thermo_writer.had_existing_data),
            )
            if trajectory is not None and not self.resume:
                trajectory.write(atoms)
            _write_checkpoint(self.checkpoint_path, atoms, integrator, mobile, energy, forces)

            remaining = max(self.config.steps - completed_before, 0)
            for _ in range(remaining):
                snapshot_positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
                snapshot_velocities = integrator.velocities()
                snapshot_rng = integrator.rng_state
                snapshot_step = integrator.step_index
                new_positions = integrator.begin_step(snapshot_positions, forces)
                atoms.set_positions(new_positions, apply_constraint=False)
                if self.config.wrap_positions and np.any(atoms.pbc):
                    atoms.wrap()
                try:
                    new_energy, new_forces, electronic_time = _electronic_evaluation(atoms)
                except Exception:
                    atoms.set_positions(snapshot_positions, apply_constraint=False)
                    integrator = AIMDIntegrator(
                        masses,
                        mobile=mobile,
                        timestep_fs=self.config.timestep_fs,
                        ensemble=self.config.ensemble.value,
                        temperature_K=self.config.temperature_K,
                        friction_per_fs=self.config.friction_per_fs,
                        seed=self.config.random_seed,
                        backend=self.config.integrator_backend,
                    )
                    integrator.set_velocities(snapshot_velocities)
                    integrator.step_index = snapshot_step
                    integrator.rng_state = snapshot_rng
                    _set_ase_velocities(atoms, snapshot_velocities)
                    raise
                integrator.end_step(new_forces)
                energy, forces = new_energy, new_forces
                completed = integrator.step_index
                _set_ase_velocities(atoms, integrator.velocities())
                sample = _sample(
                    completed,
                    self.config.timestep_fs,
                    energy,
                    forces,
                    electronic_time,
                    integrator,
                    self.config.remove_center_of_mass,
                )
                if completed % self.config.safety.check_interval == 0:
                    previous_energy_per_atom = _check_safety(
                        atoms,
                        sample,
                        forces,
                        self.config.safety,
                        previous_energy_per_atom,
                    )
                else:
                    previous_energy_per_atom = energy / max(len(atoms), 1)
                if output.log_interval > 0 and completed % output.log_interval == 0:
                    thermo_writer.write(sample)
                if trajectory is not None and completed % output.trajectory_interval == 0:
                    trajectory.write(atoms)
                if output.checkpoint_interval > 0 and completed % output.checkpoint_interval == 0:
                    _write_checkpoint(
                        self.checkpoint_path,
                        atoms,
                        integrator,
                        mobile,
                        energy,
                        forces,
                    )
            _write_checkpoint(self.checkpoint_path, atoms, integrator, mobile, energy, forces)
            status = "completed"
        except AIMDDivergedError as exc:
            status = "diverged"
            error = str(exc)
            if not integrator.awaiting_forces and np.isfinite(energy):
                _write_checkpoint(self.checkpoint_path, atoms, integrator, mobile, energy, forces)
            raise
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            if not integrator.awaiting_forces and np.isfinite(energy):
                _write_checkpoint(self.checkpoint_path, atoms, integrator, mobile, energy, forces)
            raise
        finally:
            wall_time = time.perf_counter() - run_started
            if trajectory is not None:
                trajectory.close()
            thermo_writer.close()
            metadata.update(
                {
                    "status": status,
                    "error": error,
                    "completed_steps": int(completed),
                    "wall_time_s": wall_time,
                    "finished_at_unix": time.time(),
                }
            )
            _write_json(self.metadata_path, metadata)

        return AIMDResult(
            initial_structure=initial_structure,
            final_structure=to_structure_data(atoms),
            completed_steps=completed,
            requested_steps=self.config.steps,
            wall_time_s=wall_time,
            thermo=thermo_writer.series,
            trajectory_path=(self.trajectory_path if output.trajectory_interval > 0 else None),
            thermo_path=self.thermo_path if output.log_interval > 0 else None,
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            status=status,
            backend=electronic_backend,
            metadata=metadata,
        )


def run_aimd(
    structure: Any,
    calculator: Any | None = None,
    config: AIMDConfig | None = None,
    *,
    electronic: ElectronicConfig | None = None,
    backend: str | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    resume: bool = False,
) -> AIMDResult:
    """Run Born–Oppenheimer AIMD using forces from an electronic calculator."""
    return AIMDSession(
        structure,
        calculator,
        config,
        electronic=electronic,
        backend=backend,
        backend_kwargs=backend_kwargs,
        resume=resume,
    ).run()
