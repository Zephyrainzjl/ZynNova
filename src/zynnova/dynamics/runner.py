from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .adapters import to_ase_atoms, to_structure_data
from .config import RunConfig
from .ensembles import build_dynamics, initialize_velocities
from .exceptions import MissingBackendError, RestartError, SimulationDivergedError
from .observers import CheckpointWriter, SafetyMonitor, ThermoRecorder, write_metadata
from .results import SimulationResult


class DynamicsSession:
    """Stateful, restartable molecular-dynamics session."""

    def __init__(
        self,
        structure: Any,
        calculator: Any,
        config: RunConfig | None = None,
        *,
        resume: bool = False,
    ) -> None:
        self.structure = structure
        self.calculator = calculator
        self.config = config or RunConfig()
        self.config.validate()
        self.resume = resume
        self.directory = Path(self.config.output.directory)

    @property
    def trajectory_path(self) -> Path:
        return self.directory / self.config.output.trajectory_filename

    @property
    def thermo_path(self) -> Path:
        return self.directory / self.config.output.thermo_filename

    @property
    def metadata_path(self) -> Path:
        return self.directory / self.config.output.metadata_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / self.config.output.checkpoint_filename

    @property
    def checkpoint_state_path(self) -> Path:
        return self.directory / self.config.output.checkpoint_state_filename

    def _prepare_directory(self) -> None:
        if self.directory.exists() and not self.resume:
            has_content = any(self.directory.iterdir())
            if has_content and self.config.output.overwrite:
                shutil.rmtree(self.directory)
            elif has_content and not self.config.output.append:
                raise FileExistsError(
                    f"Output directory is not empty: {self.directory}. "
                    "Set output.overwrite=True or output.append=True."
                )
        self.directory.mkdir(parents=True, exist_ok=True)

    def _load_atoms(self):
        if not self.resume:
            return to_ase_atoms(self.structure), 0
        if not self.checkpoint_path.exists() or not self.checkpoint_state_path.exists():
            raise RestartError("resume=True but checkpoint files are missing")
        try:
            from ase.io import read
        except ImportError as exc:
            raise MissingBackendError("ASE is required to restore checkpoints") from exc
        atoms = read(self.checkpoint_path)
        try:
            state = json.loads(self.checkpoint_state_path.read_text(encoding="utf-8"))
            completed = int(state.get("completed_steps", 0))
        except Exception as exc:
            raise RestartError("checkpoint state is invalid") from exc
        return atoms, completed

    def run(self) -> SimulationResult:
        self._prepare_directory()
        atoms, completed_before = self._load_atoms()
        if completed_before > self.config.md.steps:
            raise RestartError(
                "checkpoint completed_steps exceeds the requested total MD steps"
            )
        initial = to_structure_data(atoms)
        atoms.calc = self.calculator
        if not self.resume:
            initialize_velocities(
                atoms,
                self.config.velocities,
                self.config.md.temperature_K,
            )
        elif atoms.get_velocities() is None:
            raise RestartError("checkpoint contains no velocities")

        remaining = max(self.config.md.steps - completed_before, 0)
        dynamics = build_dynamics(atoms, self.config.md)
        dynamics.nsteps = completed_before
        output = self.config.output
        recorder = ThermoRecorder(
            atoms,
            dynamics,
            self.thermo_path if output.log_interval > 0 else None,
            append=self.resume or output.append,
            store_in_memory=output.store_in_memory,
        )
        trajectory = None
        if output.trajectory_interval > 0:
            try:
                from ase.io.trajectory import Trajectory
            except ImportError as exc:
                raise MissingBackendError("ASE trajectory support is unavailable") from exc
            mode = "a" if (self.resume or output.append) else "w"
            trajectory = Trajectory(self.trajectory_path, mode, atoms)
            dynamics.attach(trajectory.write, interval=output.trajectory_interval)
        if output.log_interval > 0:
            dynamics.attach(recorder, interval=output.log_interval)
        safety = SafetyMonitor(atoms, dynamics, self.config.safety)
        dynamics.attach(safety, interval=self.config.safety.check_interval)
        checkpoint = CheckpointWriter(
            atoms,
            dynamics,
            self.checkpoint_path,
            self.checkpoint_state_path,
            metadata={"configuration": self.config.to_dict()},
        )
        if output.checkpoint_interval > 0:
            dynamics.attach(checkpoint, interval=output.checkpoint_interval)
        if self.config.md.remove_com_interval > 0:
            try:
                from ase.md.velocitydistribution import Stationary
            except ImportError as exc:
                raise MissingBackendError("ASE Stationary helper is unavailable") from exc

            dynamics.attach(
                lambda: Stationary(atoms, preserve_temperature=True),
                interval=self.config.md.remove_com_interval,
            )

        payload = {
            "status": "running",
            "started_at_unix": time.time(),
            "completed_before_restart": completed_before,
            "configuration": self.config.to_dict(),
            "calculator": type(self.calculator).__qualname__,
        }
        write_metadata(self.metadata_path, payload)
        # ASE calls attached observers at the initial step before integration.
        # Calling the recorder manually as well duplicates step zero.  A manual
        # sample is needed only when no integration steps remain.
        if remaining == 0:
            recorder()
        started = time.perf_counter()
        status = "completed"
        error: str | None = None
        try:
            if remaining:
                dynamics.run(remaining)
            checkpoint()
        except SimulationDivergedError as exc:
            status = "diverged"
            error = str(exc)
            checkpoint()
            raise
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            try:
                checkpoint()
            except Exception:
                pass
            raise
        finally:
            wall = time.perf_counter() - started
            if trajectory is not None:
                trajectory.close()
            recorder.close()
            payload.update(
                {
                    "status": status,
                    "error": error,
                    "completed_steps": int(dynamics.nsteps),
                    "wall_time_s": wall,
                    "finished_at_unix": time.time(),
                }
            )
            write_metadata(self.metadata_path, payload)

        return SimulationResult(
            initial_structure=initial,
            final_structure=to_structure_data(atoms),
            completed_steps=int(dynamics.nsteps),
            requested_steps=self.config.md.steps,
            wall_time_s=wall,
            thermo=recorder.series,
            trajectory_path=self.trajectory_path if output.trajectory_interval > 0 else None,
            thermo_path=self.thermo_path if output.log_interval > 0 else None,
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            status=status,
            metadata=payload,
        )


def run_md(
    structure: Any,
    calculator: Any,
    config: RunConfig | None = None,
    *,
    resume: bool = False,
) -> SimulationResult:
    return DynamicsSession(structure, calculator, config, resume=resume).run()
