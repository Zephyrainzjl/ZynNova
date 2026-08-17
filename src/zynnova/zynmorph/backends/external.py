"""JSON/NPZ contract for isolated public-repository microstructure generators."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...core.backend import Availability
from ...core.process import run_process
from ...core.serialization import dump_json, to_jsonable
from ..generation import GenerationResult
from ..schema import DEFAULT_PHASE_NAMES, MicrostructureCondition
from ..volume import MicrostructureVolume


class ExternalMicrostructureBackend:
    """Invoke a repository wrapper without importing its dependency stack.

    The command receives ``--request REQUEST.json --output OUTPUT.npz``. The NPZ
    must contain a three-dimensional integer array named ``labels``.
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.name = str(name)
        self.command = tuple(str(item) for item in command)
        self.cwd = None if cwd is None else Path(cwd)
        self.environment = dict(environment or {})
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if not self.command:
            return Availability(False, "external command was not configured")
        executable = Path(self.command[0])
        if (executable.is_absolute() or executable.parent != Path(".")) and not executable.exists():
            return Availability(False, f"command executable does not exist: {executable}")
        if self.cwd is not None and not self.cwd.is_dir():
            return Availability(False, f"working directory does not exist: {self.cwd}")
        return Availability(True, details={"command": self.command, "cwd": str(self.cwd)})

    def generate(
        self,
        condition: MicrostructureCondition,
        *,
        refinement_steps: int = 0,
        temperature: float = 0.15,
    ) -> GenerationResult:
        self.availability().require(self.name)
        with tempfile.TemporaryDirectory(prefix="zynmorph-") as temporary:
            request = Path(temporary) / "request.json"
            output = Path(temporary) / "output.npz"
            dump_json(
                request,
                {
                    "schema": "zynnova.zynmorph.external.v1",
                    "condition": to_jsonable(condition),
                    "refinement_steps": refinement_steps,
                    "temperature": temperature,
                },
            )
            run_process(
                [*self.command, "--request", str(request), "--output", str(output)],
                cwd=self.cwd,
                env=self.environment,
                timeout_s=self.timeout_s,
            )
            if not output.is_file():
                raise FileNotFoundError(f"external backend did not create {output}")
            with np.load(output, allow_pickle=False) as data:
                labels = data["labels"]
        volume = MicrostructureVolume(
            labels=labels,
            voxel_size_m=condition.voxel_size_m,
            phase_names=DEFAULT_PHASE_NAMES,
            metadata={"generator": self.name, "external_command": self.command},
        )
        achieved = {
            phase: int(np.count_nonzero(volume.labels == phase)) for phase in condition.phases
        }
        return GenerationResult(
            volume=volume,
            backend=self.name,
            exact_counts=condition.exact_phase_counts(),
            achieved_counts=achieved,
            refinement_loss=None,
            metadata={"external": True},
        )


__all__ = ["ExternalMicrostructureBackend"]
