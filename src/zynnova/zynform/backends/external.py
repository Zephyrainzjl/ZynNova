"""Versioned file adapter for additional image-to-object repositories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from ...core import Availability, dump_json, run_process
from ..schema import ObjectConfig, ObjectRequest
from ..types import ObjectBackendOutput
from .base import ObjectBackend


class ExternalObjectBackend(ObjectBackend):
    def __init__(
        self,
        *,
        name: str = "external-object",
        command: Sequence[str] = (),
        output_mesh: str = "",
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.name = str(name)
        self.command = tuple(str(item) for item in command)
        self.output_mesh = str(output_mesh)
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if not self.command:
            return Availability(False, "command is required for the external object contract")
        if not self.output_mesh:
            return Availability(False, "output_mesh is required for the external object contract")
        if not Path(self.command[0]).is_file() and shutil.which(self.command[0]) is None:
            return Availability(False, f"executable not found: {self.command[0]}")
        return Availability(True)

    def run(
        self,
        request: ObjectRequest,
        config: ObjectConfig,
        work_directory: Path,
    ) -> ObjectBackendOutput:
        output = work_directory / "external_output"
        output.mkdir(parents=True, exist_ok=True)
        payload = dump_json(
            work_directory / "object_request.json",
            {"schema": "zynnova.object-request.v1", "request": request, "config": config},
        )
        values = {
            "request": str(payload),
            "image": str(request.image.resolve()),
            "output": str(output),
            "work": str(work_directory),
        }
        result = run_process(
            [token.format_map(values) for token in self.command],
            cwd=self.cwd,
            timeout_s=self.timeout_s,
        )
        mesh = output / self.output_mesh
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        return ObjectBackendOutput(
            backend=self.name,
            native_mesh=mesh,
            metadata={"elapsed_s": result.elapsed_s, "stdout_tail": result.stdout.splitlines()[-20:]},
        )


__all__ = ["ExternalObjectBackend"]
