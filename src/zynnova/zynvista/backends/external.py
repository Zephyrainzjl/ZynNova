"""Stable subprocess contract for private or future image-to-scene models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from ...core import Availability, dump_json, run_process
from ..schema import SceneConfig, SceneRequest
from ..types import SceneBackendOutput
from .base import SceneBackend


class ExternalSceneBackend(SceneBackend):
    """Invoke an argv template and collect declared outputs without a shell.

    Supported token placeholders are ``{request}``, ``{output}``, and ``{work}``.
    The request JSON is a versioned repository-independent interface.
    """

    def __init__(
        self,
        *,
        name: str = "external-scene",
        command: Sequence[str] = (),
        output_files: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.name = str(name)
        self.command = tuple(str(item) for item in command)
        self.output_files = {str(k): str(v) for k, v in (output_files or {}).items()}
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if not self.command:
            return Availability(False, "external command is empty")
        executable = self.command[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            return Availability(False, f"executable not found: {executable}")
        if self.cwd is not None and not self.cwd.is_dir():
            return Availability(False, f"working directory not found: {self.cwd}")
        return Availability(True)

    def run(
        self,
        request: SceneRequest,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        output = work_directory / "external_output"
        output.mkdir(parents=True, exist_ok=True)
        request_path = dump_json(
            work_directory / "scene_request.json",
            {"schema": "zynnova.scene-request.v1", "request": request, "config": config},
        )
        values = {
            "request": str(request_path),
            "output": str(output),
            "work": str(work_directory),
        }
        result = run_process(
            [token.format_map(values) for token in self.command],
            cwd=self.cwd,
            timeout_s=self.timeout_s,
        )
        assets: dict[str, Path] = {}
        for role, relative in self.output_files.items():
            path = output / relative
            if not path.is_file():
                raise FileNotFoundError(f"external backend did not create {role}: {path}")
            assets[role] = path
        return SceneBackendOutput(
            backend=self.name,
            native_assets=assets,
            metadata={
                "elapsed_s": result.elapsed_s,
                "stdout_tail": result.stdout.splitlines()[-20:],
            },
        )


__all__ = ["ExternalSceneBackend"]
