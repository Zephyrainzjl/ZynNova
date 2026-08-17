"""External style adapter for official 3D scene editing repositories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from ...core import Availability, dump_json, run_process
from ..schema import SceneConfig
from ..types import SceneBackendOutput
from .base import SceneStyleBackend


class ExternalSceneStyle(SceneStyleBackend):
    """Run a style repository through a versioned file contract and argv template."""

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        output_files: Mapping[str, str],
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.name = str(name)
        self.command = tuple(str(item) for item in command)
        self.output_files = {str(k): str(v) for k, v in output_files.items()}
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if not self.command:
            return Availability(False, "style command is empty")
        executable = self.command[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            return Availability(False, f"executable not found: {executable}")
        if self.cwd is not None and not self.cwd.is_dir():
            return Availability(False, f"working directory not found: {self.cwd}")
        return Availability(True)

    def apply(
        self,
        output: SceneBackendOutput,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        directory = work_directory / "styled"
        directory.mkdir(parents=True, exist_ok=True)
        payload = dump_json(
            work_directory / "style_request.json",
            {
                "schema": "zynnova.scene-style.v1",
                "native_assets": output.native_assets,
                "style_reference": config.style_reference,
                "style_prompt": config.style_prompt,
            },
        )
        values = {
            "request": str(payload),
            "output": str(directory),
            "work": str(work_directory),
        }
        result = run_process(
            [token.format_map(values) for token in self.command],
            cwd=self.cwd,
            timeout_s=self.timeout_s,
        )
        assets = dict(output.native_assets)
        for role, relative in self.output_files.items():
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            assets[role] = path
        return SceneBackendOutput(
            backend=output.backend,
            dense_views=output.dense_views,
            point_cloud=output.point_cloud,
            mesh=output.mesh,
            scene=output.scene,
            native_assets=assets,
            metadata={
                **output.metadata,
                "style_backend": self.name,
                "style_elapsed_s": result.elapsed_s,
            },
        )


__all__ = ["ExternalSceneStyle"]
