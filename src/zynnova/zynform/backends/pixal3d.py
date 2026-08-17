"""Official Pixal3D SIGGRAPH 2026 command adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from ...core import Availability, run_process
from ...core.backend import executable_availability
from ..schema import ObjectConfig, ObjectRequest
from ..types import ObjectBackendOutput
from .base import ObjectBackend


class Pixal3DBackend(ObjectBackend):
    name = "pixal3d"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        python_executable: str = sys.executable,
        low_vram: bool = False,
        resolution: int | None = None,
        attention_backend: str | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.python_executable = str(python_executable)
        self.low_vram = bool(low_vram)
        self.resolution = resolution
        self.attention_backend = attention_backend
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(
                False,
                "repository is required; pass backend_options={'repository': ...}",
            )
        if not (self.repository / "inference.py").is_file():
            return Availability(False, f"Pixal3D inference.py not found in {self.repository}")
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        return Availability(
            True,
            details={"low_vram": self.low_vram, "resolution": self.resolution},
        )

    def run(
        self,
        request: ObjectRequest,
        config: ObjectConfig,
        work_directory: Path,
    ) -> ObjectBackendOutput:
        if self.repository is None:  # guarded by registry, retained for direct construction
            raise ValueError("repository is required for Pixal3D")
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "pixal3d.glb"
        argv = [
            self.python_executable,
            "inference.py",
            "--image",
            str(request.image.resolve()),
            "--output",
            str(output),
        ]
        if self.low_vram:
            argv.append("--low_vram")
        if self.resolution is not None:
            argv += ["--resolution", str(int(self.resolution))]
        env = None
        if self.attention_backend:
            env = {"ATTN_BACKEND": self.attention_backend}
        result = run_process(
            argv,
            cwd=self.repository,
            env=env,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"Pixal3D did not create {output}")
        return ObjectBackendOutput(
            backend=self.name,
            native_mesh=output,
            metadata={
                "elapsed_s": result.elapsed_s,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "pbr": True,
            },
        )


__all__ = ["Pixal3DBackend"]
