"""Versioned file-contract adapter for independently maintained VC systems."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from ...core import Availability, run_process
from ..schema import VoiceConfig, VoiceRequest
from ..types import VoiceBackendOutput
from .base import VoiceBackend


class ExternalVoiceBackend(VoiceBackend):
    """Execute a shell-free argv template.

    Supported placeholders are ``{source}``, ``{target}``, ``{output}``,
    ``{workdir}``, and ``{mode}``. The external process must create ``{output}``.
    """

    name = "external-voice-contract"

    def __init__(
        self,
        *,
        argv: Sequence[str] = (),
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        backend_name: str | None = None,
        **_: object,
    ) -> None:
        self.argv = tuple(str(item) for item in argv)
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        self.env = dict(env or {})
        self.timeout_s = timeout_s
        if backend_name:
            self.name = str(backend_name)

    def availability(self) -> Availability:
        if not self.argv:
            return Availability(False, "argv is required for the external voice contract")
        executable = self.argv[0]
        if os.path.sep in executable:
            available = Path(executable).expanduser().is_file()
        else:
            import shutil

            available = shutil.which(executable) is not None
        if not available:
            return Availability(False, f"external executable not found: {executable}")
        if self.cwd is not None and not self.cwd.is_dir():
            return Availability(False, f"external working directory not found: {self.cwd}")
        return Availability(True, details={"argv0": executable})

    def run(
        self,
        request: VoiceRequest,
        config: VoiceConfig,
        work_directory: Path,
    ) -> VoiceBackendOutput:
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "external.wav"
        values = {
            "source": str(request.source_audio.resolve()),
            "target": str(request.target_reference.resolve()),
            "output": str(output.resolve()),
            "workdir": str(work_directory.resolve()),
            "mode": request.mode.value,
        }
        argv = [item.format_map(values) for item in self.argv]
        result = run_process(
            argv,
            cwd=self.cwd,
            env=self.env,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"external voice backend did not create {output}")
        return VoiceBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            metadata={
                "argv": argv,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


__all__ = ["ExternalVoiceBackend"]
