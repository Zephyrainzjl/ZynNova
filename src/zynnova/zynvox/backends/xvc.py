"""Official X-VC ``bins.infer_single`` adapter without shell-script coupling."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ...core import Availability, run_process
from ...core.backend import executable_availability
from ..schema import VoiceConfig, VoiceMode, VoiceRequest
from ..types import VoiceBackendOutput
from .base import VoiceBackend


class XVCBackend(VoiceBackend):
    name = "xvc"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        config_path: str | Path = "configs/xvc.yaml",
        checkpoint: str | Path = "ckpts/xvc.pt",
        python_executable: str = sys.executable,
        device: str = "cuda:0",
        current: int = 0,
        chunk: int = 4,
        future: int = 1,
        smooth: int = 1,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.config_path = self._resolve(config_path)
        self.checkpoint = self._resolve(checkpoint)
        self.python_executable = str(python_executable)
        self.device = str(device)
        self.current = int(current)
        self.chunk = int(chunk)
        self.future = int(future)
        self.smooth = int(smooth)
        self.timeout_s = timeout_s
        if min(self.current, self.chunk, self.future, self.smooth) < 0:
            raise ValueError("X-VC streaming parameters cannot be negative")
        if self.chunk < 1:
            raise ValueError("X-VC chunk must be positive")

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        if value.is_absolute():
            return value.resolve()
        if self.repository is None:
            return value
        return (self.repository / value).resolve()

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(
                False,
                "repository is required; pass backend_options={'repository': ...}",
            )
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        if not (self.repository / "bins" / "infer_single.py").is_file():
            return Availability(False, f"X-VC infer_single.py not found under {self.repository}")
        if not self.config_path.is_file():
            return Availability(False, f"X-VC config not found: {self.config_path}")
        if not self.checkpoint.is_file():
            return Availability(False, f"X-VC checkpoint not found: {self.checkpoint}")
        return Availability(
            True,
            details={"repository": str(self.repository), "checkpoint": str(self.checkpoint)},
        )

    def run(
        self,
        request: VoiceRequest,
        config: VoiceConfig,
        work_directory: Path,
    ) -> VoiceBackendOutput:
        if request.mode is VoiceMode.REALTIME:
            raise ValueError("X-VC microphone streaming requires its official streaming service")
        if self.repository is None:  # guarded by registry, retained for direct construction
            raise ValueError("repository is required for X-VC")
        save_directory = work_directory / "xvc_outputs"
        save_directory.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in save_directory.rglob("*.wav")}
        current = self.current
        if request.mode is VoiceMode.STREAMING_FILE and current <= 0:
            current = self.chunk
        result = run_process(
            [
                self.python_executable,
                "-m",
                "bins.infer_single",
                "--config",
                str(self.config_path),
                "--ckpt",
                str(self.checkpoint),
                "--device",
                self.device,
                "--source_wav_path",
                str(request.source_audio.resolve()),
                "--target_wav_path",
                str(request.target_reference.resolve()),
                "--save_dir",
                str(save_directory.resolve()),
                "--current",
                str(current),
                "--chunk",
                str(self.chunk),
                "--future",
                str(self.future),
                "--smooth",
                str(self.smooth),
            ],
            cwd=self.repository,
            timeout_s=self.timeout_s,
        )
        created = [
            path
            for path in save_directory.rglob("*.wav")
            if path.resolve() not in before
        ]
        if not created:
            created = list(save_directory.rglob("*.wav"))
        if not created:
            raise RuntimeError(f"X-VC did not create a WAV file under {save_directory}")
        source = max(created, key=lambda item: item.stat().st_mtime_ns)
        output = work_directory / "xvc.wav"
        shutil.copy2(source, output)
        return VoiceBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            metadata={
                "current": current,
                "chunk": self.chunk,
                "future": self.future,
                "smooth": self.smooth,
                "source_output": str(source),
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


__all__ = ["XVCBackend"]
