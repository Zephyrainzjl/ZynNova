"""Isolated adapter for the official IndexTTS-2.5 Python API."""

from __future__ import annotations

import sys
from pathlib import Path

from ...core import Availability, dump_json, run_process
from ...core.backend import executable_availability
from ...core.licenses import require_known_license
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput
from .base import TTSBackend


class IndexTTS25Backend(TTSBackend):
    name = "indextts-2.5"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        model_directory: str | Path | None = None,
        config_path: str | Path | None = None,
        python_executable: str = sys.executable,
        use_bf16: bool = True,
        accept_license: bool = False,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        if model_directory is None and self.repository is not None:
            model_directory = self.repository / "checkpoints"
        self.model_directory = (
            None if model_directory is None else Path(model_directory).expanduser().resolve()
        )
        if config_path is None and self.model_directory is not None:
            config_path = self.model_directory / "config.yaml"
        self.config_path = None if config_path is None else Path(config_path).expanduser().resolve()
        self.python_executable = str(python_executable)
        self.use_bf16 = bool(use_bf16)
        self.accept_license = bool(accept_license)
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(False, "repository is required for IndexTTS-2.5")
        if self.model_directory is None or not self.model_directory.is_dir():
            return Availability(False, f"IndexTTS-2.5 model directory not found: {self.model_directory}")
        if self.config_path is None or not self.config_path.is_file():
            return Availability(False, f"IndexTTS-2.5 config not found: {self.config_path}")
        if not (self.repository / "indextts" / "infer_v2_5.py").is_file():
            return Availability(False, f"IndexTTS-2.5 source not found under {self.repository}")
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        try:
            require_known_license("indextts-2.5", explicit=self.accept_license)
        except Exception as exc:
            return Availability(False, str(exc))
        return Availability(
            True,
            details={"repository": str(self.repository), "model_directory": str(self.model_directory)},
        )

    def run(
        self,
        request: TTSRequest,
        config: TTSConfig,
        work_directory: Path,
    ) -> TTSBackendOutput:
        self.availability().require(self.name)
        assert self.repository is not None
        assert self.model_directory is not None
        assert self.config_path is not None
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "indextts25.wav"
        payload = dump_json(
            work_directory / "request.json",
            {
                "text": request.text,
                "target_reference": str(request.target_reference.resolve()),
                "language": request.index_language,
                "emotion_reference": None
                if request.emotion_reference is None
                else str(request.emotion_reference.resolve()),
                "emotion_text": request.emotion_text,
                "emotion_vector": request.emotion_vector,
                "emotion_alpha": request.emotion_alpha,
                "duration_factor": request.duration_factor,
            },
        )
        runner = Path(__file__).resolve().parents[1] / "adapters" / "indextts25_runner.py"
        argv = [
            self.python_executable,
            str(runner),
            "--request",
            str(payload),
            "--repository",
            str(self.repository),
            "--model-dir",
            str(self.model_directory),
            "--config",
            str(self.config_path),
            "--output",
            str(output),
        ]
        if self.use_bf16:
            argv.append("--bf16")
        result = run_process(argv, cwd=self.repository, timeout_s=self.timeout_s)
        if not output.is_file():
            raise RuntimeError(f"IndexTTS-2.5 did not create {output}")
        return TTSBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            metadata={
                "model_directory": str(self.model_directory),
                "use_bf16": self.use_bf16,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


__all__ = ["IndexTTS25Backend"]
