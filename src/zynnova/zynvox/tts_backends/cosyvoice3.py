"""Isolated adapter for the official Fun-CosyVoice 3.0 API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...core import Availability, dump_json, run_process
from ...core.backend import executable_availability
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput
from .base import TTSBackend


class CosyVoice3Backend(TTSBackend):
    name = "cosyvoice-3"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        model_directory: str | Path | None = None,
        python_executable: str = sys.executable,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        if model_directory is None and self.repository is not None:
            model_directory = self.repository / "pretrained_models" / "Fun-CosyVoice3-0.5B"
        self.model_directory = (
            None if model_directory is None else Path(model_directory).expanduser().resolve()
        )
        self.python_executable = str(python_executable)
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(False, "repository is required for CosyVoice 3")
        if not (self.repository / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
            return Availability(False, f"CosyVoice source not found under {self.repository}")
        if self.model_directory is None or not self.model_directory.is_dir():
            return Availability(False, f"CosyVoice 3 model directory not found: {self.model_directory}")
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
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
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "cosyvoice3.wav"
        metadata_path = work_directory / "cosyvoice3_metadata.json"
        payload = dump_json(
            work_directory / "request.json",
            {
                "text": request.text,
                "target_reference": str(request.target_reference.resolve()),
                "reference_transcript": request.reference_transcript,
                "style_instruction": request.style_instruction,
                "streaming": request.streaming,
            },
        )
        runner = Path(__file__).resolve().parents[1] / "adapters" / "cosyvoice3_runner.py"
        result = run_process(
            [
                self.python_executable,
                str(runner),
                "--request",
                str(payload),
                "--repository",
                str(self.repository),
                "--model-dir",
                str(self.model_directory),
                "--output",
                str(output),
                "--metadata",
                str(metadata_path),
            ],
            cwd=self.repository,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"CosyVoice 3 did not create {output}")
        internal = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        return TTSBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            first_packet_latency_ms=internal.get("first_packet_latency_ms"),
            metadata={
                **internal,
                "model_directory": str(self.model_directory),
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


__all__ = ["CosyVoice3Backend"]
