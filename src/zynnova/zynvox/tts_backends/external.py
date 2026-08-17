"""Versioned file/argv contract for external speech-synthesis engines."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from ...core import Availability, dump_json, run_process
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput
from .base import TTSBackend


class ExternalTTSBackend(TTSBackend):
    name = "external-tts-contract"

    def __init__(
        self,
        *,
        argv: Sequence[str] = (),
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.argv = tuple(str(item) for item in argv)
        self.cwd = None if cwd is None else Path(cwd).expanduser().resolve()
        self.env = None if env is None else {str(key): str(value) for key, value in env.items()}
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if not self.argv:
            return Availability(False, "external TTS argv is empty")
        executable = self.argv[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            return Availability(False, f"external executable not found: {executable}")
        if self.cwd is not None and not self.cwd.is_dir():
            return Availability(False, f"external working directory not found: {self.cwd}")
        return Availability(True, details={"argv0": executable})

    def run(
        self,
        request: TTSRequest,
        config: TTSConfig,
        work_directory: Path,
    ) -> TTSBackendOutput:
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "external_tts.wav"
        payload = dump_json(
            work_directory / "tts_request.json",
            {
                "schema": "zynnova.tts-request/1.0",
                "text": request.text,
                "target_reference": request.target_reference,
                "reference_transcript": request.reference_transcript,
                "language": request.language,
                "emotion_reference": request.emotion_reference,
                "emotion_text": request.emotion_text,
                "emotion_vector": request.emotion_vector,
                "emotion_alpha": request.emotion_alpha,
                "duration_factor": request.duration_factor,
                "style_instruction": request.style_instruction,
                "streaming": request.streaming,
            },
        )
        values = {
            "request": str(payload.resolve()),
            "reference": str(request.target_reference.resolve()),
            "output": str(output.resolve()),
            "workdir": str(work_directory.resolve()),
        }
        argv = [item.format_map(values) for item in self.argv]
        result = run_process(
            argv,
            cwd=self.cwd,
            env=self.env,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"external TTS backend did not create {output}")
        return TTSBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            metadata={
                "argv": argv,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


__all__ = ["ExternalTTSBackend"]
