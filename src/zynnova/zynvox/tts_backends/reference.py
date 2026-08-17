"""Explicit test-only TTS baseline."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ...core import Availability
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput
from .base import TTSBackend


class ReferenceAudioTTSBaseline(TTSBackend):
    """Copy the reference audio for plumbing tests; never presented as synthesis."""

    name = "reference-audio-baseline"

    def __init__(self, *, allow_baseline: bool = False, **_: object) -> None:
        self.allow_baseline = bool(allow_baseline)

    def availability(self) -> Availability:
        if not self.allow_baseline:
            return Availability(False, "test-only baseline requires allow_baseline=True")
        return Availability(True)

    def run(
        self,
        request: TTSRequest,
        config: TTSConfig,
        work_directory: Path,
    ) -> TTSBackendOutput:
        del config
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / f"reference{request.target_reference.suffix or '.wav'}"
        start = time.perf_counter()
        shutil.copy2(request.target_reference, output)
        elapsed = time.perf_counter() - start
        return TTSBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=elapsed,
            metadata={"test_only": True, "synthesis_performed": False},
        )


__all__ = ["ReferenceAudioTTSBaseline"]
