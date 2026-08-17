"""Explicitly opt-in identity baseline for tests and benchmark sanity checks."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ...core import Availability
from ..schema import VoiceConfig, VoiceMode, VoiceRequest
from ..types import VoiceBackendOutput
from .base import VoiceBackend


class IdentityVoiceBaseline(VoiceBackend):
    name = "identity-baseline"

    def __init__(self, *, allow_baseline: bool = False, **_: object) -> None:
        self.allow_baseline = bool(allow_baseline)

    def availability(self) -> Availability:
        if not self.allow_baseline:
            return Availability(
                False,
                "identity baseline is disabled; pass allow_baseline=True explicitly",
            )
        return Availability(True, details={"warning": "no voice conversion is performed"})

    def run(
        self,
        request: VoiceRequest,
        config: VoiceConfig,
        work_directory: Path,
    ) -> VoiceBackendOutput:
        if request.mode is VoiceMode.REALTIME:
            raise ValueError("identity file baseline does not provide microphone streaming")
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / f"identity{request.source_audio.suffix}"
        started = time.perf_counter()
        shutil.copy2(request.source_audio, output)
        return VoiceBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=time.perf_counter() - started,
            metadata={"warning": "identity baseline: source audio copied unchanged"},
        )


__all__ = ["IdentityVoiceBaseline"]
