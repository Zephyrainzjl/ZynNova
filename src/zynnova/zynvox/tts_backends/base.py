"""Backend protocol for reference-conditioned speech synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...core import Availability
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput


class TTSBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> Availability:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        request: TTSRequest,
        config: TTSConfig,
        work_directory: Path,
    ) -> TTSBackendOutput:
        raise NotImplementedError


__all__ = ["TTSBackend"]
