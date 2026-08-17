"""Backend protocol for file and low-latency voice conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...core import Availability
from ..schema import VoiceConfig, VoiceRequest
from ..types import VoiceBackendOutput


class VoiceBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> Availability:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        request: VoiceRequest,
        config: VoiceConfig,
        work_directory: Path,
    ) -> VoiceBackendOutput:
        raise NotImplementedError


__all__ = ["VoiceBackend"]
