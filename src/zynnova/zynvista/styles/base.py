"""Style-transfer contracts isolated from geometry reconstruction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...core import Availability
from ..schema import SceneConfig
from ..types import SceneBackendOutput


class SceneStyleBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> Availability:
        """Report availability without loading weights."""

    @abstractmethod
    def apply(
        self,
        output: SceneBackendOutput,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        """Return a style-modified scene result."""


__all__ = ["SceneStyleBackend"]
