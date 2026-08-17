"""Backend contracts for scene reconstruction and world generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...core import Availability
from ..schema import SceneConfig, SceneRequest
from ..types import SceneBackendOutput


class SceneBackend(ABC):
    """Normalized contract implemented by every ZynVista scene backend."""

    name: str

    @abstractmethod
    def availability(self) -> Availability:
        """Return a side-effect-free dependency and configuration report."""

    @abstractmethod
    def run(
        self,
        request: SceneRequest,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        """Materialize one reconstruction or world-generation result."""


__all__ = ["SceneBackend"]
