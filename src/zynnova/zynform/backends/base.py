"""Backend contract for image-to-3D object systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...core import Availability
from ..schema import ObjectConfig, ObjectRequest
from ..types import ObjectBackendOutput


class ObjectBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> Availability:
        """Inspect dependencies and configuration without loading model weights."""

    @abstractmethod
    def run(
        self,
        request: ObjectRequest,
        config: ObjectConfig,
        work_directory: Path,
    ) -> ObjectBackendOutput:
        """Generate one object asset from the requested image."""


__all__ = ["ObjectBackend"]
