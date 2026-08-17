"""Backend protocol for conditional battery microstructure generation."""

from __future__ import annotations

from typing import Protocol

from ...core.backend import Availability
from ..generation import GenerationResult
from ..schema import MicrostructureCondition


class MicrostructureBackend(Protocol):
    name: str

    def availability(self) -> Availability: ...

    def generate(
        self,
        condition: MicrostructureCondition,
        *,
        refinement_steps: int = 0,
        temperature: float = 0.15,
    ) -> GenerationResult: ...


__all__ = ["MicrostructureBackend"]
