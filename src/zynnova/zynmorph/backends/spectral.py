"""Always-available exact-composition spectral backend."""

from __future__ import annotations

from ...core.backend import Availability
from ..generation import GenerationResult, SpectralConditionalGenerator
from ..schema import MicrostructureCondition


class SpectralBackend:
    name = "spectral-exact"

    def availability(self) -> Availability:
        return Availability(True, details={"dependencies": ["numpy"]})

    def generate(
        self,
        condition: MicrostructureCondition,
        *,
        refinement_steps: int = 0,
        temperature: float = 0.15,
    ) -> GenerationResult:
        return SpectralConditionalGenerator().generate(
            condition,
            refinement_steps=refinement_steps,
            temperature=temperature,
        )


__all__ = ["SpectralBackend"]
