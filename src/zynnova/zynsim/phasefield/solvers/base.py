"""Common solver utilities and callback protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Protocol

import numpy as np

from ..config import SolverConfig
from ..fields import PhaseFieldResult, PhaseFieldState
from ..models import PhaseFieldModel


class PhaseFieldCallback(Protocol):
    def __call__(self, state: PhaseFieldState, diagnostics) -> None: ...


class PhaseFieldSolver(ABC):
    """Base interface implemented by all phase-field numerical backends."""

    def __init__(self, model: PhaseFieldModel, config: SolverConfig):
        self.model = model
        self.config = config

    @abstractmethod
    def run(
        self,
        initial_state: PhaseFieldState,
        *,
        callbacks: Iterable[PhaseFieldCallback] = (),
    ) -> PhaseFieldResult:
        raise NotImplementedError

    def _target_time(self, state: PhaseFieldState) -> float:
        if self.config.final_time is not None:
            return float(self.config.final_time)
        assert self.config.steps is not None
        return state.time + self.config.steps * self.config.dt

    def _maximum_steps(self) -> int:
        if self.config.steps is not None:
            return min(self.config.steps, self.config.maximum_steps)
        return self.config.maximum_steps



def normalized_error(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    maximum = 0.0
    for name in coarse:
        lhs = np.asarray(coarse[name])
        rhs = np.asarray(fine[name])
        scale = absolute_tolerance + relative_tolerance * np.maximum(
            np.abs(lhs), np.abs(rhs)
        )
        error = np.sqrt(np.mean(((lhs - rhs) / scale) ** 2))
        maximum = max(maximum, float(error))
    return maximum


__all__ = ["PhaseFieldCallback", "PhaseFieldSolver", "normalized_error"]
