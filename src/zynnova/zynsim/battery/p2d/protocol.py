"""Current/voltage protocols and compact P2D trajectory storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .state import P2DState


CurrentFunction = Callable[[float, P2DState], float]


@dataclass(frozen=True, slots=True)
class CurrentSegment:
    duration_s: float
    current_A: float | CurrentFunction
    time_step_s: float
    minimum_voltage_V: float | None = None
    maximum_voltage_V: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if (
            not np.isfinite((self.duration_s, self.time_step_s)).all()
            or self.duration_s <= 0.0
            or self.time_step_s <= 0.0
        ):
            raise ValueError("protocol duration and time step must be positive")
        if not callable(self.current_A) and not np.isfinite(self.current_A):
            raise ValueError("constant protocol current must be finite")
        limits = (
            value
            for value in (self.minimum_voltage_V, self.maximum_voltage_V)
            if value is not None
        )
        if any(not np.isfinite(value) for value in limits):
            raise ValueError("protocol voltage limits must be finite")
        if (
            self.minimum_voltage_V is not None
            and self.maximum_voltage_V is not None
            and self.minimum_voltage_V >= self.maximum_voltage_V
        ):
            raise ValueError("minimum voltage must be below maximum voltage")

    def current(self, local_time_s: float, state: P2DState) -> float:
        value = float(
            self.current_A(local_time_s, state)
            if callable(self.current_A)
            else self.current_A
        )
        if not np.isfinite(value):
            raise ValueError("protocol current function returned a non-finite value")
        return value


@dataclass(slots=True)
class P2DTrajectory:
    states: list[P2DState] = field(default_factory=list)
    segment_labels: list[str] = field(default_factory=list)
    termination_reason: str = "completed"

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray([state.time_s for state in self.states])

    @property
    def voltage_V(self) -> np.ndarray:
        return np.asarray([state.terminal_voltage_V for state in self.states])

    @property
    def current_A(self) -> np.ndarray:
        return np.asarray([state.current_A for state in self.states])

    @property
    def temperature_K(self) -> np.ndarray:
        return np.asarray([state.temperature_K for state in self.states])


__all__ = ["CurrentFunction", "CurrentSegment", "P2DTrajectory"]
