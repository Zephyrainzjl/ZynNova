"""Online reduced-order model management and error-triggered enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..studies.surrogate import PODReducer, PolynomialSurrogate


@dataclass(slots=True)
class AdaptiveROM:
    rank: int | None = None
    energy_fraction: float = 0.999
    error_tolerance: float = 0.02
    reducer: PODReducer = field(init=False)
    dynamics: PolynomialSurrogate = field(init=False)
    snapshots: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.error_tolerance <= 0.0:
            raise ValueError("ROM error tolerance must be positive")
        self.reducer = PODReducer(self.rank, energy_fraction=self.energy_fraction)
        self.dynamics = PolynomialSurrogate(degree=2, ridge=1.0e-8)

    def fit(self, states: np.ndarray, controls: np.ndarray) -> "AdaptiveROM":
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        if states.ndim != 2 or controls.ndim != 2 or len(states) != len(controls) + 1:
            raise ValueError("ROM requires state sequence and one control per transition")
        coordinates = self.reducer.fit_transform(states)
        inputs = np.column_stack((coordinates[:-1], controls))
        self.dynamics.fit(inputs, coordinates[1:])
        self.snapshots = [state.copy() for state in states]
        return self

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        coordinate = self.reducer.encode(np.asarray(state, dtype=float).reshape(1, -1))
        model_input = np.column_stack(
            (coordinate, np.asarray(control, dtype=float).reshape(1, -1))
        )
        next_coordinate = self.dynamics.predict(model_input)
        return self.reducer.decode(next_coordinate)[0]

    def validate_or_enrich(
        self,
        state: np.ndarray,
        control: np.ndarray,
        full_order_step: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, bool, float]:
        rom_prediction = self.step(state, control)
        full_prediction = np.asarray(full_order_step(state, control), dtype=float)
        error = float(
            np.linalg.norm(rom_prediction - full_prediction)
            / max(np.linalg.norm(full_prediction), 1.0e-12)
        )
        if error <= self.error_tolerance:
            return rom_prediction, False, error
        self.snapshots.extend([np.asarray(state, dtype=float).copy(), full_prediction.copy()])
        return full_prediction, True, error


__all__ = ["AdaptiveROM"]
