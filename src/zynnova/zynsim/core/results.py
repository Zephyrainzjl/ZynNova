"""Numerical result containers with explicit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class SolverDiagnostics:
    converged: bool
    iterations: int
    residual_norm: float
    message: str = ""
    history: tuple[float, ...] = ()


@dataclass(slots=True)
class FieldResult:
    values: np.ndarray
    components: tuple[str, ...]
    unit: str
    location: str = "node"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.location not in {"node", "cell", "face", "particle"}:
            raise ValueError("field location is invalid")
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class TimeSeriesResult:
    times_s: np.ndarray
    fields: Mapping[str, np.ndarray]
    diagnostics: tuple[SolverDiagnostics, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        if self.times_s.ndim != 1 or np.any(np.diff(self.times_s) < 0.0):
            raise ValueError("times_s must be a monotonically increasing vector")
        normalized: dict[str, np.ndarray] = {}
        for name, values in self.fields.items():
            array = np.asarray(values)
            if array.shape[0] != len(self.times_s):
                raise ValueError(f"field {name!r} does not align with times_s")
            normalized[str(name)] = array
        self.fields = normalized
        self.metadata = dict(self.metadata)


__all__ = ["FieldResult", "SolverDiagnostics", "TimeSeriesResult"]
