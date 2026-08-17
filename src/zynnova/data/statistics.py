from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .record import MaterialSample
from .transforms import Compose, StandardizeField


@dataclass(frozen=True, slots=True)
class FieldStatistics:
    path: str
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def standardizer(self) -> StandardizeField:
        return StandardizeField(self.path, self.mean, self.std)


def fit_field_statistics(
    samples: Iterable[MaterialSample],
    fields: Sequence[str],
    *,
    ignore_non_finite: bool = True,
) -> dict[str, FieldStatistics]:
    """Fit scalar standardization statistics using a stable online algorithm."""
    accumulators = {
        path: {"count": 0, "mean": 0.0, "m2": 0.0, "min": np.inf, "max": -np.inf}
        for path in fields
    }
    for sample in samples:
        for path, state in accumulators.items():
            value = sample.get(path)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            for scalar in array:
                if ignore_non_finite and not np.isfinite(scalar):
                    continue
                state["count"] += 1
                delta = float(scalar) - state["mean"]
                state["mean"] += delta / state["count"]
                state["m2"] += delta * (float(scalar) - state["mean"])
                state["min"] = min(state["min"], float(scalar))
                state["max"] = max(state["max"], float(scalar))
    result: dict[str, FieldStatistics] = {}
    for path, state in accumulators.items():
        count = int(state["count"])
        if count == 0:
            raise ValueError(f"cannot fit statistics for empty field {path!r}")
        variance = state["m2"] / max(count - 1, 1)
        result[path] = FieldStatistics(
            path=path,
            count=count,
            mean=float(state["mean"]),
            std=float(np.sqrt(max(variance, 0.0))),
            minimum=float(state["min"]),
            maximum=float(state["max"]),
        )
    return result


def standardization_pipeline(
    statistics: dict[str, FieldStatistics],
) -> Compose:
    return Compose([value.standardizer() for value in statistics.values()])
