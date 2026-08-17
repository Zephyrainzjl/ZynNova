from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ConformalCalibrator:
    """Per-property split-conformal residual calibration in physical units."""

    quantiles: dict[str, float]
    coverage: float = 0.90

    @classmethod
    def fit(
        cls,
        predictions: Sequence[Mapping[str, float]],
        observations: Sequence[Mapping[str, float]],
        *,
        coverage: float = 0.90,
    ) -> ConformalCalibrator:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie between zero and one")
        residuals: dict[str, list[float]] = {}
        for predicted, observed in zip(predictions, observations, strict=True):
            for name, truth in observed.items():
                if name in predicted and truth is not None:
                    residuals.setdefault(name, []).append(
                        abs(float(predicted[name]) - float(truth))
                    )
        quantiles = {}
        for name, values in residuals.items():
            array = np.sort(np.asarray(values, dtype=float))
            rank = min(
                int(np.ceil((len(array) + 1) * coverage)) - 1,
                len(array) - 1,
            )
            quantiles[name] = float(array[max(rank, 0)])
        return cls(quantiles=quantiles, coverage=coverage)

    def interval(self, name: str, center: float) -> tuple[float, float] | None:
        if name not in self.quantiles:
            return None
        radius = self.quantiles[name]
        return float(center - radius), float(center + radius)

    def state_dict(self) -> dict[str, Any]:
        return {"quantiles": dict(self.quantiles), "coverage": self.coverage}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ConformalCalibrator:
        return cls(
            quantiles={name: float(value) for name, value in state["quantiles"].items()},
            coverage=float(state["coverage"]),
        )


__all__ = ["ConformalCalibrator"]
