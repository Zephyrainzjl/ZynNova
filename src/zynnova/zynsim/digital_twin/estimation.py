"""Online SOC, SOH, and remaining-useful-life estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class HealthEstimate:
    time_s: float
    soc_mean: float
    soc_standard_deviation: float
    soh_mean: float
    soh_standard_deviation: float
    rul_s: float
    rul_standard_deviation_s: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class OnlineHealthEstimator:
    soc_index: int
    soh_index: int
    end_of_life_soh: float = 0.8
    minimum_fade_rate_s_inv: float = 1.0e-12
    history: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.end_of_life_soh < 1.0:
            raise ValueError("end_of_life_soh must lie in (0,1)")

    def estimate(self, time_s: float, ensemble: np.ndarray) -> HealthEstimate:
        values = np.asarray(ensemble, dtype=float)
        if values.ndim != 2:
            raise ValueError("health estimator requires a two-dimensional ensemble")
        soc = np.clip(values[:, self.soc_index], 0.0, 1.0)
        soh = np.clip(values[:, self.soh_index], 0.0, 1.0)
        mean_soh = float(np.mean(soh))
        self.history.append((float(time_s), mean_soh))
        fade_rate = self._fade_rate()
        sample_rates = np.maximum(
            fade_rate * (1.0 + (soh - mean_soh)), self.minimum_fade_rate_s_inv
        )
        rul_samples = np.maximum(
            (soh - self.end_of_life_soh) / sample_rates,
            0.0,
        )
        return HealthEstimate(
            time_s=float(time_s),
            soc_mean=float(np.mean(soc)),
            soc_standard_deviation=float(np.std(soc, ddof=1)),
            soh_mean=mean_soh,
            soh_standard_deviation=float(np.std(soh, ddof=1)),
            rul_s=float(np.mean(rul_samples)),
            rul_standard_deviation_s=float(np.std(rul_samples, ddof=1)),
            metadata={"estimated_fade_rate_s_inv": fade_rate},
        )

    def _fade_rate(self) -> float:
        if len(self.history) < 3:
            return self.minimum_fade_rate_s_inv
        recent = self.history[-20:]
        time = np.asarray([item[0] for item in recent])
        soh = np.asarray([item[1] for item in recent])
        if np.ptp(time) <= 0.0:
            return self.minimum_fade_rate_s_inv
        slope = float(np.polyfit(time, soh, 1)[0])
        return max(-slope, self.minimum_fade_rate_s_inv)


__all__ = ["HealthEstimate", "OnlineHealthEstimator"]
