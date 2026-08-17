from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .predictor import PolymerPrediction


@dataclass(frozen=True, slots=True)
class PropertyConstraint:
    name: str
    lower: float | None = None
    upper: float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("constraint weight must be positive")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("constraint lower bound cannot exceed upper bound")


@dataclass(slots=True)
class ScreenedPolymer:
    prediction: PolymerPrediction
    feasibility_probability: float
    score: float
    constraint_probabilities: dict[str, float]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _constraint_probability(
    mean: float,
    std: float,
    constraint: PropertyConstraint,
) -> float:
    if std <= 1.0e-12:
        feasible = (constraint.lower is None or mean >= constraint.lower) and (
            constraint.upper is None or mean <= constraint.upper
        )
        return float(feasible)
    lower_cdf = 0.0 if constraint.lower is None else _normal_cdf((constraint.lower - mean) / std)
    upper_cdf = 1.0 if constraint.upper is None else _normal_cdf((constraint.upper - mean) / std)
    return max(min(upper_cdf - lower_cdf, 1.0), 0.0)


def screen_predictions(
    predictions: Sequence[PolymerPrediction],
    constraints: Sequence[PropertyConstraint],
    *,
    uncertainty_penalty: float = 0.02,
) -> list[ScreenedPolymer]:
    ranked = []
    for prediction in predictions:
        probabilities = {}
        log_probability = 0.0
        normalized_uncertainty = 0.0
        for constraint in constraints:
            if constraint.name not in prediction.mean:
                probability = 0.0
            else:
                mean = prediction.mean[constraint.name]
                std = prediction.standard_deviation[constraint.name]
                probability = _constraint_probability(mean, std, constraint)
                normalized_uncertainty += std / max(abs(mean), 1.0e-8)
            probabilities[constraint.name] = probability
            log_probability += constraint.weight * math.log(max(probability, 1.0e-12))
        feasibility = math.exp(
            log_probability / max(sum(item.weight for item in constraints), 1.0e-12)
        )
        score = log_probability - uncertainty_penalty * normalized_uncertainty
        ranked.append(
            ScreenedPolymer(
                prediction=prediction,
                feasibility_probability=feasibility,
                score=score,
                constraint_probabilities=probabilities,
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


__all__ = [
    "PropertyConstraint",
    "ScreenedPolymer",
    "screen_predictions",
]
