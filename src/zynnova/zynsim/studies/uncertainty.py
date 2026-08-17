"""Uncertainty quantification and local sensitivity analysis.

The routines in this module deliberately operate on arbitrary nested,
dataclass-based parameter objects.  They therefore work with the battery,
thermal, mechanics, and electrochemistry parameter trees without requiring a
second parameter representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .parameters import get_parameter, replace_parameter, replace_parameters

Distribution = Literal["normal", "uniform", "lognormal"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UncertainParameter:
    """Probability distribution attached to a nested parameter path.

    ``normal`` uses ``location`` as the mean and ``scale`` as the standard
    deviation.  ``lognormal`` uses them as the mean and standard deviation in
    log space.  ``uniform`` requires ``lower`` and ``upper``.
    """

    path: str
    distribution: Distribution
    location: float = 0.0
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path cannot be empty")
        if self.distribution not in {"normal", "uniform", "lognormal"}:
            raise ValueError(f"unsupported distribution: {self.distribution}")
        if self.distribution == "uniform":
            if (
                self.lower is None
                or self.upper is None
                or not np.isfinite((self.lower, self.upper)).all()
                or self.upper <= self.lower
            ):
                raise ValueError("uniform distribution requires lower < upper")
        elif (
            not np.isfinite((self.location, self.scale)).all()
            or self.scale < 0.0
        ):
            raise ValueError("scale must be non-negative")

    def sample(self, generator: np.random.Generator, count: int) -> FloatArray:
        if count <= 0:
            raise ValueError("count must be positive")
        if self.distribution == "normal":
            values = generator.normal(self.location, self.scale, count)
        elif self.distribution == "lognormal":
            values = generator.lognormal(self.location, self.scale, count)
        else:
            assert self.lower is not None and self.upper is not None
            values = generator.uniform(self.lower, self.upper, count)
        return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class MonteCarloResult:
    """Samples and output statistics from a Monte Carlo study."""

    parameter_samples: Mapping[str, FloatArray]
    metric_samples: Mapping[str, FloatArray]
    metric_mean: Mapping[str, float]
    metric_standard_deviation: Mapping[str, float]
    metric_quantiles: Mapping[str, Mapping[float, float]]
    parameter_metric_correlation: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class SensitivityResult:
    """Central finite-difference sensitivities for scalar output metrics."""

    baseline: Mapping[str, float]
    derivatives: Mapping[str, Mapping[str, float]]
    normalized_sensitivities: Mapping[str, Mapping[str, float]]
    perturbations: Mapping[str, float]


def monte_carlo(
    base_parameters: object,
    uncertain_parameters: Sequence[UncertainParameter],
    evaluator: Callable[[object], Mapping[str, float]],
    *,
    samples: int = 100,
    seed: int | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> MonteCarloResult:
    """Propagate independent input distributions through ``evaluator``.

    The evaluator must return the same finite scalar metrics for every sample.
    Independence is an explicit assumption; correlated sampling can be added by
    constructing a custom evaluator or by pre-transforming independent draws.
    """

    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be an integer")
    if samples <= 0:
        raise ValueError("samples must be positive")
    quantile_values = tuple(float(value) for value in quantiles)
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in quantile_values
    ):
        raise ValueError("quantiles must lie in [0, 1]")
    paths = [parameter.path for parameter in uncertain_parameters]
    if len(paths) != len(set(paths)):
        raise ValueError("uncertain parameter paths must be unique")

    generator = np.random.default_rng(seed)
    parameter_samples = {
        parameter.path: parameter.sample(generator, samples)
        for parameter in uncertain_parameters
    }

    collected: dict[str, list[float]] = {}
    expected_metrics: tuple[str, ...] | None = None
    for index in range(samples):
        updates = {
            path: float(values[index]) for path, values in parameter_samples.items()
        }
        candidate = replace_parameters(base_parameters, updates)
        metrics = _finite_metrics(evaluator(candidate))
        names = tuple(sorted(metrics))
        if expected_metrics is None:
            expected_metrics = names
            collected = {name: [] for name in names}
        elif names != expected_metrics:
            raise ValueError(
                "evaluator returned inconsistent metric names: "
                f"expected {expected_metrics}, received {names}"
            )
        for name in names:
            collected[name].append(metrics[name])

    metric_samples = {
        name: np.asarray(values, dtype=float) for name, values in collected.items()
    }
    means = {name: float(np.mean(values)) for name, values in metric_samples.items()}
    standard_deviations = {
        name: float(np.std(values, ddof=1 if samples > 1 else 0))
        for name, values in metric_samples.items()
    }
    metric_quantiles = {
        name: {
            quantile: float(np.quantile(values, quantile))
            for quantile in quantile_values
        }
        for name, values in metric_samples.items()
    }
    correlations = {
        parameter_name: {
            metric_name: _safe_correlation(parameter_values, metric_values)
            for metric_name, metric_values in metric_samples.items()
        }
        for parameter_name, parameter_values in parameter_samples.items()
    }

    return MonteCarloResult(
        parameter_samples=parameter_samples,
        metric_samples=metric_samples,
        metric_mean=means,
        metric_standard_deviation=standard_deviations,
        metric_quantiles=metric_quantiles,
        parameter_metric_correlation=correlations,
    )


def local_sensitivity(
    base_parameters: object,
    parameter_paths: Sequence[str],
    evaluator: Callable[[object], Mapping[str, float]],
    *,
    relative_step: float = 1.0e-4,
    absolute_step: float = 1.0e-8,
) -> SensitivityResult:
    """Compute central-difference and dimensionless local sensitivities."""

    if relative_step <= 0.0 or absolute_step <= 0.0:
        raise ValueError("relative_step and absolute_step must be positive")
    baseline = _finite_metrics(evaluator(base_parameters))
    derivatives: dict[str, dict[str, float]] = {}
    normalized: dict[str, dict[str, float]] = {}
    perturbations: dict[str, float] = {}

    for path in parameter_paths:
        original = float(get_parameter(base_parameters, path))
        step = max(abs(original) * relative_step, absolute_step)
        plus = _finite_metrics(
            evaluator(replace_parameter(base_parameters, path, original + step))
        )
        minus = _finite_metrics(
            evaluator(replace_parameter(base_parameters, path, original - step))
        )
        if set(plus) != set(baseline) or set(minus) != set(baseline):
            raise ValueError("evaluator returned inconsistent metric names")
        path_derivatives: dict[str, float] = {}
        path_normalized: dict[str, float] = {}
        for metric_name, baseline_value in baseline.items():
            derivative = (plus[metric_name] - minus[metric_name]) / (2.0 * step)
            path_derivatives[metric_name] = float(derivative)
            denominator = max(abs(baseline_value), np.finfo(float).eps)
            path_normalized[metric_name] = float(derivative * original / denominator)
        derivatives[path] = path_derivatives
        normalized[path] = path_normalized
        perturbations[path] = step

    return SensitivityResult(
        baseline=baseline,
        derivatives=derivatives,
        normalized_sensitivities=normalized,
        perturbations=perturbations,
    )


def _finite_metrics(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("evaluator must return at least one metric")
    result = {str(name): float(value) for name, value in values.items()}
    invalid = [name for name, value in result.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"evaluator returned non-finite metrics: {invalid}")
    return result


def _safe_correlation(left: FloatArray, right: FloatArray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
