from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def robust_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    scale = (q75 - q25) / 1.3489795003921634
    standard = np.nanstd(values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, standard)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - center) / scale, center, scale


def standard_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    center = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - center) / scale, center, scale


def soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


def elastic_net_coordinate_descent(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    sample_weight: np.ndarray | None = None,
    max_iterations: int = 5000,
    tolerance: float = 1.0e-8,
) -> tuple[float, np.ndarray]:
    """Fit standardized elastic net without a hard dependency on scikit-learn."""

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("matrix and target shapes are incompatible")
    if alpha <= 0 or not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("invalid elastic-net regularization")
    weights = _sample_weights(sample_weight, y.size)
    weight_sum = float(np.sum(weights))
    intercept = float(np.sum(weights * y) / weight_sum)
    centered_y = y - intercept
    coefficients = np.zeros(x.shape[1], dtype=float)
    column_norm = np.sum(weights[:, None] * x * x, axis=0) / weight_sum
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    residual = centered_y.copy()
    for _ in range(max_iterations):
        maximum_change = 0.0
        for column in range(x.shape[1]):
            previous = coefficients[column]
            residual += x[:, column] * previous
            partial = float(
                np.sum(weights * x[:, column] * residual) / weight_sum
            )
            updated = soft_threshold(partial, l1) / max(column_norm[column] + l2, 1.0e-15)
            coefficients[column] = updated
            residual -= x[:, column] * updated
            maximum_change = max(maximum_change, abs(updated - previous))
        if maximum_change < tolerance:
            break
    return intercept, coefficients


def prediction(intercept: float, coefficients: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return float(intercept) + np.asarray(matrix, dtype=float) @ np.asarray(
        coefficients, dtype=float
    )


def mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    weights = _sample_weights(sample_weight, residual.size)
    return float(np.sum(weights * residual * residual) / np.sum(weights))


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=float)
    residual = y - np.asarray(predicted, dtype=float)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(residual @ residual) / max(denominator, 1.0e-15)


def make_folds(
    sample_count: int,
    fold_count: int,
    *,
    environments: Sequence[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    fold_count = min(max(int(fold_count), 2), sample_count)
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(fold_count)]
    if environments is None:
        indices = rng.permutation(sample_count)
        for position, index in enumerate(indices):
            buckets[position % fold_count].append(int(index))
    else:
        values = np.asarray(environments, dtype=object)
        if values.shape != (sample_count,):
            raise ValueError("environment labels have the wrong length")
        for environment in sorted(set(values.tolist())):
            indices = np.flatnonzero(values == environment)
            indices = rng.permutation(indices)
            for position, index in enumerate(indices):
                buckets[position % fold_count].append(int(index))
    return tuple(np.asarray(sorted(bucket), dtype=int) for bucket in buckets if bucket)


def select_elastic_net_alpha(
    matrix: np.ndarray,
    target: np.ndarray,
    alpha_grid: Sequence[float],
    *,
    l1_ratio: float,
    fold_count: int,
    environments: Sequence[str] | None = None,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[float, dict[float, float]]:
    folds = make_folds(
        len(target),
        fold_count,
        environments=environments,
        seed=seed,
    )
    all_indices = np.arange(len(target), dtype=int)
    weights = _sample_weights(sample_weight, len(target))
    scores: dict[float, float] = {}
    for alpha in alpha_grid:
        fold_scores = []
        for valid in folds:
            train = np.setdiff1d(all_indices, valid, assume_unique=True)
            intercept, coefficients = elastic_net_coordinate_descent(
                matrix[train],
                target[train],
                alpha=float(alpha),
                l1_ratio=l1_ratio,
                sample_weight=weights[train],
            )
            fold_scores.append(
                mean_squared_error(
                    target[valid],
                    prediction(intercept, coefficients, matrix[valid]),
                    sample_weight=weights[valid],
                )
            )
        scores[float(alpha)] = float(np.mean(fold_scores))
    best_score = min(scores.values())
    # Prefer the stronger penalty when scores are numerically tied.
    best = max(alpha for alpha, score in scores.items() if score <= best_score + 1.0e-12)
    return float(best), scores


def stratified_bootstrap_indices(
    environments: Sequence[str],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(environments, dtype=object)
    indices: list[int] = []
    for environment in sorted(set(values.tolist())):
        members = np.flatnonzero(values == environment)
        indices.extend(rng.choice(members, size=len(members), replace=True).tolist())
    return np.asarray(indices, dtype=int)


def ordinary_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float = 1.0e-10,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    design = np.column_stack((np.ones(x.shape[0]), x))
    weights = _sample_weights(sample_weight, y.size)
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_target = y * root_weight
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return float(coefficients[0]), coefficients[1:]


def _sample_weights(
    sample_weight: np.ndarray | None,
    sample_count: int,
) -> np.ndarray:
    if sample_weight is None:
        return np.ones(sample_count, dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (sample_count,):
        raise ValueError("sample weights have the wrong length")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample weights must be finite and non-negative")
    if float(np.sum(weights)) <= 0:
        raise ValueError("at least one sample weight must be positive")
    return weights


def bic_score(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> float:
    y = np.asarray(observed, dtype=float)
    residual_sum = float(np.sum((y - np.asarray(predicted, dtype=float)) ** 2))
    residual_sum = max(residual_sum, 1.0e-15)
    return float(y.size * np.log(residual_sum / y.size) + parameter_count * np.log(y.size))


__all__ = [
    "bic_score",
    "elastic_net_coordinate_descent",
    "make_folds",
    "mean_squared_error",
    "ordinary_least_squares",
    "prediction",
    "r2_score",
    "robust_scale",
    "select_elastic_net_alpha",
    "standard_scale",
    "stratified_bootstrap_indices",
]
