from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

import numpy as np


def as_array(values: Any, *, dtype: Any = float, flatten: bool = True) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()
    array = np.asarray(values, dtype=dtype)
    return array.reshape(-1) if flatten else array


def finite_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = as_array(x)
    y_arr = as_array(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have equal shape; got {x_arr.shape} and {y_arr.shape}")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask], mask


def finite_columns(*columns: Any) -> tuple[list[np.ndarray], np.ndarray]:
    arrays = [as_array(column) for column in columns]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError("all columns must contain the same number of values")
    mask = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    return [array[mask] for array in arrays], mask


def group_values(values: Any, groups: Any | None = None) -> dict[str, np.ndarray]:
    values_arr = as_array(values)
    if groups is None:
        return {"all": values_arr[np.isfinite(values_arr)]}
    groups_arr = np.asarray(groups).reshape(-1)
    if groups_arr.size != values_arr.size:
        raise ValueError("groups and values must have equal length")
    result: dict[str, np.ndarray] = {}
    for group in dict.fromkeys(groups_arr.tolist()):
        mask = groups_arr == group
        result[str(group)] = values_arr[mask & np.isfinite(values_arr)]
    return result


def normalize_weights(weights: Any | None, size: int) -> np.ndarray:
    if weights is None:
        return np.full(size, 1.0 / max(size, 1), dtype=float)
    array = as_array(weights)
    if array.size != size:
        raise ValueError("weights must match data length")
    if np.any(array < 0):
        raise ValueError("weights must be non-negative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return array / total


def robust_limits(values: Any, *, quantiles: tuple[float, float] = (0.01, 0.99), pad: float = 0.05) -> tuple[float, float]:
    array = as_array(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(array, quantiles)
    if np.isclose(low, high):
        delta = max(abs(low), 1.0) * 0.1
        return float(low - delta), float(high + delta)
    margin = (high - low) * pad
    return float(low - margin), float(high + margin)


def density_values(x: Any, y: Any | None = None, *, bandwidth: str | float = "scott") -> np.ndarray:
    x_arr = as_array(x)
    points = x_arr[None, :] if y is None else np.vstack([x_arr, as_array(y)])
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(points, bw_method=bandwidth)
        return np.asarray(kde(points), dtype=float)
    except Exception:
        if y is None:
            hist, edges = np.histogram(x_arr, bins="auto", density=True)
            indices = np.clip(np.searchsorted(edges, x_arr, side="right") - 1, 0, len(hist) - 1)
            return hist[indices]
        y_arr = as_array(y)
        hist, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=40)
        ix = np.clip(np.searchsorted(x_edges, x_arr, side="right") - 1, 0, hist.shape[0] - 1)
        iy = np.clip(np.searchsorted(y_edges, y_arr, side="right") - 1, 0, hist.shape[1] - 1)
        return hist[ix, iy]


def regression_metrics(reference: Any, prediction: Any) -> dict[str, float]:
    y_true, y_pred, _ = finite_xy(reference, prediction)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error))) if error.size else float("nan")
    rmse = float(np.sqrt(np.mean(error**2))) if error.size else float("nan")
    bias = float(np.mean(error)) if error.size else float("nan")
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if error.size > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "r2": r2, "pearson_r": pearson, "n": int(error.size)}


def classification_metrics(y_true: Any, probability: Any, *, threshold: float = 0.5) -> dict[str, float]:
    truth = as_array(y_true, dtype=int)
    prob = as_array(probability)
    mask = np.isfinite(prob)
    truth, prob = truth[mask], prob[mask]
    predicted = prob >= threshold
    tp = int(np.sum((truth == 1) & predicted))
    tn = int(np.sum((truth == 0) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(truth.size, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_interval(
    values: Any,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    array = as_array(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        estimates[index] = statistic(rng.choice(array, size=array.size, replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(statistic(array)), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def pareto_mask(values: Any, *, maximize: Sequence[bool] | bool = True) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (n_samples, n_objectives)")
    directions = np.full(matrix.shape[1], bool(maximize)) if isinstance(maximize, bool) else np.asarray(maximize, dtype=bool)
    if directions.size != matrix.shape[1]:
        raise ValueError("maximize must specify one direction per objective")
    transformed = matrix.copy()
    transformed[:, ~directions] *= -1.0
    efficient = np.ones(matrix.shape[0], dtype=bool)
    for i, candidate in enumerate(transformed):
        if not efficient[i]:
            continue
        dominates_i = np.all(transformed[efficient] >= candidate, axis=1) & np.any(transformed[efficient] > candidate, axis=1)
        indices = np.flatnonzero(efficient)
        if np.any(dominates_i):
            efficient[i] = False
            continue
        dominated_by_i = np.all(candidate >= transformed[efficient], axis=1) & np.any(candidate > transformed[efficient], axis=1)
        efficient[indices[dominated_by_i]] = False
        efficient[i] = True
    return efficient


def rolling_statistics(values: Any, window: int) -> tuple[np.ndarray, np.ndarray]:
    array = as_array(values)
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1:
        return array.copy(), np.zeros_like(array)
    kernel = np.ones(window) / window
    mean = np.convolve(array, kernel, mode="valid")
    squared = np.convolve(array**2, kernel, mode="valid")
    std = np.sqrt(np.maximum(squared - mean**2, 0.0))
    return mean, std


def make_labels(labels: Sequence[Any] | None, size: int, prefix: str = "Series") -> list[str]:
    if labels is None:
        return [f"{prefix} {index + 1}" for index in range(size)]
    if len(labels) != size:
        raise ValueError(f"expected {size} labels, received {len(labels)}")
    return [str(label) for label in labels]


def validate_square(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return array


def bin_centers(edges: Any) -> np.ndarray:
    array = as_array(edges)
    return 0.5 * (array[:-1] + array[1:])


def safe_log(values: Any, *, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(as_array(values), floor, None))


def format_metric(value: float, *, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 10 ** (-precision) or magnitude >= 10 ** (precision + 2)):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"
