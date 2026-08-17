from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .schema import InteractionDecomposition, InteractionEdge


@dataclass(slots=True)
class QuadraticInteractionOracle:
    """Deterministic fallback oracle for Hessian interaction discovery."""

    ridge: float = 1.0e-5
    coefficients_: np.ndarray | None = None
    feature_count_: int = 0
    validation_r2_: float = float("nan")

    def fit(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        train_indices: np.ndarray | None = None,
        validation_indices: np.ndarray | None = None,
    ) -> QuadraticInteractionOracle:
        matrix = _matrix(matrix)
        target = _target(target, len(matrix))
        if train_indices is None:
            train_indices = np.arange(len(matrix))
        design = _quadratic_design(matrix)
        train_design = design[train_indices]
        train_target = target[train_indices]
        penalty = np.eye(train_design.shape[1], dtype=float) * float(self.ridge)
        penalty[0, 0] = 0.0
        gram = train_design.T @ train_design + penalty
        right = train_design.T @ train_target
        try:
            self.coefficients_ = np.linalg.solve(gram, right)
        except np.linalg.LinAlgError:
            self.coefficients_ = np.linalg.lstsq(
                train_design,
                train_target,
                rcond=None,
            )[0]
        self.feature_count_ = matrix.shape[1]
        if validation_indices is not None and len(validation_indices):
            prediction = design[validation_indices] @ self.coefficients_
            self.validation_r2_ = _r2(target[validation_indices], prediction)
        else:
            self.validation_r2_ = _r2(target, design @ self.coefficients_)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("fit the quadratic oracle before prediction")
        matrix = _matrix(matrix)
        if matrix.shape[1] != self.feature_count_:
            raise ValueError("oracle feature count mismatch")
        return _quadratic_design(matrix) @ self.coefficients_


class HessianInteractionDecomposer:
    """Estimate a mixed-derivative interaction graph by finite differences."""

    def __init__(
        self,
        *,
        step: float = 0.05,
        sample_count: int = 64,
        relative_threshold: float = 0.10,
        score_quantile: float = 0.70,
        random_seed: int = 42,
    ) -> None:
        if step <= 0 or sample_count < 4:
            raise ValueError("step and sample_count must be positive")
        if not 0.0 <= relative_threshold <= 1.0:
            raise ValueError("relative_threshold must lie in [0, 1]")
        if not 0.0 <= score_quantile <= 1.0:
            raise ValueError("score_quantile must lie in [0, 1]")
        self.step = float(step)
        self.sample_count = int(sample_count)
        self.relative_threshold = float(relative_threshold)
        self.score_quantile = float(score_quantile)
        self.random_seed = int(random_seed)

    def decompose(
        self,
        matrix: np.ndarray,
        feature_names: Sequence[str],
        predictor: Callable[[np.ndarray], np.ndarray],
        *,
        oracle_name: str,
        oracle_validation_r2: float,
        hessian_provider: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> InteractionDecomposition:
        matrix = _matrix(matrix)
        names = tuple(str(name) for name in feature_names)
        if matrix.shape[1] != len(names):
            raise ValueError("feature_names length does not match the matrix")
        sample = self._sample(matrix)
        if hessian_provider is None:
            absolute, signed = self._mixed_hessian(sample, predictor)
        else:
            absolute, signed = _summarize_hessian(
                hessian_provider(sample),
                sample_count=len(sample),
                feature_count=len(names),
            )
        return self._report(
            names,
            absolute,
            signed,
            oracle_name=oracle_name,
            oracle_validation_r2=oracle_validation_r2,
            sample_count=len(sample),
        )

    def _report(
        self,
        names: tuple[str, ...],
        absolute: np.ndarray,
        signed: np.ndarray,
        *,
        oracle_name: str,
        oracle_validation_r2: float,
        sample_count: int,
    ) -> InteractionDecomposition:
        off_diagonal = absolute[np.triu_indices(len(names), k=1)]
        positive = off_diagonal[off_diagonal > 1.0e-12]
        if len(positive):
            threshold = max(
                float(np.max(positive)) * self.relative_threshold,
                float(np.quantile(positive, self.score_quantile)),
            )
        else:
            threshold = float("inf")
        edges = []
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                score = float(absolute[left, right])
                if score + 1.0e-15 < threshold:
                    continue
                edges.append(
                    InteractionEdge(
                        left=names[left],
                        right=names[right],
                        score=score,
                        signed_score=float(signed[left, right]),
                    )
                )
        edges.sort(key=lambda edge: (-edge.score, edge.left, edge.right))
        components = _connected_components(names, edges)
        return InteractionDecomposition(
            feature_names=names,
            score_matrix=tuple(
                tuple(float(value) for value in row)
                for row in absolute
            ),
            signed_matrix=tuple(
                tuple(float(value) for value in row)
                for row in signed
            ),
            edges=tuple(edges),
            components=components,
            oracle=str(oracle_name),
            oracle_validation_r2=float(oracle_validation_r2),
            threshold=threshold,
            sample_count=sample_count,
        )

    def _sample(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) <= self.sample_count:
            return matrix.copy()
        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(len(matrix), size=self.sample_count, replace=False)
        return matrix[np.sort(indices)]

    def _mixed_hessian(
        self,
        sample: np.ndarray,
        predictor: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_count = sample.shape[1]
        absolute = np.zeros((feature_count, feature_count), dtype=float)
        signed = np.zeros_like(absolute)
        step = self.step
        for left in range(feature_count):
            for right in range(left + 1, feature_count):
                plus_plus = sample.copy()
                plus_minus = sample.copy()
                minus_plus = sample.copy()
                minus_minus = sample.copy()
                plus_plus[:, left] += step
                plus_plus[:, right] += step
                plus_minus[:, left] += step
                plus_minus[:, right] -= step
                minus_plus[:, left] -= step
                minus_plus[:, right] += step
                minus_minus[:, left] -= step
                minus_minus[:, right] -= step
                derivative = (
                    _prediction(predictor, plus_plus)
                    - _prediction(predictor, plus_minus)
                    - _prediction(predictor, minus_plus)
                    + _prediction(predictor, minus_minus)
                ) / (4.0 * step**2)
                finite = derivative[np.isfinite(derivative)]
                if not len(finite):
                    continue
                absolute_score = float(np.median(np.abs(finite)))
                signed_score = float(np.median(finite))
                absolute[left, right] = absolute[right, left] = absolute_score
                signed[left, right] = signed[right, left] = signed_score
        return absolute, signed


def _connected_components(
    names: tuple[str, ...],
    edges: Sequence[InteractionEdge],
) -> tuple[tuple[str, ...], ...]:
    parent = list(range(len(names)))
    positions = {name: index for index, name in enumerate(names)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        union(positions[edge.left], positions[edge.right])
    groups: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(find(index), []).append(name)
    result = [tuple(group) for group in groups.values()]
    result.sort(key=lambda group: (-len(group), group))
    return tuple(result)


def _quadratic_design(matrix: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(matrix), dtype=float)]
    columns.extend(matrix[:, index] for index in range(matrix.shape[1]))
    columns.extend(matrix[:, index] ** 2 for index in range(matrix.shape[1]))
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            columns.append(matrix[:, left] * matrix[:, right])
    return np.column_stack(columns)


def _prediction(
    predictor: Callable[[np.ndarray], np.ndarray],
    matrix: np.ndarray,
) -> np.ndarray:
    values = np.asarray(predictor(matrix), dtype=float).reshape(-1)
    if len(values) != len(matrix):
        raise ValueError("predictor returned the wrong number of values")
    return values


def _summarize_hessian(
    values: np.ndarray,
    *,
    sample_count: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    hessian = np.asarray(values, dtype=float)
    expected = (sample_count, feature_count, feature_count)
    if hessian.shape != expected:
        raise ValueError(
            f"hessian provider returned shape {hessian.shape}; expected {expected}"
        )
    hessian = 0.5 * (hessian + np.swapaxes(hessian, 1, 2))
    hessian = np.where(np.isfinite(hessian), hessian, np.nan)
    absolute = np.nanmedian(np.abs(hessian), axis=0)
    signed = np.nanmedian(hessian, axis=0)
    absolute = np.nan_to_num(absolute, nan=0.0, posinf=0.0, neginf=0.0)
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(absolute, 0.0)
    np.fill_diagonal(signed, 0.0)
    return absolute, signed


def _matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("matrix must have shape (samples, features)")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    return matrix


def _target(values: np.ndarray, sample_count: int) -> np.ndarray:
    target = np.asarray(values, dtype=float).reshape(-1)
    if len(target) != sample_count or np.any(~np.isfinite(target)):
        raise ValueError("target must be finite and match the matrix")
    return target


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return float("nan") if total < 1.0e-15 else 1.0 - residual / total


__all__ = [
    "HessianInteractionDecomposer",
    "QuadraticInteractionOracle",
]
