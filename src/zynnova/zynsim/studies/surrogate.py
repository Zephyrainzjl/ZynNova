"""Reduced-order and polynomial surrogate models for repeated studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PODFit:
    mean: FloatArray
    modes: FloatArray
    singular_values: FloatArray
    retained_energy: float


class PODReducer:
    """Proper orthogonal decomposition for vector-valued simulation states."""

    def __init__(
        self,
        rank: int | None = None,
        *,
        energy_fraction: float = 0.999,
    ) -> None:
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
                raise TypeError("rank must be an integer")
            if rank <= 0:
                raise ValueError("rank must be positive")
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError("energy_fraction must lie in (0, 1]")
        self.rank = rank
        self.energy_fraction = energy_fraction
        self.fit_: PODFit | None = None

    def fit(self, snapshots: ArrayLike) -> PODReducer:
        matrix = _matrix(snapshots, "snapshots")
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        energy = singular_values * singular_values
        if self.rank is None:
            if np.sum(energy) <= np.finfo(float).eps:
                retained_rank = 1
            else:
                cumulative = np.cumsum(energy) / np.sum(energy)
                retained_rank = int(np.searchsorted(cumulative, self.energy_fraction) + 1)
        else:
            retained_rank = min(self.rank, right_vectors.shape[0])
        total_energy = float(np.sum(energy))
        retained_energy = (
            1.0
            if total_energy <= np.finfo(float).eps
            else float(np.sum(energy[:retained_rank]) / total_energy)
        )
        self.fit_ = PODFit(
            mean=np.asarray(mean, dtype=float),
            modes=np.asarray(right_vectors[:retained_rank], dtype=float),
            singular_values=np.asarray(singular_values[:retained_rank], dtype=float),
            retained_energy=retained_energy,
        )
        return self

    def encode(self, snapshots: ArrayLike) -> FloatArray:
        fit = self._require_fit()
        matrix = _matrix(snapshots, "snapshots")
        if matrix.shape[1] != fit.mean.size:
            raise ValueError("snapshot state dimension differs from fitted POD model")
        return np.asarray((matrix - fit.mean) @ fit.modes.T, dtype=float)

    def decode(self, coordinates: ArrayLike) -> FloatArray:
        fit = self._require_fit()
        matrix = _matrix(coordinates, "coordinates")
        if matrix.shape[1] != fit.modes.shape[0]:
            raise ValueError("coordinate dimension differs from retained POD rank")
        return np.asarray(matrix @ fit.modes + fit.mean, dtype=float)

    def fit_transform(self, snapshots: ArrayLike) -> FloatArray:
        return self.fit(snapshots).encode(snapshots)

    def _require_fit(self) -> PODFit:
        if self.fit_ is None:
            raise RuntimeError("PODReducer must be fitted first")
        return self.fit_


@dataclass(frozen=True)
class PolynomialFit:
    coefficients: FloatArray
    input_mean: FloatArray
    input_scale: FloatArray
    output_mean: FloatArray
    output_scale: FloatArray
    feature_powers: tuple[tuple[int, int], ...]


class PolynomialSurrogate:
    """Ridge-regularized polynomial response surface of degree one or two."""

    def __init__(self, *, degree: int = 2, ridge: float = 1.0e-10) -> None:
        if degree not in {1, 2}:
            raise ValueError("degree must be 1 or 2")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.degree = degree
        self.ridge = ridge
        self.fit_: PolynomialFit | None = None

    def fit(self, inputs: ArrayLike, outputs: ArrayLike) -> PolynomialSurrogate:
        x = _matrix(inputs, "inputs")
        y = _matrix(outputs, "outputs")
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and outputs must contain the same samples")
        input_mean = np.mean(x, axis=0)
        input_scale = np.std(x, axis=0)
        input_scale[input_scale <= np.finfo(float).eps] = 1.0
        output_mean = np.mean(y, axis=0)
        output_scale = np.std(y, axis=0)
        output_scale[output_scale <= np.finfo(float).eps] = 1.0

        normalized_x = (x - input_mean) / input_scale
        normalized_y = (y - output_mean) / output_scale
        features, powers = _polynomial_features(normalized_x, self.degree)
        normal = features.T @ features
        regularizer = self.ridge * np.eye(normal.shape[0])
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(
            normal + regularizer, features.T @ normalized_y
        )
        self.fit_ = PolynomialFit(
            coefficients=np.asarray(coefficients, dtype=float),
            input_mean=np.asarray(input_mean, dtype=float),
            input_scale=np.asarray(input_scale, dtype=float),
            output_mean=np.asarray(output_mean, dtype=float),
            output_scale=np.asarray(output_scale, dtype=float),
            feature_powers=powers,
        )
        return self

    def predict(self, inputs: ArrayLike) -> FloatArray:
        fit = self._require_fit()
        x = _matrix(inputs, "inputs")
        if x.shape[1] != fit.input_mean.size:
            raise ValueError("input dimension differs from fitted surrogate")
        normalized_x = (x - fit.input_mean) / fit.input_scale
        features, powers = _polynomial_features(normalized_x, self.degree)
        if powers != fit.feature_powers:
            raise RuntimeError("surrogate feature layout changed unexpectedly")
        normalized_y = features @ fit.coefficients
        return np.asarray(
            normalized_y * fit.output_scale + fit.output_mean, dtype=float
        )

    def score(self, inputs: ArrayLike, outputs: ArrayLike) -> FloatArray:
        """Return one coefficient of determination per output column."""

        observed = _matrix(outputs, "outputs")
        predicted = self.predict(inputs)
        if predicted.shape != observed.shape:
            raise ValueError("predicted and observed output shapes differ")
        residual = np.sum((observed - predicted) ** 2, axis=0)
        total = np.sum((observed - np.mean(observed, axis=0)) ** 2, axis=0)
        return np.asarray(
            np.where(
                total <= np.finfo(float).eps,
                np.where(residual <= np.finfo(float).eps, 1.0, 0.0),
                1.0 - residual / total,
            ),
            dtype=float,
        )

    def _require_fit(self) -> PolynomialFit:
        if self.fit_ is None:
            raise RuntimeError("PolynomialSurrogate must be fitted first")
        return self.fit_


def _polynomial_features(
    inputs: FloatArray, degree: int
) -> tuple[FloatArray, tuple[tuple[int, int], ...]]:
    columns = [np.ones(inputs.shape[0], dtype=float)]
    powers: list[tuple[int, int]] = [(-1, -1)]
    for index in range(inputs.shape[1]):
        columns.append(inputs[:, index])
        powers.append((index, -1))
    if degree == 2:
        for left in range(inputs.shape[1]):
            for right in range(left, inputs.shape[1]):
                columns.append(inputs[:, left] * inputs[:, right])
                powers.append((left, right))
    return np.column_stack(columns), tuple(powers)


def _matrix(values: ArrayLike, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty one- or two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    return matrix
