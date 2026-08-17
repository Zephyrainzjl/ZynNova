from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Sequence

import numpy as np

from .schema import (
    DynamicsDiscoveryReport,
    DynamicsEquation,
)


class SparseDynamicsDiscoverer:
    """Discover sparse relaxation or transport equations from trajectories.

    The native weak form regresses state increments against time-integrated
    candidate functions. It therefore avoids differentiating a noisy polymer
    trajectory. PySINDy is available as an optional conventional SINDy backend.
    """

    def __init__(
        self,
        *,
        polynomial_degree: int = 2,
        threshold: float = 0.05,
        ridge: float = 1.0e-8,
        maximum_iterations: int = 12,
        integration_window: int = 7,
    ) -> None:
        if polynomial_degree not in {1, 2, 3}:
            raise ValueError("polynomial_degree must be 1, 2, or 3")
        if threshold < 0 or ridge < 0:
            raise ValueError("threshold and ridge must be non-negative")
        if maximum_iterations < 1 or integration_window < 2:
            raise ValueError("iteration and integration limits must be positive")
        self.polynomial_degree = int(polynomial_degree)
        self.threshold = float(threshold)
        self.ridge = float(ridge)
        self.maximum_iterations = int(maximum_iterations)
        self.integration_window = int(integration_window)

    def discover(
        self,
        time: Sequence[float] | np.ndarray,
        states: Sequence[Sequence[float]] | np.ndarray,
        *,
        state_names: Sequence[str] | None = None,
        time_name: str = "time",
        backend: str = "auto",
        weak_form: bool = True,
    ) -> DynamicsDiscoveryReport:
        time_array, state_matrix, names = _validate_trajectory(
            time,
            states,
            state_names,
        )
        backend = str(backend).lower()
        if backend not in {"auto", "native", "pysindy"}:
            raise ValueError(f"unknown dynamics backend: {backend}")

        warnings: list[str] = []
        if backend == "auto":
            # Integral regression is deliberately preferred for noisy molecular
            # trajectories. PySINDy remains explicit for reproducible opt-in.
            resolved_backend = "native-integral" if weak_form else (
                "pysindy"
                if importlib.util.find_spec("pysindy") is not None
                else "native-derivative"
            )
        elif backend == "native":
            resolved_backend = "native-integral" if weak_form else "native-derivative"
        else:
            if weak_form:
                raise ValueError(
                    "backend='pysindy' currently supports derivative SINDy; "
                    "use backend='native', weak_form=True for integral weak SINDy"
                )
            resolved_backend = "pysindy"

        if resolved_backend == "pysindy":
            try:
                return self._discover_pysindy(
                    time_array,
                    state_matrix,
                    names,
                    time_name=time_name,
                )
            except ImportError:
                if backend == "pysindy":
                    raise
                warnings.append(
                    "PySINDy is unavailable; using native derivative SINDy."
                )
                resolved_backend = "native-derivative"

        library, terms = _polynomial_library(
            state_matrix,
            names,
            degree=self.polynomial_degree,
        )
        if resolved_backend == "native-integral":
            design, response = _integral_regression(
                time_array,
                state_matrix,
                library,
                window=self.integration_window,
            )
            weak = True
        else:
            design = library
            response = np.column_stack(
                [
                    np.gradient(state_matrix[:, index], time_array)
                    for index in range(state_matrix.shape[1])
                ]
            )
            weak = False

        coefficients = _sequential_thresholded_least_squares(
            design,
            response,
            threshold=self.threshold,
            ridge=self.ridge,
            maximum_iterations=self.maximum_iterations,
        )
        prediction = design @ coefficients.T
        equations = tuple(
            _dynamics_equation(
                state=state,
                terms=terms,
                coefficients=coefficients[index],
                observed=response[:, index],
                predicted=prediction[:, index],
            )
            for index, state in enumerate(names)
        )
        return DynamicsDiscoveryReport(
            time_name=str(time_name),
            state_names=names,
            equations=equations,
            backend=resolved_backend,
            weak_form=weak,
            sample_count=len(time_array),
            diagnostics={
                "regression_rows": int(len(design)),
                "candidate_terms": len(terms),
                "polynomial_degree": self.polynomial_degree,
                "threshold": self.threshold,
                "ridge": self.ridge,
                "integration_window": (
                    self.integration_window if weak else None
                ),
                "validation_note": (
                    "R2 is measured on integrated state increments"
                    if weak
                    else "R2 is measured on numerically differentiated states"
                ),
            },
            warnings=tuple(warnings),
        )

    def _discover_pysindy(
        self,
        time: np.ndarray,
        states: np.ndarray,
        names: tuple[str, ...],
        *,
        time_name: str,
    ) -> DynamicsDiscoveryReport:
        try:
            import pysindy as ps
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PySINDy is not installed; install "
                "zynnova[physics-symbolic]"
            ) from exc

        optimizer = ps.STLSQ(
            threshold=self.threshold,
            alpha=self.ridge,
            max_iter=self.maximum_iterations,
        )
        library = ps.PolynomialLibrary(
            degree=self.polynomial_degree,
            include_bias=True,
        )
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=ps.SmoothedFiniteDifference(),
            feature_names=list(names),
        )
        model.fit(states, t=time)
        coefficients = np.asarray(model.coefficients(), dtype=float)
        terms = tuple(str(term) for term in model.get_feature_names())
        score = float(model.score(states, t=time))
        expressions = tuple(str(value) for value in model.equations())
        equations = tuple(
            DynamicsEquation(
                state=state,
                expression=f"d({state})/d({time_name}) = {expressions[index]}",
                coefficients=tuple(
                    float(value) for value in coefficients[index]
                ),
                terms=terms,
                train_r2=score,
                complexity=int(
                    np.count_nonzero(np.abs(coefficients[index]) > 0.0)
                ),
            )
            for index, state in enumerate(names)
        )
        return DynamicsDiscoveryReport(
            time_name=str(time_name),
            state_names=names,
            equations=equations,
            backend="pysindy",
            weak_form=False,
            sample_count=len(time),
            diagnostics={
                "candidate_terms": len(terms),
                "polynomial_degree": self.polynomial_degree,
                "threshold": self.threshold,
                "package_version": _package_version("pysindy"),
                "differentiation": "SmoothedFiniteDifference",
            },
        )


def discover_sparse_dynamics(
    time: Sequence[float] | np.ndarray,
    states: Sequence[Sequence[float]] | np.ndarray,
    *,
    state_names: Sequence[str] | None = None,
    time_name: str = "time",
    backend: str = "auto",
    weak_form: bool = True,
    polynomial_degree: int = 2,
    threshold: float = 0.05,
    ridge: float = 1.0e-8,
    maximum_iterations: int = 12,
    integration_window: int = 7,
) -> DynamicsDiscoveryReport:
    """Functional entry point for sparse polymer-dynamics discovery."""

    return SparseDynamicsDiscoverer(
        polynomial_degree=polynomial_degree,
        threshold=threshold,
        ridge=ridge,
        maximum_iterations=maximum_iterations,
        integration_window=integration_window,
    ).discover(
        time,
        states,
        state_names=state_names,
        time_name=time_name,
        backend=backend,
        weak_form=weak_form,
    )


def _validate_trajectory(
    time: Sequence[float] | np.ndarray,
    states: Sequence[Sequence[float]] | np.ndarray,
    state_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    time_array = np.asarray(time, dtype=float).reshape(-1)
    state_matrix = np.asarray(states, dtype=float)
    if state_matrix.ndim == 1:
        state_matrix = state_matrix[:, None]
    if state_matrix.ndim != 2 or len(state_matrix) != len(time_array):
        raise ValueError("states must have shape (time samples, state variables)")
    if len(time_array) < 12:
        raise ValueError("at least 12 trajectory samples are required")
    if np.any(~np.isfinite(time_array)) or np.any(~np.isfinite(state_matrix)):
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing")
    names = (
        tuple(f"x{index}" for index in range(state_matrix.shape[1]))
        if state_names is None
        else tuple(str(name) for name in state_names)
    )
    if len(names) != state_matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("state_names must be unique and match the state matrix")
    return time_array, state_matrix, names


def _polynomial_library(
    states: np.ndarray,
    names: tuple[str, ...],
    *,
    degree: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = [np.ones(len(states), dtype=float)]
    terms = ["1"]
    for index, name in enumerate(names):
        columns.append(states[:, index])
        terms.append(name)
    if degree >= 2:
        for left in range(len(names)):
            for right in range(left, len(names)):
                columns.append(states[:, left] * states[:, right])
                terms.append(
                    f"{names[left]}^2"
                    if left == right
                    else f"{names[left]}*{names[right]}"
                )
    if degree >= 3:
        for first in range(len(names)):
            for second in range(first, len(names)):
                for third in range(second, len(names)):
                    columns.append(
                        states[:, first]
                        * states[:, second]
                        * states[:, third]
                    )
                    terms.append(
                        "*".join(
                            (names[first], names[second], names[third])
                        )
                    )
    return np.column_stack(columns), tuple(terms)


def _integral_regression(
    time: np.ndarray,
    states: np.ndarray,
    library: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    window = min(max(int(window), 2), len(time) - 1)
    designs = []
    responses = []
    for start in range(len(time) - window):
        stop = start + window
        designs.append(
            _trapezoidal_integral(
                library[start : stop + 1],
                time[start : stop + 1],
            )
        )
        responses.append(states[stop] - states[start])
    return np.asarray(designs, dtype=float), np.asarray(responses, dtype=float)


def _trapezoidal_integral(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return np.asarray(trapezoid(values, x=time, axis=0), dtype=float)
    return np.asarray(np.trapz(values, x=time, axis=0), dtype=float)


def _sequential_thresholded_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    ridge: float,
    maximum_iterations: int,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    feature_scale = np.sqrt(np.mean(matrix**2, axis=0))
    feature_scale[feature_scale < 1.0e-12] = 1.0
    target_scale = np.sqrt(np.mean(target**2, axis=0))
    target_scale[target_scale < 1.0e-12] = 1.0
    normalized_matrix = matrix / feature_scale
    normalized_target = target / target_scale
    result = np.zeros((target.shape[1], matrix.shape[1]), dtype=float)
    for output in range(target.shape[1]):
        active = np.ones(matrix.shape[1], dtype=bool)
        coefficients = np.zeros(matrix.shape[1], dtype=float)
        for _ in range(maximum_iterations):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            design = normalized_matrix[:, indices]
            gram = design.T @ design
            regularizer = ridge * np.eye(len(indices), dtype=float)
            try:
                fitted = np.linalg.solve(
                    gram + regularizer,
                    design.T @ normalized_target[:, output],
                )
            except np.linalg.LinAlgError:
                fitted = np.linalg.lstsq(
                    design,
                    normalized_target[:, output],
                    rcond=None,
                )[0]
            coefficients[:] = 0.0
            coefficients[indices] = fitted
            retained = np.abs(coefficients) >= threshold
            if np.array_equal(retained, active):
                break
            active = retained
        result[output] = coefficients * target_scale[output] / feature_scale
    return result


def _dynamics_equation(
    *,
    state: str,
    terms: tuple[str, ...],
    coefficients: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> DynamicsEquation:
    active = np.flatnonzero(np.abs(coefficients) > 0.0)
    pieces = []
    for position, index in enumerate(active):
        coefficient = float(coefficients[index])
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        token = f"{magnitude:.8g}" if terms[index] == "1" else (
            f"{magnitude:.8g}*{terms[index]}"
        )
        if position == 0:
            pieces.append(token if sign == "+" else f"-{token}")
        else:
            pieces.append(f" {sign} {token}")
    expression = "".join(pieces) if pieces else "0"
    return DynamicsEquation(
        state=state,
        expression=f"d({state})/dt = {expression}",
        coefficients=tuple(float(value) for value in coefficients),
        terms=terms,
        train_r2=_r2(observed, predicted),
        complexity=len(active),
    )


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return float("nan") if total < 1.0e-15 else 1.0 - residual / total


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "SparseDynamicsDiscoverer",
    "discover_sparse_dynamics",
]
