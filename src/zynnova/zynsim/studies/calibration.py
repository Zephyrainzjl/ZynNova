"""Bounded nonlinear parameter identification with uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .parameters import get_parameter, replace_parameters


PredictionFunction = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class CalibrationData:
    coordinates: np.ndarray
    observed: np.ndarray
    standard_deviation: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=np.float64)
        uncertainty = np.broadcast_to(
            np.asarray(self.standard_deviation, dtype=np.float64),
            observed.shape,
        )
        if (
            coordinates.ndim != 1
            or observed.shape != coordinates.shape
            or len(coordinates) < 2
            or not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise ValueError("calibration coordinates, observations, or uncertainty are invalid")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "standard_deviation", uncertainty)


@dataclass(frozen=True, slots=True)
class CalibrationParameter:
    path: str
    lower: float
    upper: float
    initial: float | None = None
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.path or not np.isfinite((self.lower, self.upper)).all():
            raise ValueError("calibration parameter path and bounds are required")
        if self.lower >= self.upper or self.scale not in {"linear", "log"}:
            raise ValueError("calibration bounds or scale are invalid")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError("log-scaled calibration bounds must be positive")
        if self.initial is not None and not self.lower <= self.initial <= self.upper:
            raise ValueError("calibration initial value lies outside bounds")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    parameters: Any
    values: dict[str, float]
    residuals: np.ndarray
    covariance: np.ndarray
    standard_errors: dict[str, float]
    rmse: float
    success: bool
    message: str
    evaluations: int


class ParameterCalibrator:
    """Fit arbitrary nested simulation parameters to scalar observations."""

    def __init__(
        self,
        base_parameters: Any,
        parameters: tuple[CalibrationParameter, ...],
        prediction: PredictionFunction,
    ) -> None:
        if not parameters:
            raise ValueError("at least one calibration parameter is required")
        self.base_parameters = base_parameters
        self.parameters = tuple(parameters)
        self.prediction = prediction
        if not callable(prediction):
            raise TypeError("prediction must be callable")

    def fit(
        self,
        data: CalibrationData,
        *,
        loss: str = "linear",
        maximum_evaluations: int = 200,
    ) -> CalibrationResult:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ImportError("parameter calibration requires SciPy") from exc
        if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unsupported robust least-squares loss")
        if maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive")
        lower = np.asarray([self._encode(item.lower, item) for item in self.parameters])
        upper = np.asarray([self._encode(item.upper, item) for item in self.parameters])
        initial = []
        for item in self.parameters:
            value = (
                float(get_parameter(self.base_parameters, item.path))
                if item.initial is None
                else item.initial
            )
            if not item.lower <= value <= item.upper:
                raise ValueError(f"base value for {item.path!r} lies outside bounds")
            initial.append(self._encode(value, item))

        def residual(encoded: np.ndarray) -> np.ndarray:
            physical = self._decode_vector(encoded)
            candidate = replace_parameters(
                self.base_parameters,
                {
                    item.path: physical[index]
                    for index, item in enumerate(self.parameters)
                },
            )
            predicted = np.asarray(
                self.prediction(candidate, data.coordinates),
                dtype=np.float64,
            )
            if predicted.shape != data.observed.shape or not np.all(np.isfinite(predicted)):
                raise ValueError("prediction returned invalid calibration values")
            return (predicted - data.observed) / data.standard_deviation

        optimized = least_squares(
            residual,
            np.asarray(initial),
            bounds=(lower, upper),
            loss=loss,
            max_nfev=maximum_evaluations,
        )
        physical = self._decode_vector(optimized.x)
        values = {
            item.path: float(physical[index])
            for index, item in enumerate(self.parameters)
        }
        fitted = replace_parameters(self.base_parameters, values)
        weighted_residual = np.asarray(optimized.fun, dtype=np.float64)
        raw_residual = weighted_residual * np.asarray(data.standard_deviation)
        covariance = _least_squares_covariance(optimized.jac, weighted_residual)
        transform = np.diag(
            [
                physical[index] if item.scale == "log" else 1.0
                for index, item in enumerate(self.parameters)
            ]
        )
        covariance_physical = transform @ covariance @ transform.T
        errors = np.sqrt(np.maximum(np.diag(covariance_physical), 0.0))
        return CalibrationResult(
            parameters=fitted,
            values=values,
            residuals=raw_residual,
            covariance=covariance_physical,
            standard_errors={
                item.path: float(errors[index])
                for index, item in enumerate(self.parameters)
            },
            rmse=float(np.sqrt(np.mean(raw_residual**2))),
            success=bool(optimized.success),
            message=str(optimized.message),
            evaluations=int(optimized.nfev),
        )

    def _decode_vector(self, encoded: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.exp(encoded[index]) if item.scale == "log" else encoded[index]
                for index, item in enumerate(self.parameters)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _encode(value: float, parameter: CalibrationParameter) -> float:
        return float(np.log(value) if parameter.scale == "log" else value)


def _least_squares_covariance(
    jacobian: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2:
        raise ValueError("least-squares Jacobian must be a matrix")
    degrees = max(jacobian.shape[0] - jacobian.shape[1], 1)
    variance = float(np.dot(residual, residual) / degrees)
    return variance * np.linalg.pinv(jacobian.T @ jacobian)


__all__ = [
    "CalibrationData",
    "CalibrationParameter",
    "CalibrationResult",
    "ParameterCalibrator",
    "PredictionFunction",
]
