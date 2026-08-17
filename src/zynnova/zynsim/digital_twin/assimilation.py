"""Real-time sensor packets and ensemble Kalman data assimilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorPacket:
    time_s: float
    values: Mapping[str, float]
    standard_deviation: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_s < 0.0 or not self.values:
            raise ValueError("sensor packet time/values are invalid")
        if set(self.values) != set(self.standard_deviation):
            raise ValueError("sensor values and uncertainties must have the same keys")
        if any(value <= 0.0 for value in self.standard_deviation.values()):
            raise ValueError("sensor standard deviations must be positive")


ProcessModel = Callable[[np.ndarray, Mapping[str, float], float], np.ndarray]
ObservationModel = Callable[[np.ndarray, Sequence[str]], np.ndarray]


@dataclass(slots=True)
class EnsembleKalmanFilter:
    ensemble: np.ndarray
    process_model: ProcessModel
    observation_model: ObservationModel
    process_noise_covariance: np.ndarray
    random_seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ensemble = np.asarray(self.ensemble, dtype=float)
        if self.ensemble.ndim != 2 or self.ensemble.shape[0] < 3:
            raise ValueError("EnKF ensemble must have shape (members, state_dim) with >=3 members")
        self.process_noise_covariance = np.asarray(
            self.process_noise_covariance, dtype=float
        )
        state_dim = self.ensemble.shape[1]
        if self.process_noise_covariance.shape != (state_dim, state_dim):
            raise ValueError("process-noise covariance shape is inconsistent")
        self._rng = np.random.default_rng(self.random_seed)

    def predict(self, controls: Mapping[str, float], dt_s: float) -> np.ndarray:
        if dt_s <= 0.0:
            raise ValueError("EnKF prediction dt must be positive")
        propagated = np.asarray(
            [self.process_model(member, controls, dt_s) for member in self.ensemble],
            dtype=float,
        )
        noise = self._rng.multivariate_normal(
            np.zeros(self.ensemble.shape[1]),
            self.process_noise_covariance * dt_s,
            size=len(self.ensemble),
        )
        self.ensemble = propagated + noise
        return self.mean

    def update(self, packet: SensorPacket) -> np.ndarray:
        names = tuple(packet.values)
        predicted = np.asarray(
            [self.observation_model(member, names) for member in self.ensemble],
            dtype=float,
        )
        if predicted.shape != (len(self.ensemble), len(names)):
            raise ValueError("observation model returned an inconsistent shape")
        state_anomaly = self.ensemble - np.mean(self.ensemble, axis=0)
        observation_anomaly = predicted - np.mean(predicted, axis=0)
        denominator = max(len(self.ensemble) - 1, 1)
        cross_covariance = state_anomaly.T @ observation_anomaly / denominator
        observation_covariance = observation_anomaly.T @ observation_anomaly / denominator
        noise_covariance = np.diag(
            [packet.standard_deviation[name] ** 2 for name in names]
        )
        gain = cross_covariance @ np.linalg.pinv(
            observation_covariance + noise_covariance
        )
        observed = np.asarray([packet.values[name] for name in names], dtype=float)
        perturbations = self._rng.multivariate_normal(
            np.zeros(len(names)), noise_covariance, size=len(self.ensemble)
        )
        innovations = observed[None, :] + perturbations - predicted
        self.ensemble = self.ensemble + innovations @ gain.T
        return self.mean

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.ensemble, axis=0)

    @property
    def covariance(self) -> np.ndarray:
        return np.cov(self.ensemble, rowvar=False)


__all__ = ["EnsembleKalmanFilter", "SensorPacket"]
