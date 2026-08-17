"""Cell-to-pack digital-twin orchestration with assimilation and control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .assimilation import EnsembleKalmanFilter, SensorPacket
from .control import ClosedLoopController, ControlAction, ControlDecision
from .estimation import HealthEstimate, OnlineHealthEstimator
from .risk import ProbabilisticRiskModel, RiskForecast


@dataclass(slots=True)
class DigitalTwinRecord:
    time_s: float
    state_mean: np.ndarray
    state_covariance: np.ndarray
    health: HealthEstimate
    control: ControlDecision | None
    risk: RiskForecast | None
    measurements: Mapping[str, float] = field(default_factory=dict)


class BatteryDigitalTwin:
    def __init__(
        self,
        filter_: EnsembleKalmanFilter,
        health_estimator: OnlineHealthEstimator,
        *,
        controller: ClosedLoopController | None = None,
        risk_model: ProbabilisticRiskModel | None = None,
        control_predictor: Any | None = None,
        risk_simulator: Any | None = None,
    ) -> None:
        self.filter = filter_
        self.health_estimator = health_estimator
        self.controller = controller
        self.risk_model = risk_model
        self.control_predictor = control_predictor
        self.risk_simulator = risk_simulator
        self.time_s = 0.0
        self.previous_action: ControlAction | None = None
        self.records: list[DigitalTwinRecord] = []

    def advance(
        self,
        controls: Mapping[str, float],
        dt_s: float,
        *,
        sensor_packet: SensorPacket | None = None,
        future_controls: Mapping[str, np.ndarray] | None = None,
        future_time_s: np.ndarray | None = None,
    ) -> DigitalTwinRecord:
        self.filter.predict(controls, dt_s)
        self.time_s += dt_s
        measurements: Mapping[str, float] = {}
        if sensor_packet is not None:
            self.filter.update(sensor_packet)
            measurements = dict(sensor_packet.values)
        health = self.health_estimator.estimate(self.time_s, self.filter.ensemble)
        control = None
        if self.controller is not None:
            if self.control_predictor is None:
                raise RuntimeError("closed-loop control requires control_predictor")
            control = self.controller.decide(
                self.filter.mean,
                self.control_predictor,
                previous_action=self.previous_action,
            )
            self.previous_action = control.action
        risk = None
        if self.risk_model is not None and future_controls is not None:
            if self.risk_simulator is None:
                raise RuntimeError("risk forecast requires risk_simulator")
            risk = self.risk_model.forecast(
                self.filter.ensemble,
                future_controls,
                self.risk_simulator,
                time_s=future_time_s,
            )
        record = DigitalTwinRecord(
            time_s=self.time_s,
            state_mean=self.filter.mean.copy(),
            state_covariance=self.filter.covariance.copy(),
            health=health,
            control=control,
            risk=risk,
            measurements=measurements,
        )
        self.records.append(record)
        return record


__all__ = ["BatteryDigitalTwin", "DigitalTwinRecord"]
