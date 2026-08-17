"""Joint differentiable fitting of voltage, EIS, temperature, swelling, and images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .observations import MultimodalObservationSet, ObservationModality


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("differentiable calibration requires PyTorch") from exc
    return torch


@dataclass(slots=True)
class JointCalibrationConfig:
    epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    gradient_clip_norm: float = 10.0
    modality_weights: Mapping[ObservationModality, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("joint calibration controls are invalid")


@dataclass(slots=True)
class JointCalibrationResult:
    history: list[dict[str, float]]
    best_loss: float
    best_state_dict: Mapping[str, Any]
    predictions: Mapping[str, Any]


class JointInverseProblem:
    def __init__(
        self,
        solver: Any,
        observations: MultimodalObservationSet,
        *,
        mechanism_selector: Any | None = None,
        config: JointCalibrationConfig | None = None,
    ) -> None:
        self.solver = solver
        self.observations = observations
        self.mechanism_selector = mechanism_selector
        self.config = config or JointCalibrationConfig()

    def fit(self, time_s: Any, current_A: Any) -> JointCalibrationResult:
        torch = _torch()
        parameters = list(self.solver.parameters())
        if self.mechanism_selector is not None:
            parameters.extend(self.mechanism_selector.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_loss = float("inf")
        best_state: dict[str, Any] = {}
        history: list[dict[str, float]] = []
        final_predictions: Mapping[str, Any] = {}
        for epoch in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = self.solver(time_s, current_A)
            loss, metrics = self.loss(predictions)
            if self.mechanism_selector is not None:
                penalty = self.mechanism_selector.sparsity_penalty()
                loss = loss + penalty
                metrics["mechanism_sparsity"] = float(penalty.detach().cpu().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm)
            optimizer.step()
            value = float(loss.detach().cpu().item())
            record = {"epoch": float(epoch), "loss": value, **metrics}
            history.append(record)
            if value < best_loss:
                best_loss = value
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.solver.state_dict().items()
                }
            final_predictions = predictions
        return JointCalibrationResult(history, best_loss, best_state, final_predictions)

    def loss(self, predictions: Mapping[str, Any]) -> tuple[Any, dict[str, float]]:
        torch = _torch()
        total = None
        metrics: dict[str, float] = {}
        for series in self.observations.series:
            prediction = self._prediction_for(series.modality, predictions, series.coordinates)
            target = torch.as_tensor(series.values, device=prediction.device, dtype=prediction.dtype)
            sigma = torch.as_tensor(
                series.standard_deviation, device=prediction.device, dtype=prediction.dtype
            )
            mask = torch.as_tensor(series.mask, device=prediction.device, dtype=torch.bool)
            residual = (prediction - target) / sigma
            modality_loss = torch.mean(residual[mask].square())
            weight = float(self.config.modality_weights.get(series.modality, 1.0))
            total = weight * modality_loss if total is None else total + weight * modality_loss
            metrics[series.modality.value] = float(modality_loss.detach().cpu().item())
        if total is None:
            raise RuntimeError("no observation losses were assembled")
        return total, metrics

    def _prediction_for(self, modality: ObservationModality, predictions: Mapping[str, Any], coordinates: Any) -> Any:
        torch = _torch()
        mapping = {
            ObservationModality.VOLTAGE: "voltage_V",
            ObservationModality.TEMPERATURE: "temperature_K",
            ObservationModality.EXPANSION: "expansion",
            ObservationModality.IMAGE_FEATURES: "image_features",
        }
        if modality in {ObservationModality.EIS_REAL, ObservationModality.EIS_IMAG}:
            impedance = self.solver.impedance(torch.as_tensor(coordinates))
            return impedance.real if modality == ObservationModality.EIS_REAL else impedance.imag
        key = mapping[modality]
        predicted = predictions[key]
        if predicted.shape[0] == len(coordinates):
            return predicted
        indices = torch.linspace(0, predicted.shape[0] - 1, len(coordinates), device=predicted.device)
        nearest = torch.round(indices).long()
        return predicted[nearest]


__all__ = [
    "JointCalibrationConfig",
    "JointCalibrationResult",
    "JointInverseProblem",
]
