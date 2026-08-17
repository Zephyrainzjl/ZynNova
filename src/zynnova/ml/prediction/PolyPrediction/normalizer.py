from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import PropertySpec

_EPSILON = 1.0e-8


def transform_value(value: float, spec: PropertySpec) -> float:
    number = float(value)
    if spec.transform == "identity":
        return number
    if spec.transform == "log":
        return math.log(max(number, _EPSILON))
    if spec.transform == "log1p":
        return math.log1p(max(number, 0.0))
    if spec.transform == "log10":
        return math.log10(max(number, _EPSILON))
    if spec.transform == "logit":
        lower = float(spec.lower_bound)
        upper = float(spec.upper_bound)
        fraction = (number - lower) / (upper - lower)
        fraction = min(max(fraction, _EPSILON), 1.0 - _EPSILON)
        return math.log(fraction / (1.0 - fraction))
    raise ValueError(f"unknown target transform: {spec.transform}")


def inverse_value(value: float, spec: PropertySpec) -> float:
    if spec.transform == "identity":
        result = float(value)
    elif spec.transform == "log":
        result = math.exp(float(value))
    elif spec.transform == "log1p":
        result = math.expm1(float(value))
    elif spec.transform == "log10":
        result = 10.0 ** float(value)
    elif spec.transform == "logit":
        lower = float(spec.lower_bound)
        upper = float(spec.upper_bound)
        number = float(value)
        if number >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-number))
        else:
            exponential = math.exp(number)
            sigmoid = exponential / (1.0 + exponential)
        result = lower + (upper - lower) * sigmoid
    else:
        raise ValueError(f"unknown target transform: {spec.transform}")
    if spec.lower_bound is not None:
        result = max(result, spec.lower_bound)
    if spec.upper_bound is not None:
        result = min(result, spec.upper_bound)
    return result


class MaskedTargetNormalizer:
    def __init__(
        self,
        specs: Sequence[PropertySpec],
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self.specs = tuple(specs)
        self.mean = np.asarray(
            mean if mean is not None else np.zeros(len(self.specs)),
            dtype=np.float64,
        )
        self.std = np.asarray(
            std if std is not None else np.ones(len(self.specs)),
            dtype=np.float64,
        )
        if self.mean.shape != (len(self.specs),) or self.std.shape != (len(self.specs),):
            raise ValueError("normalizer statistics do not match property specifications")
        self.std = np.maximum(self.std, _EPSILON)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def fit(self, values: np.ndarray, mask: np.ndarray) -> MaskedTargetNormalizer:
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.shape != mask.shape or values.shape[1:] != (len(self.specs),):
            raise ValueError("target values and mask have incompatible shapes")
        transformed = np.zeros_like(values)
        for column, spec in enumerate(self.specs):
            observed = mask[:, column]
            if observed.any():
                transformed[observed, column] = [
                    transform_value(value, spec) for value in values[observed, column]
                ]
                self.mean[column] = float(transformed[observed, column].mean())
                self.std[column] = max(
                    float(transformed[observed, column].std()),
                    _EPSILON,
                )
            else:
                self.mean[column] = 0.0
                self.std[column] = 1.0
        return self

    def encode_row(self, values: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.zeros(len(self.specs), dtype=np.float32)
        mask = np.zeros(len(self.specs), dtype=bool)
        for index, spec in enumerate(self.specs):
            value = values.get(spec.name)
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            transformed = transform_value(number, spec)
            encoded[index] = (transformed - self.mean[index]) / self.std[index]
            mask[index] = True
        return encoded, mask

    def decode_row(
        self,
        normalized: Sequence[float],
    ) -> dict[str, float]:
        values = np.asarray(normalized, dtype=np.float64)
        return {
            spec.name: inverse_value(
                values[index] * self.std[index] + self.mean[index],
                spec,
            )
            for index, spec in enumerate(self.specs)
        }

    def decode_tensor(self, normalized):
        import torch

        mean = torch.as_tensor(self.mean, device=normalized.device, dtype=normalized.dtype)
        std = torch.as_tensor(self.std, device=normalized.device, dtype=normalized.dtype)
        transformed = normalized * std + mean
        outputs = []
        for index, spec in enumerate(self.specs):
            value = transformed[..., index]
            if spec.transform == "identity":
                decoded = value
            elif spec.transform == "log":
                decoded = value.exp()
            elif spec.transform == "log1p":
                decoded = value.exp() - 1.0
            elif spec.transform == "log10":
                decoded = torch.pow(torch.as_tensor(10.0, device=value.device), value)
            elif spec.transform == "logit":
                lower = float(spec.lower_bound)
                upper = float(spec.upper_bound)
                decoded = lower + (upper - lower) * value.sigmoid()
            else:
                raise ValueError(f"unknown target transform: {spec.transform}")
            outputs.append(decoded)
        return torch.stack(outputs, dim=-1)

    def physical_std(
        self,
        normalized_mean: Sequence[float],
        normalized_std: Sequence[float],
    ) -> dict[str, float]:
        means = np.asarray(normalized_mean, dtype=np.float64)
        stds = np.asarray(normalized_std, dtype=np.float64)
        result: dict[str, float] = {}
        for index, spec in enumerate(self.specs):
            center = means[index]
            width = max(stds[index], 0.0)
            low = inverse_value(
                (center - width) * self.std[index] + self.mean[index],
                spec,
            )
            high = inverse_value(
                (center + width) * self.std[index] + self.mean[index],
                spec,
            )
            result[spec.name] = abs(high - low) / 2.0
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "specs": [asdict(spec) for spec in self.specs],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> MaskedTargetNormalizer:
        specs = tuple(PropertySpec(**payload) for payload in state["specs"])
        return cls(specs, mean=state["mean"], std=state["std"])


class MaskedFeatureNormalizer:
    def __init__(
        self,
        names: Sequence[str],
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self.names = tuple(str(name) for name in names)
        self.mean = np.asarray(
            mean if mean is not None else np.zeros(len(self.names)),
            dtype=np.float64,
        )
        self.std = np.asarray(
            std if std is not None else np.ones(len(self.names)),
            dtype=np.float64,
        )
        self.std = np.maximum(self.std, _EPSILON)

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> MaskedFeatureNormalizer:
        for index, name in enumerate(self.names):
            values: list[float] = []
            for row in rows:
                value = row.get(name)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values.append(number)
            if values:
                array = np.asarray(values, dtype=np.float64)
                self.mean[index] = float(array.mean())
                self.std[index] = max(float(array.std()), _EPSILON)
            else:
                self.mean[index] = 0.0
                self.std[index] = 1.0
        return self

    def encode_row(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.zeros(len(self.names), dtype=np.float32)
        mask = np.zeros(len(self.names), dtype=bool)
        for index, name in enumerate(self.names):
            value = row.get(name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                encoded[index] = (number - self.mean[index]) / self.std[index]
                mask[index] = True
        return encoded, mask

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> MaskedFeatureNormalizer:
        return cls(state["names"], mean=state["mean"], std=state["std"])


__all__ = [
    "MaskedFeatureNormalizer",
    "MaskedTargetNormalizer",
    "inverse_value",
    "transform_value",
]
