"""Aligned multimodal observations for inverse battery modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np


class ObservationModality(str, Enum):
    VOLTAGE = "voltage"
    EIS_REAL = "eis_real"
    EIS_IMAG = "eis_imag"
    TEMPERATURE = "temperature"
    EXPANSION = "expansion"
    IMAGE_FEATURES = "image_features"


@dataclass(frozen=True, slots=True)
class ObservationSeries:
    modality: ObservationModality
    coordinates: np.ndarray
    values: np.ndarray
    standard_deviation: np.ndarray | float
    mask: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        values = np.asarray(self.values, dtype=float)
        sigma = np.broadcast_to(np.asarray(self.standard_deviation, dtype=float), values.shape).copy()
        if coordinates.shape[0] != values.shape[0] or values.shape != sigma.shape:
            raise ValueError("observation coordinates, values, and uncertainty are inconsistent")
        if np.any(sigma <= 0.0) or not np.isfinite(values).all() or not np.isfinite(sigma).all():
            raise ValueError("observation values must be finite and uncertainty positive")
        mask = np.ones(values.shape, dtype=bool) if self.mask is None else np.asarray(self.mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("observation mask shape is inconsistent")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "standard_deviation", sigma)
        object.__setattr__(self, "mask", mask)


@dataclass(slots=True)
class MultimodalObservationSet:
    series: list[ObservationSeries]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("at least one observation series is required")
        modalities = [series.modality for series in self.series]
        if len(set(modalities)) != len(modalities):
            raise ValueError("observation modalities must be unique")

    def by_modality(self) -> dict[ObservationModality, ObservationSeries]:
        return {series.modality: series for series in self.series}


__all__ = [
    "MultimodalObservationSet",
    "ObservationModality",
    "ObservationSeries",
]
