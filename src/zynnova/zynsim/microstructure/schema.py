"""Phase schema and manufacturing controls for explicit battery microstructures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import numpy as np


class BatteryPhase(IntEnum):
    SEPARATOR_ELECTROLYTE = 0
    POSITIVE_ACTIVE = 1
    POSITIVE_ELECTROLYTE = 2
    NEGATIVE_ACTIVE = 3
    NEGATIVE_ELECTROLYTE = 4
    POSITIVE_CBD = 5
    NEGATIVE_CBD = 6
    POSITIVE_CEI = 7
    NEGATIVE_SEI = 8
    CRACK = 9
    NEGATIVE_CURRENT_COLLECTOR = 10
    POSITIVE_CURRENT_COLLECTOR = 11


DEFAULT_PHASE_NAMES: Mapping[int, str] = {
    int(phase): phase.name.lower() for phase in BatteryPhase
}


@dataclass(slots=True)
class ManufacturingProcessControl:
    """Controllable process variables used during generation and meshing."""

    positive_porosity: float = 0.30
    negative_porosity: float = 0.32
    positive_cbd_fraction: float = 0.06
    negative_cbd_fraction: float = 0.05
    calendering_ratio: float = 0.0
    particle_coalescence: float = 0.0
    sei_thickness_voxels: int = 0
    cei_thickness_voxels: int = 0
    crack_volume_fraction: float = 0.0
    crack_anisotropy: float = 0.5
    smoothing: float = 0.0
    refinement: int | tuple[int, int, int] = 1
    random_seed: int = 42

    def __post_init__(self) -> None:
        fractions = (
            self.positive_porosity,
            self.negative_porosity,
            self.positive_cbd_fraction,
            self.negative_cbd_fraction,
            self.calendering_ratio,
            self.particle_coalescence,
            self.crack_volume_fraction,
            self.crack_anisotropy,
            self.smoothing,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in fractions):
            raise ValueError("manufacturing fractions must lie in [0,1]")
        if self.positive_porosity + self.positive_cbd_fraction >= 1.0:
            raise ValueError("positive porosity + CBD fraction must be below one")
        if self.negative_porosity + self.negative_cbd_fraction >= 1.0:
            raise ValueError("negative porosity + CBD fraction must be below one")
        if self.sei_thickness_voxels < 0 or self.cei_thickness_voxels < 0:
            raise ValueError("interphase thickness cannot be negative")


def validate_phase_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("phase labels must be a non-empty three-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.equal(values, np.round(values))):
            raise ValueError("phase labels must be integers")
    values = np.ascontiguousarray(values, dtype=np.int32)
    valid = {int(phase) for phase in BatteryPhase}
    unknown = sorted(set(np.unique(values).tolist()) - valid)
    if unknown:
        raise ValueError(f"unknown battery phase labels: {unknown}")
    return values


__all__ = [
    "BatteryPhase",
    "DEFAULT_PHASE_NAMES",
    "ManufacturingProcessControl",
    "validate_phase_labels",
]
