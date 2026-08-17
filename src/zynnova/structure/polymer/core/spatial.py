from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .enums import Resolution


@dataclass
class PeriodicBox:
    matrix: np.ndarray
    periodic: tuple[bool, bool, bool] = (True, True, True)
    unit: str = "angstrom"

    def validate(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise ValueError("box matrix must have shape (3, 3)")
        if abs(float(np.linalg.det(self.matrix))) < 1e-12:
            raise ValueError("box matrix must be nonsingular")


@dataclass
class SpatialFrame:
    resolution: Resolution
    node_ids: list[str]
    coordinates: np.ndarray
    box: PeriodicBox | None = None
    velocities: np.ndarray | None = None
    forces: np.ndarray | None = None
    orientations: np.ndarray | None = None
    spatial_edge_index: np.ndarray | None = None
    periodic_edge_shift: np.ndarray | None = None
    phase_labels: np.ndarray | None = None
    time: float | None = None
    units: dict[str, str] = field(default_factory=lambda: {"length": "angstrom"})
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        n = len(self.node_ids)
        if self.coordinates.shape != (n, 3):
            raise ValueError(f"coordinates must have shape ({n}, 3)")
        for name in ("velocities", "forces"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (n, 3):
                    raise ValueError(f"{name} must have shape ({n}, 3)")
                setattr(self, name, value)
        if self.orientations is not None:
            self.orientations = np.asarray(self.orientations, dtype=float)
            if self.orientations.shape not in {(n, 4), (n, 3, 3)}:
                raise ValueError("orientations must be quaternions [N,4] or matrices [N,3,3]")
        if self.box is not None:
            self.box.validate()
        if self.spatial_edge_index is not None:
            self.spatial_edge_index = np.asarray(self.spatial_edge_index, dtype=np.int64)
            if self.spatial_edge_index.ndim != 2 or self.spatial_edge_index.shape[0] != 2:
                raise ValueError("spatial_edge_index must have shape [2, E]")
            e = self.spatial_edge_index.shape[1]
            if self.periodic_edge_shift is not None:
                self.periodic_edge_shift = np.asarray(
                    self.periodic_edge_shift, dtype=np.int64
                )
                if self.periodic_edge_shift.shape != (e, 3):
                    raise ValueError("periodic_edge_shift must have shape [E, 3]")


@dataclass
class SpatialState:
    id: str
    frames: list[SpatialFrame]
    temperature: float | None = None
    pressure: float | None = None
    solvent: dict[str, Any] | None = None
    density: float | None = None
    crystallinity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("spatial state id cannot be empty")
        if not self.frames:
            raise ValueError("spatial state must contain at least one frame")
        for frame in self.frames:
            frame.validate()
        if self.crystallinity is not None and not 0 <= self.crystallinity <= 1:
            raise ValueError("crystallinity must lie in [0, 1]")
