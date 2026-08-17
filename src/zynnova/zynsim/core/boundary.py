"""Boundary-condition value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..exceptions import ConfigurationError


@dataclass(slots=True)
class DirichletBC:
    """Values prescribed at assembled degrees of freedom."""

    dofs: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        self.dofs = np.ascontiguousarray(self.dofs, dtype=np.int64).reshape(-1)
        raw_values = np.asarray(self.values, dtype=np.float64)
        if raw_values.ndim == 0:
            raw_values = np.full(len(self.dofs), float(raw_values), dtype=np.float64)
        self.values = np.ascontiguousarray(raw_values, dtype=np.float64).reshape(-1)
        if self.values.shape != self.dofs.shape:
            raise ConfigurationError("Dirichlet dofs and values must have equal length")
        if len(np.unique(self.dofs)) != len(self.dofs):
            raise ConfigurationError("Dirichlet dofs cannot contain duplicates")
        if np.any(self.dofs < 0) or not np.all(np.isfinite(self.values)):
            raise ConfigurationError("Dirichlet boundary data are invalid")

    @classmethod
    def scalar(cls, nodes: Iterable[int], value: float | np.ndarray) -> DirichletBC:
        return cls(np.asarray(tuple(nodes), dtype=np.int64), np.asarray(value, dtype=float))

    @classmethod
    def vector(
        cls,
        nodes: Iterable[int],
        *,
        components: Iterable[int] = (0, 1, 2),
        value: float | Iterable[float] = 0.0,
        dimension: int = 3,
    ) -> DirichletBC:
        node_array = np.asarray(tuple(nodes), dtype=np.int64)
        component_array = np.asarray(tuple(components), dtype=np.int64)
        if np.any(component_array < 0) or np.any(component_array >= dimension):
            raise ConfigurationError("vector Dirichlet component is out of range")
        dofs = (node_array[:, None] * dimension + component_array[None, :]).reshape(-1)
        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            values = np.full(len(dofs), float(raw))
        elif raw.shape == (len(component_array),):
            values = np.tile(raw, len(node_array))
        elif raw.shape == (len(node_array), len(component_array)):
            values = raw.reshape(-1)
        else:
            raise ConfigurationError(
                "vector boundary value must be scalar, per-component, or per-node/component"
            )
        return cls(dofs, values)


@dataclass(slots=True)
class SurfaceLoad:
    """Constant scalar flux or vector traction on triangular faces."""

    faces: np.ndarray
    value: np.ndarray

    def __post_init__(self) -> None:
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ConfigurationError("surface-load faces must have shape (n_faces, 3)")
        self.value = np.asarray(self.value, dtype=np.float64)
        if self.value.ndim > 1 or self.value.size not in {1, 3}:
            raise ConfigurationError("surface-load value must be a scalar or 3-vector")
        if not np.all(np.isfinite(self.value)):
            raise ConfigurationError("surface-load value must be finite")


__all__ = ["DirichletBC", "SurfaceLoad"]
