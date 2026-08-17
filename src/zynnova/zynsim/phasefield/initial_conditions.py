"""Initial-condition generators for common phase-field benchmarks and experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import GridSpec


def random_noise(
    grid: GridSpec,
    *,
    mean: float = 0.0,
    amplitude: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean + amplitude * rng.standard_normal(grid.shape)


def planar_interface(
    grid: GridSpec,
    *,
    axis: int = 0,
    position: float | None = None,
    width: float = 1.0,
    lower: float = -1.0,
    upper: float = 1.0,
) -> np.ndarray:
    if not 0 <= axis < grid.dimensions:
        raise ValueError("axis is outside the grid dimensions")
    coordinate = grid.coordinates()
    coordinate_axis = coordinate if grid.dimensions == 1 else coordinate[axis]
    if position is None:
        position = grid.origin[axis] + 0.5 * grid.lengths[axis]
    profile = 0.5 * (1.0 + np.tanh((coordinate_axis - position) / max(width, 1.0e-15)))
    return lower + (upper - lower) * profile


def spherical_nucleus(
    grid: GridSpec,
    *,
    center: Sequence[float] | None = None,
    radius: float | None = None,
    width: float = 1.0,
    inside: float = 1.0,
    outside: float = -1.0,
) -> np.ndarray:
    if center is None:
        center = tuple(
            x0 + 0.5 * length for x0, length in zip(grid.origin, grid.lengths, strict=True)
        )
    if len(center) != grid.dimensions:
        raise ValueError("center must contain one coordinate per dimension")
    if radius is None:
        radius = 0.15 * min(grid.lengths)
    coordinates = grid.coordinates()
    components = (coordinates,) if grid.dimensions == 1 else coordinates
    distance_squared = np.zeros(grid.shape, dtype=float)
    for values, center_value in zip(components, center, strict=True):
        distance_squared += (values - center_value) ** 2
    signed_distance = np.sqrt(distance_squared) - radius
    indicator = 0.5 * (1.0 - np.tanh(signed_distance / max(width, 1.0e-15)))
    return outside + (inside - outside) * indicator


def multiple_nuclei(
    grid: GridSpec,
    *,
    count: int = 8,
    radius: float | None = None,
    width: float = 1.0,
    seed: int = 0,
    inside: float = 1.0,
    outside: float = -1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if radius is None:
        radius = 0.05 * min(grid.lengths)
    result = np.full(grid.shape, outside, dtype=float)
    for _ in range(count):
        center = tuple(
            x0 + rng.random() * length
            for x0, length in zip(grid.origin, grid.lengths, strict=True)
        )
        nucleus = spherical_nucleus(
            grid,
            center=center,
            radius=radius,
            width=width,
            inside=inside,
            outside=outside,
        )
        result = np.maximum(result, nucleus)
    return result


def voronoi_order_parameters(
    grid: GridSpec,
    grains: int,
    *,
    seed: int = 0,
    diffuse_width: float = 0.0,
) -> dict[str, np.ndarray]:
    if grains < 2:
        raise ValueError("at least two grains are required")
    rng = np.random.default_rng(seed)
    centers = np.column_stack(
        [
            x0 + rng.random(grains) * length
            for x0, length in zip(grid.origin, grid.lengths, strict=True)
        ]
    )
    coordinates = grid.coordinates()
    components = (coordinates,) if grid.dimensions == 1 else coordinates
    distance = np.zeros((grains, *grid.shape), dtype=float)
    for grain in range(grains):
        for axis, values in enumerate(components):
            distance[grain] += (values - centers[grain, axis]) ** 2
    labels = np.argmin(distance, axis=0)
    fields = {f"eta{grain}": (labels == grain).astype(float) for grain in range(grains)}
    if diffuse_width > 0.0:
        try:
            from scipy.ndimage import gaussian_filter

            sigma = tuple(diffuse_width / dx for dx in grid.spacing)
            fields = {name: gaussian_filter(values, sigma=sigma, mode="wrap") for name, values in fields.items()}
            total = sum(fields.values())
            fields = {name: values / np.maximum(total, 1.0e-12) for name, values in fields.items()}
        except ImportError:
            pass
    return fields


def from_labels(
    labels: Any,
    *,
    phase_count: int | None = None,
    prefix: str = "eta",
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    if phase_count is None:
        phase_count = int(labels.max()) + 1
    return {f"{prefix}{index}": (labels == index).astype(float) for index in range(phase_count)}


__all__ = [
    "from_labels",
    "multiple_nuclei",
    "planar_interface",
    "random_noise",
    "spherical_nucleus",
    "voronoi_order_parameters",
]
