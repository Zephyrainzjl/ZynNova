from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def harmonic(
    grid: ArrayLike,
    *,
    force_constant: float = 1.0,
    center: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Return ``0.5 * force_constant * (x - center)**2 + offset``."""
    x = np.asarray(grid, dtype=np.float64)
    return 0.5 * float(force_constant) * (x - float(center)) ** 2 + float(offset)


def morse(
    grid: ArrayLike,
    *,
    depth: float,
    width: float,
    equilibrium: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Return a Morse potential in the units of the supplied parameters."""
    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be positive")
    x = np.asarray(grid, dtype=np.float64)
    displacement = x - float(equilibrium)
    return float(depth) * (1.0 - np.exp(-float(width) * displacement)) ** 2 + offset


def symmetric_double_well(
    grid: ArrayLike,
    *,
    coefficient: float = 1.0,
    minima: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Return ``coefficient * (x**2 - minima**2)**2 + offset``."""
    if coefficient <= 0 or minima <= 0:
        raise ValueError("coefficient and minima must be positive")
    x = np.asarray(grid, dtype=np.float64)
    return float(coefficient) * (x**2 - float(minima) ** 2) ** 2 + float(offset)


def finite_square_well(
    grid: ArrayLike,
    *,
    half_width: float,
    outside_energy: float,
    inside_energy: float = 0.0,
    center: float = 0.0,
) -> np.ndarray:
    """Return a finite square well."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    x = np.asarray(grid, dtype=np.float64)
    inside = np.abs(x - float(center)) <= float(half_width)
    return np.where(inside, float(inside_energy), float(outside_energy))


def gaussian_barrier(
    grid: ArrayLike,
    *,
    height: float,
    width: float,
    center: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Return a Gaussian barrier with standard deviation ``width``."""
    if width <= 0:
        raise ValueError("width must be positive")
    x = np.asarray(grid, dtype=np.float64)
    exponent = -0.5 * ((x - float(center)) / float(width)) ** 2
    return float(offset) + float(height) * np.exp(exponent)
