"""Microstructure and numerical diagnostics for phase-field trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GridSpec
from .operators import NumpyDifferentialOperators


@dataclass(frozen=True, slots=True)
class StructureFactorResult:
    wave_number: np.ndarray
    intensity: np.ndarray
    characteristic_length: float


def structure_factor(field: Any, grid: GridSpec, *, radial_bins: int = 128) -> StructureFactorResult:
    values = np.asarray(field, dtype=float)
    centered = values - values.mean()
    spectrum = np.abs(np.fft.fftn(centered)) ** 2
    axes = [2.0 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(grid.shape, grid.spacing, strict=True)]
    mesh = np.meshgrid(*axes, indexing="ij")
    magnitude = np.sqrt(sum(component**2 for component in mesh))
    maximum = float(magnitude.max())
    edges = np.linspace(0.0, maximum, radial_bins + 1)
    indices = np.digitize(magnitude.ravel(), edges) - 1
    radial = np.zeros(radial_bins, dtype=float)
    counts = np.zeros(radial_bins, dtype=float)
    np.add.at(radial, np.clip(indices, 0, radial_bins - 1), spectrum.ravel())
    np.add.at(counts, np.clip(indices, 0, radial_bins - 1), 1.0)
    radial /= np.maximum(counts, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if radial_bins > 1 and np.any(radial[1:] > 0.0):
        peak = 1 + int(np.argmax(radial[1:]))
        characteristic = 2.0 * np.pi / max(centers[peak], 1.0e-15)
    else:
        characteristic = float("inf")
    return StructureFactorResult(centers, radial, characteristic)


def interfacial_measure(field: Any, grid: GridSpec) -> float:
    operators = NumpyDifferentialOperators(
        grid,
        method="spectral" if grid.periodic else "finite-difference",
    )
    gradient_magnitude = np.sqrt(operators.grad_squared(np.asarray(field, dtype=float)))
    return operators.integrate(gradient_magnitude)


def phase_fraction(field: Any, *, threshold: float = 0.0) -> float:
    return float(np.mean(np.asarray(field) >= threshold))


def mass(field: Any, grid: GridSpec) -> float:
    return float(np.sum(np.asarray(field)) * np.prod(grid.spacing))


def energy_history(result) -> np.ndarray:
    return np.asarray([record.free_energy for record in result.trajectory.diagnostics], dtype=float)


__all__ = [
    "StructureFactorResult",
    "energy_history",
    "interfacial_measure",
    "mass",
    "phase_fraction",
    "structure_factor",
]
