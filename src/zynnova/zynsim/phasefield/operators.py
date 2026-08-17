"""Spatial differential operators shared by phase-field models and solvers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ._backend import native_module
from .config import BoundaryCondition, GridSpec


def _boundary_name(boundary: BoundaryCondition) -> str:
    return str(BoundaryCondition(boundary).value)


def _shift_numpy(
    values: np.ndarray,
    shift: int,
    axis: int,
    boundary: BoundaryCondition,
    dirichlet_value: float = 0.0,
) -> np.ndarray:
    if boundary == BoundaryCondition.PERIODIC:
        return np.roll(values, shift, axis=axis)

    result = np.roll(values, shift, axis=axis)
    index = [slice(None)] * values.ndim
    if shift > 0:
        index[axis] = slice(0, shift)
        source = [slice(None)] * values.ndim
        source[axis] = shift if boundary == BoundaryCondition.NEUMANN else slice(0, shift)
    else:
        index[axis] = slice(shift, None)
        source = [slice(None)] * values.ndim
        source[axis] = shift - 1 if boundary == BoundaryCondition.NEUMANN else slice(shift, None)

    if boundary == BoundaryCondition.NEUMANN:
        result[tuple(index)] = np.take(values, indices=source[axis], axis=axis)
    else:
        result[tuple(index)] = dirichlet_value
    return result


def finite_difference_laplacian(
    values: np.ndarray,
    grid: GridSpec,
    *,
    order: int = 4,
    backend: str = "auto",
) -> np.ndarray:
    """Return a second- or fourth-order Cartesian Laplacian."""

    array = np.asarray(values, dtype=float)
    native = native_module() if backend in {"auto", "cpp"} else None
    if native is not None:
        return np.asarray(
            native.laplacian(
                array,
                tuple(float(value) for value in grid.spacing),
                tuple(_boundary_name(item) for item in grid.boundary),
                int(order),
            )
        )
    if backend == "cpp":
        raise RuntimeError("the native phase-field backend is not available")

    result = np.zeros_like(array)
    for axis, (dx, boundary) in enumerate(zip(grid.spacing, grid.boundary, strict=True)):
        if order == 2:
            result += (
                _shift_numpy(array, 1, axis, boundary)
                - 2.0 * array
                + _shift_numpy(array, -1, axis, boundary)
            ) / dx**2
        elif order == 4:
            result += (
                -_shift_numpy(array, 2, axis, boundary)
                + 16.0 * _shift_numpy(array, 1, axis, boundary)
                - 30.0 * array
                + 16.0 * _shift_numpy(array, -1, axis, boundary)
                - _shift_numpy(array, -2, axis, boundary)
            ) / (12.0 * dx**2)
        else:
            raise ValueError("order must be 2 or 4")
    return result


def finite_difference_gradient(
    values: np.ndarray,
    grid: GridSpec,
    *,
    order: int = 4,
) -> tuple[np.ndarray, ...]:
    array = np.asarray(values, dtype=float)
    gradient: list[np.ndarray] = []
    for axis, (dx, boundary) in enumerate(zip(grid.spacing, grid.boundary, strict=True)):
        if order == 2:
            derivative = (
                _shift_numpy(array, -1, axis, boundary)
                - _shift_numpy(array, 1, axis, boundary)
            ) / (2.0 * dx)
        elif order == 4:
            derivative = (
                _shift_numpy(array, 2, axis, boundary)
                - 8.0 * _shift_numpy(array, 1, axis, boundary)
                + 8.0 * _shift_numpy(array, -1, axis, boundary)
                - _shift_numpy(array, -2, axis, boundary)
            ) / (12.0 * dx)
        else:
            raise ValueError("order must be 2 or 4")
        gradient.append(derivative)
    return tuple(gradient)


@lru_cache(maxsize=64)
def numpy_wave_numbers(grid: GridSpec) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return squared wave number and per-axis wave-number arrays."""

    axes = [
        2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        for n, dx in zip(grid.shape, grid.spacing, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    k2 = np.zeros(grid.shape, dtype=float)
    for component in mesh:
        k2 += component**2
    return k2, tuple(mesh)


@dataclass(slots=True)
class NumpyDifferentialOperators:
    """Differential operators with spectral or finite-difference discretization."""

    grid: GridSpec
    method: str = "spectral"
    order: int = 4
    finite_difference_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.method not in {"spectral", "finite-difference"}:
            raise ValueError("method must be 'spectral' or 'finite-difference'")
        if self.method == "spectral" and not self.grid.periodic:
            raise ValueError("spectral operators require periodic boundary conditions")

    def laplacian(self, values: np.ndarray) -> np.ndarray:
        if self.method == "spectral":
            k2, _ = numpy_wave_numbers(self.grid)
            return np.fft.ifftn(-k2 * np.fft.fftn(values)).real
        return finite_difference_laplacian(
            values,
            self.grid,
            order=self.order,
            backend=self.finite_difference_backend,
        )

    def biharmonic(self, values: np.ndarray) -> np.ndarray:
        if self.method == "spectral":
            k2, _ = numpy_wave_numbers(self.grid)
            return np.fft.ifftn(k2**2 * np.fft.fftn(values)).real
        return self.laplacian(self.laplacian(values))

    def gradient(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.method == "spectral":
            _, wave_numbers = numpy_wave_numbers(self.grid)
            transformed = np.fft.fftn(values)
            return tuple(
                np.fft.ifftn(1j * component * transformed).real
                for component in wave_numbers
            )
        return finite_difference_gradient(values, self.grid, order=self.order)

    def divergence(self, vector: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        if len(vector) != self.grid.dimensions:
            raise ValueError("vector must have one component per spatial dimension")
        result = np.zeros_like(np.asarray(vector[0], dtype=float))
        if self.method == "spectral":
            _, wave_numbers = numpy_wave_numbers(self.grid)
            for values, component in zip(vector, wave_numbers, strict=True):
                result += np.fft.ifftn(1j * component * np.fft.fftn(values)).real
            return result
        for axis, (values, dx, boundary) in enumerate(
            zip(vector, self.grid.spacing, self.grid.boundary, strict=True)
        ):
            result += (
                _shift_numpy(np.asarray(values), -1, axis, boundary)
                - _shift_numpy(np.asarray(values), 1, axis, boundary)
            ) / (2.0 * dx)
        return result

    def grad_squared(self, values: np.ndarray) -> np.ndarray:
        components = self.gradient(values)
        total = np.zeros_like(values, dtype=float)
        for component in components:
            total += component**2
        return total

    def integrate(self, values: np.ndarray) -> float:
        return float(np.sum(values) * np.prod(self.grid.spacing))


class ArrayDifferentialOperators:
    """Array-API differential operators for Torch and JAX periodic simulations."""

    def __init__(self, grid: GridSpec, xp: Any):
        if not grid.periodic:
            raise ValueError("array spectral operators require periodic boundaries")
        self.grid = grid
        self.xp = xp
        axes = [
            2.0 * np.pi * np.fft.fftfreq(n, d=dx)
            for n, dx in zip(grid.shape, grid.spacing, strict=True)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.k = tuple(self._as_array(component) for component in mesh)
        self.k2 = self._as_array(sum(component**2 for component in mesh))

    def _as_array(self, values: np.ndarray):
        if getattr(self.xp, "__name__", "").startswith("torch"):
            import torch

            return torch.as_tensor(values)
        return self.xp.asarray(values)

    def _fft(self, values):
        return self.xp.fft.fftn(values, dim=tuple(range(-self.grid.dimensions, 0))) if getattr(
            self.xp, "__name__", ""
        ).startswith("torch") else self.xp.fft.fftn(values)

    def _ifft(self, values):
        transformed = self.xp.fft.ifftn(
            values,
            dim=tuple(range(-self.grid.dimensions, 0)),
        ) if getattr(self.xp, "__name__", "").startswith("torch") else self.xp.fft.ifftn(values)
        return transformed.real

    def laplacian(self, values):
        k2 = self.k2.to(device=values.device, dtype=values.real.dtype) if hasattr(
            self.k2, "to"
        ) else self.k2
        return self._ifft(-k2 * self._fft(values))

    def biharmonic(self, values):
        k2 = self.k2.to(device=values.device, dtype=values.real.dtype) if hasattr(
            self.k2, "to"
        ) else self.k2
        return self._ifft(k2**2 * self._fft(values))

    def gradient(self, values):
        transformed = self._fft(values)
        output = []
        for component in self.k:
            if hasattr(component, "to"):
                component = component.to(device=values.device, dtype=values.real.dtype)
            output.append(self._ifft(1j * component * transformed))
        return tuple(output)

    def divergence(self, vector):
        output = 0.0
        for values, component in zip(vector, self.k, strict=True):
            if hasattr(component, "to"):
                component = component.to(device=values.device, dtype=values.real.dtype)
            output = output + self._ifft(1j * component * self._fft(values))
        return output

    def grad_squared(self, values):
        output = 0.0
        for component in self.gradient(values):
            output = output + component**2
        return output


__all__ = [
    "ArrayDifferentialOperators",
    "NumpyDifferentialOperators",
    "finite_difference_gradient",
    "finite_difference_laplacian",
    "numpy_wave_numbers",
]
