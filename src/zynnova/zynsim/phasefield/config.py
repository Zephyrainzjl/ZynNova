"""Configuration objects for one-, two-, and three-dimensional phase-field simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Sequence


class BoundaryCondition(str, Enum):
    """Boundary condition applied independently to each spatial axis."""

    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


class SolverBackend(str, Enum):
    """Available numerical execution backends."""

    AUTO = "auto"
    NUMPY_SPECTRAL = "numpy-spectral"
    NUMPY_FD = "numpy-fd"
    CPP_FD = "cpp-fd"
    TORCH_SPECTRAL = "torch-spectral"
    JAX_SPECTRAL = "jax-spectral"


class TimeScheme(str, Enum):
    """Time integration schemes exposed by the phase-field solvers."""

    ETDRK4 = "etdrk4"
    IMEX_BDF2 = "imex-bdf2"
    SEMI_IMPLICIT_EULER = "semi-implicit-euler"
    SSPRK3 = "ssprk3"
    RK4 = "rk4"


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Uniform Cartesian grid supporting 1D, 2D, and 3D domains."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...] | float = 1.0
    boundary: tuple[BoundaryCondition, ...] | BoundaryCondition = BoundaryCondition.PERIODIC
    origin: tuple[float, ...] | float = 0.0

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if not 1 <= len(shape) <= 3:
            raise ValueError("phase-field grids must be one-, two-, or three-dimensional")
        if any(value < 3 for value in shape):
            raise ValueError("every grid axis must contain at least three points")

        spacing = self._expand_numeric(self.spacing, len(shape), "spacing")
        origin = self._expand_numeric(self.origin, len(shape), "origin")
        if any(value <= 0.0 for value in spacing):
            raise ValueError("grid spacing must be strictly positive")

        if isinstance(self.boundary, BoundaryCondition | str):
            boundary = (BoundaryCondition(self.boundary),) * len(shape)
        else:
            boundary = tuple(BoundaryCondition(value) for value in self.boundary)
        if len(boundary) != len(shape):
            raise ValueError("boundary must provide one entry per spatial axis")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "boundary", boundary)

    @staticmethod
    def _expand_numeric(
        value: Sequence[float] | float,
        dimensions: int,
        name: str,
    ) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            return (float(value),) * dimensions
        expanded = tuple(float(item) for item in value)
        if len(expanded) != dimensions:
            raise ValueError(f"{name} must provide one entry per spatial axis")
        return expanded

    @property
    def dimensions(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(n * dx for n, dx in zip(self.shape, self.spacing, strict=True))

    @property
    def periodic(self) -> bool:
        return all(item == BoundaryCondition.PERIODIC for item in self.boundary)

    def coordinates(self):
        import numpy as np

        axes = [
            x0 + dx * np.arange(n, dtype=float)
            for n, dx, x0 in zip(self.shape, self.spacing, self.origin, strict=True)
        ]
        if self.dimensions == 1:
            return axes[0]
        return np.meshgrid(*axes, indexing="ij")


@dataclass(slots=True)
class AdaptiveTimeConfig:
    """Adaptive step-size and free-energy acceptance settings."""

    enabled: bool = True
    minimum_dt: float = 1.0e-10
    maximum_dt: float | None = None
    relative_tolerance: float = 1.0e-5
    absolute_tolerance: float = 1.0e-8
    safety: float = 0.85
    growth_limit: float = 1.5
    shrink_limit: float = 0.2
    enforce_energy_decay: bool = True
    energy_tolerance: float = 1.0e-9
    maximum_rejections: int = 12


@dataclass(slots=True)
class SolverConfig:
    """Common configuration for all phase-field solver implementations."""

    dt: float = 1.0e-3
    scheme: TimeScheme = TimeScheme.ETDRK4
    backend: SolverBackend = SolverBackend.AUTO
    steps: int | None = None
    final_time: float | None = None
    save_every: int = 1
    diagnostics_every: int = 1
    derivative_order: int = 4
    dtype: str = "float64"
    device: str = "auto"
    adaptive: AdaptiveTimeConfig = field(default_factory=AdaptiveTimeConfig)
    maximum_steps: int = 1_000_000
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.scheme = TimeScheme(self.scheme)
        self.backend = SolverBackend(self.backend)
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive")
        if self.steps is None and self.final_time is None:
            raise ValueError("either steps or final_time must be provided")
        if self.steps is not None and self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.final_time is not None and self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        if self.save_every <= 0 or self.diagnostics_every <= 0:
            raise ValueError("save_every and diagnostics_every must be positive")
        if self.derivative_order not in {2, 4}:
            raise ValueError("derivative_order must be 2 or 4")
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")


__all__ = [
    "AdaptiveTimeConfig",
    "BoundaryCondition",
    "GridSpec",
    "SolverBackend",
    "SolverConfig",
    "TimeScheme",
]
