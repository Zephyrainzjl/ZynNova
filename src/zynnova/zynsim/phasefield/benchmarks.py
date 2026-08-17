"""Reference benchmark builders inspired by PFHub and manufactured-solution tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridSpec, SolverConfig, TimeScheme
from .initial_conditions import random_noise, spherical_nucleus
from .models import AllenCahnModel, CahnHilliardModel, DendriticSolidificationModel


@dataclass(frozen=True, slots=True)
class PhaseFieldBenchmark:
    name: str
    model: object
    grid: GridSpec
    fields: dict[str, np.ndarray]
    solver: SolverConfig
    description: str


def spinodal_decomposition_2d(*, resolution: int = 128, seed: int = 0) -> PhaseFieldBenchmark:
    grid = GridSpec((resolution, resolution), spacing=1.0)
    model = CahnHilliardModel(
        mobility=1.0,
        quadratic=-1.0,
        quartic=1.0,
        gradient_coefficient=1.0,
    )
    return PhaseFieldBenchmark(
        "spinodal-decomposition-2d",
        model,
        grid,
        {"c": random_noise(grid, mean=0.0, amplitude=0.02, seed=seed)},
        SolverConfig(
            dt=0.1,
            final_time=20.0,
            scheme=TimeScheme.ETDRK4,
            save_every=10,
        ),
        "Mass-conserving Cahn-Hilliard spinodal decomposition.",
    )


def shrinking_circle_2d(*, resolution: int = 128) -> PhaseFieldBenchmark:
    grid = GridSpec((resolution, resolution), spacing=1.0)
    model = AllenCahnModel(
        mobility=1.0,
        quadratic=-1.0,
        quartic=1.0,
        gradient_coefficient=1.0,
    )
    return PhaseFieldBenchmark(
        "shrinking-circle-2d",
        model,
        grid,
        {
            "phi": spherical_nucleus(
                grid,
                radius=0.2 * min(grid.lengths),
                width=2.0,
                inside=1.0,
                outside=-1.0,
            )
        },
        SolverConfig(
            dt=0.05,
            final_time=5.0,
            scheme=TimeScheme.ETDRK4,
            save_every=5,
        ),
        "Allen-Cahn curvature-driven interface motion.",
    )


def dendrite_growth_2d(*, resolution: int = 192, seed: int = 0) -> PhaseFieldBenchmark:
    grid = GridSpec((resolution, resolution), spacing=1.0)
    model = DendriticSolidificationModel(
        anisotropy_strength=0.04,
        symmetry_order=4,
        thermal_coupling=2.0,
    )
    phi = spherical_nucleus(
        grid,
        radius=5.0,
        width=1.0,
        inside=1.0,
        outside=-1.0,
    )
    temperature = -0.45 + random_noise(grid, amplitude=1.0e-4, seed=seed)
    return PhaseFieldBenchmark(
        "dendrite-growth-2d",
        model,
        grid,
        {"phi": phi, "temperature": temperature},
        SolverConfig(
            dt=0.01,
            final_time=4.0,
            scheme=TimeScheme.ETDRK4,
            save_every=10,
        ),
        "Anisotropic thermal dendritic solidification.",
    )


__all__ = [
    "PhaseFieldBenchmark",
    "dendrite_growth_2d",
    "shrinking_circle_2d",
    "spinodal_decomposition_2d",
]
