"""Reciprocal-mesh planning primitives for ZIVAR electrostatics.

The objects in this module contain only integer mesh topology and immutable
planning metadata.  They deliberately do not cache tensors derived from a
cell: doing so would sever autograd's cell/virial path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._deps import require_torch

torch = require_torch()


@dataclass(frozen=True, slots=True)
class MeshPlan:
    """A three-dimensional particle-mesh assignment plan.

    ``interpolation_order`` is the order of the cardinal B-spline used both
    for charge assignment and for reciprocal-space window deconvolution.
    ZIVAR currently implements the even orders 2, 4 and 6.
    """

    shape: tuple[int, int, int]
    interpolation_order: int
    spacing_A: tuple[float, float, float]
    error_target: float

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value < 4 for value in self.shape):
            raise ValueError("mesh shape must contain three dimensions >= 4")
        if self.interpolation_order not in {2, 4, 6}:
            raise ValueError("interpolation_order must be one of 2, 4 or 6")
        if len(self.spacing_A) != 3 or any(value <= 0.0 for value in self.spacing_A):
            raise ValueError("mesh spacing must contain three positive values")
        if not 0.0 < self.error_target < 1.0:
            raise ValueError("error_target must lie strictly between zero and one")


def validate_boundary(pbc: Any, *, periodic: bool) -> None:
    """Validate ZIVAR's explicit 3D-periodic or fully-isolated contract.

    Slab and wire boundary conditions need distinct surface corrections and
    are therefore rejected instead of being silently treated as 3D periodic.
    """

    if pbc is None:
        if periodic:
            raise ValueError("periodic electrostatics requires pbc=(True, True, True)")
        return
    flags = torch.as_tensor(pbc, dtype=torch.bool)
    if flags.numel() != 3:
        raise ValueError("pbc must contain exactly three flags")
    flags = flags.reshape(3)
    any_periodic = bool(torch.any(flags).detach())
    all_periodic = bool(torch.all(flags).detach())
    if any_periodic and not all_periodic:
        raise ValueError(
            "partial PBC is unsupported: use fully periodic 3D Ewald/PME "
            "or fully isolated direct electrostatics"
        )
    if periodic and not all_periodic:
        raise ValueError("periodic electrostatics requires pbc=(True, True, True)")
    if not periodic and any_periodic:
        raise ValueError("isolated electrostatics requires pbc=(False, False, False)")


def _smooth_fft_size(minimum: int) -> int:
    """Return the smallest 2/3/5-smooth FFT length at least ``minimum``."""

    candidate = max(4, int(minimum))
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def plan_mesh(
    cell: Any,
    error_target: float,
    alpha_inv_A: float,
    *,
    interpolation_order: int = 4,
    shape: tuple[int, int, int] | None = None,
) -> MeshPlan:
    """Choose an FFT mesh from the Ewald splitting and requested error.

    The spectral cutoff is selected from
    ``exp(-k_cut**2 / (4 alpha**2)) <= error_target / 10``.  An order-dependent
    oversampling factor controls B-spline aliasing.  This is a conservative
    planning heuristic rather than an a-posteriori error certificate; callers
    that need certification must compare against :func:`ewald_energy`.
    """

    if not 0.0 < float(error_target) < 1.0:
        raise ValueError("error_target must lie strictly between zero and one")
    if float(alpha_inv_A) <= 0.0:
        raise ValueError("alpha_inv_A must be positive")
    if interpolation_order not in {2, 4, 6}:
        raise ValueError("interpolation_order must be one of 2, 4 or 6")
    if tuple(cell.shape) != (3, 3):
        raise ValueError("cell must have shape [3, 3]")
    determinant = float(torch.linalg.det(cell).detach().abs())
    if not math.isfinite(determinant) or determinant <= 1.0e-12:
        raise ValueError("cell must be nonsingular")

    lengths = torch.linalg.vector_norm(cell.detach(), dim=1).cpu().tolist()
    if shape is None:
        reciprocal = 2.0 * math.pi * torch.linalg.inv(cell.detach()).T
        reciprocal_steps = torch.linalg.vector_norm(reciprocal, dim=1).cpu().tolist()
        decay = math.sqrt(-math.log(float(error_target) * 0.1))
        reciprocal_cutoff = 2.0 * float(alpha_inv_A) * decay
        alias_factor = min(
            2.5,
            max(1.25, 0.65 * float(error_target) ** (-1.0 / (2 * interpolation_order))),
        )
        selected = []
        for step in reciprocal_steps:
            minimum = math.ceil(alias_factor * (2.0 * reciprocal_cutoff / step + 2.0))
            selected.append(_smooth_fft_size(minimum))
        shape = tuple(selected)  # type: ignore[assignment]
    else:
        shape = tuple(int(value) for value in shape)

    spacing = tuple(float(lengths[axis]) / shape[axis] for axis in range(3))
    return MeshPlan(
        shape=shape,
        interpolation_order=interpolation_order,
        spacing_A=spacing,
        error_target=float(error_target),
    )


__all__ = ["MeshPlan", "plan_mesh", "validate_boundary"]
