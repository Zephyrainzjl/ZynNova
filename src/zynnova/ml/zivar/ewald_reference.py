"""Differentiable float64 point-charge Ewald reference for ZIVAR.

This module is intentionally independent from the production electrostatic
path.  It provides a slow, explicit real/reciprocal lattice sum suitable for
physics gates and for certifying particle-mesh error.  The periodic convention
is three-dimensional conducting (tin-foil) Ewald with a uniform neutralising
background for a nonzero net cell charge.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .mesh import validate_boundary

torch = require_torch()

COULOMB_CONSTANT_EV_A = 14.3996454784255


@dataclass(frozen=True, slots=True)
class EwaldParameters:
    """Numerical cutoffs for a direct three-dimensional Ewald sum."""

    alpha_inv_A: float
    real_cutoff_A: float
    reciprocal_cutoff_inv_A: float
    error_target: float

    def __post_init__(self) -> None:
        if self.alpha_inv_A <= 0.0:
            raise ValueError("alpha_inv_A must be positive")
        if self.real_cutoff_A <= 0.0:
            raise ValueError("real_cutoff_A must be positive")
        if self.reciprocal_cutoff_inv_A <= 0.0:
            raise ValueError("reciprocal_cutoff_inv_A must be positive")
        if not 0.0 < self.error_target < 1.0:
            raise ValueError("error_target must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class EwaldResult:
    """Energy and the four signed terms of the Ewald decomposition."""

    energy: Any
    real_energy: Any
    reciprocal_energy: Any
    self_energy: Any
    background_energy: Any


def _validate_cell(cell: Any) -> None:
    if tuple(cell.shape) != (3, 3):
        raise ValueError("cell must have shape [3, 3]")
    determinant = float(torch.linalg.det(cell).detach().abs())
    if not math.isfinite(determinant) or determinant <= 1.0e-12:
        raise ValueError("periodic electrostatics requires a nonsingular cell")


def _validate_inputs(positions: Any, charges: Any, cell: Any | None = None) -> None:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [atoms, 3]")
    if charges.ndim != 1 or charges.shape[0] != positions.shape[0]:
        raise ValueError("charges must have shape [atoms]")
    if not positions.is_floating_point() or not charges.is_floating_point():
        raise TypeError("positions and charges must be floating-point tensors")
    if positions.device != charges.device or positions.dtype != charges.dtype:
        raise ValueError("positions and charges must share device and dtype")
    if cell is not None:
        _validate_cell(cell)
        if cell.device != positions.device or cell.dtype != positions.dtype:
            raise ValueError("cell, positions and charges must share device and dtype")


def _require_reference_precision(reference: Any) -> None:
    if reference.dtype != torch.float64:
        raise TypeError("direct Ewald reference requires torch.float64 inputs")


def plan_ewald(
    cell: Any,
    error_target: float,
    *,
    real_cutoff_A: float | None = None,
) -> EwaldParameters:
    """Choose balanced direct-Ewald cutoffs from a requested error scale.

    Both omitted tails are controlled by the exponential factors
    ``exp(-(alpha*r_cut)**2)`` and
    ``exp(-(k_cut/(2*alpha))**2)``.  The extra factor of ten is a conservative
    margin; the resulting target remains an estimate and must be checked by
    cutoff-doubling for a certified reference value.
    """

    _validate_cell(cell)
    target = float(error_target)
    if not 0.0 < target < 1.0:
        raise ValueError("error_target must lie strictly between zero and one")
    inverse = torch.linalg.inv(cell.detach())
    cell_heights = 1.0 / torch.linalg.vector_norm(inverse, dim=0)
    shortest_height = float(cell_heights.min().cpu())
    cutoff = 0.45 * shortest_height if real_cutoff_A is None else float(real_cutoff_A)
    if cutoff <= 0.0:
        raise ValueError("real_cutoff_A must be positive")
    decay = math.sqrt(-math.log(target * 0.1))
    alpha = decay / cutoff
    reciprocal_cutoff = 2.0 * alpha * decay
    return EwaldParameters(alpha, cutoff, reciprocal_cutoff, target)


def _integer_vectors(bounds: tuple[int, int, int], reference: Any) -> Any:
    count = math.prod(2 * value + 1 for value in bounds)
    if count > 5_000_000:
        raise ValueError(
            "direct Ewald lattice would exceed five million vectors; "
            "increase the real-space cutoff balance or use PME"
        )
    vectors = itertools.product(
        *(range(-value, value + 1) for value in bounds)
    )
    return torch.as_tensor(tuple(vectors), device=reference.device, dtype=reference.dtype)


def _wrapped_positions(positions: Any, cell: Any) -> tuple[Any, Any]:
    fractional = positions @ torch.linalg.inv(cell)
    fractional = fractional - torch.floor(fractional)
    return fractional @ cell, fractional


def real_space_ewald_energy(
    positions: Any,
    charges: Any,
    cell: Any,
    alpha_inv_A: float,
    real_cutoff_A: float,
    *,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
    image_block: int = 128,
) -> Any:
    """Evaluate the explicit erfc-screened real-space lattice sum."""

    _validate_inputs(positions, charges, cell)
    if alpha_inv_A <= 0.0 or real_cutoff_A <= 0.0:
        raise ValueError("Ewald alpha and real cutoff must be positive")
    if image_block < 1:
        raise ValueError("image_block must be positive")
    _, fractional = _wrapped_positions(positions, cell)
    inverse = torch.linalg.inv(cell.detach())
    fractional_radius = (
        float(real_cutoff_A) * torch.linalg.vector_norm(inverse, dim=0)
    )
    bounds = tuple(int(math.ceil(float(value.cpu()))) + 1 for value in fractional_radius)
    integer_images = _integer_vectors(bounds, positions)
    delta_fractional = fractional[:, None, :] - fractional[None, :, :]
    pair_charge = charges[:, None] * charges[None, :]
    atom_count = int(positions.shape[0])
    diagonal = torch.eye(atom_count, device=positions.device, dtype=torch.bool)
    total = positions.new_zeros(())
    tiny = torch.finfo(positions.dtype).tiny

    for start in range(0, integer_images.shape[0], image_block):
        images = integer_images[start : start + image_block]
        displacement = (delta_fractional.unsqueeze(0) + images[:, None, None, :]) @ cell
        squared = displacement.square().sum(-1)
        distance = torch.sqrt(squared.clamp_min(tiny))
        inside = distance <= float(real_cutoff_A)
        zero_image = torch.all(images == 0, dim=-1)[:, None, None]
        valid = inside & ~(zero_image & diagonal[None, :, :])
        coincident = valid & (squared <= torch.finfo(positions.dtype).eps)
        if bool(torch.any(coincident).detach()):
            raise ValueError("distinct periodic point charges occupy a coincident position")
        kernel = torch.special.erfc(float(alpha_inv_A) * distance) / distance
        total = total + (torch.where(valid, kernel, torch.zeros_like(kernel)) * pair_charge).sum()
    return 0.5 * float(coulomb_constant_eV_A) * total


def reciprocal_space_ewald_energy(
    positions: Any,
    charges: Any,
    cell: Any,
    alpha_inv_A: float,
    reciprocal_cutoff_inv_A: float,
    *,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
    wave_block: int = 16_384,
) -> Any:
    """Evaluate the explicit reciprocal-space structure-factor sum."""

    _validate_inputs(positions, charges, cell)
    if alpha_inv_A <= 0.0 or reciprocal_cutoff_inv_A <= 0.0:
        raise ValueError("Ewald alpha and reciprocal cutoff must be positive")
    if wave_block < 1:
        raise ValueError("wave_block must be positive")
    reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
    inverse_reciprocal = torch.linalg.inv(reciprocal.detach())
    integer_radius = float(reciprocal_cutoff_inv_A) * torch.linalg.vector_norm(
        inverse_reciprocal, dim=0
    )
    bounds = tuple(int(math.ceil(float(value.cpu()))) + 1 for value in integer_radius)
    integers = _integer_vectors(bounds, positions)
    nonzero = torch.any(integers != 0, dim=-1)
    integers = integers[nonzero]
    k_vectors = integers @ reciprocal
    k2 = k_vectors.square().sum(-1)
    inside = k2 <= float(reciprocal_cutoff_inv_A) ** 2
    k_vectors, k2 = k_vectors[inside], k2[inside]
    total = positions.new_zeros(())
    for start in range(0, k_vectors.shape[0], wave_block):
        local_k = k_vectors[start : start + wave_block]
        local_k2 = k2[start : start + wave_block]
        phase = positions @ local_k.T
        structure = torch.sum(charges[:, None] * torch.exp(1j * phase), dim=0)
        weight = torch.exp(-local_k2 / (4.0 * float(alpha_inv_A) ** 2)) / local_k2
        total = total + (weight * structure.abs().square()).sum().real
    volume = torch.linalg.det(cell).abs()
    return float(coulomb_constant_eV_A) * (2.0 * math.pi / volume) * total


def ewald_corrections(
    charges: Any,
    cell: Any,
    alpha_inv_A: float,
    *,
    neutralizing_background: bool = True,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
) -> tuple[Any, Any]:
    """Return signed point self and tin-foil uniform-background terms."""

    if alpha_inv_A <= 0.0:
        raise ValueError("alpha_inv_A must be positive")
    volume = torch.linalg.det(cell).abs()
    constant = float(coulomb_constant_eV_A)
    self_energy = -constant * float(alpha_inv_A) / math.sqrt(math.pi) * charges.square().sum()
    total_charge = charges.sum()
    if neutralizing_background:
        background = (
            -constant
            * math.pi
            * total_charge.square()
            / (2.0 * float(alpha_inv_A) ** 2 * volume)
        )
    else:
        if abs(float(total_charge.detach())) > 1.0e-12:
            raise ValueError(
                "a charged periodic cell requires neutralizing_background=True"
            )
        background = charges.new_zeros(())
    return self_energy, background


def ewald_energy(
    positions: Any,
    charges: Any,
    cell: Any,
    pbc: Any,
    parameters: EwaldParameters | None = None,
    *,
    error_target: float = 1.0e-10,
    neutralizing_background: bool = True,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
) -> EwaldResult:
    """Evaluate a differentiable float64 direct Ewald reference energy."""

    validate_boundary(pbc, periodic=True)
    _validate_inputs(positions, charges, cell)
    _require_reference_precision(positions)
    selected = plan_ewald(cell, error_target) if parameters is None else parameters
    real = real_space_ewald_energy(
        positions,
        charges,
        cell,
        selected.alpha_inv_A,
        selected.real_cutoff_A,
        coulomb_constant_eV_A=coulomb_constant_eV_A,
    )
    reciprocal = reciprocal_space_ewald_energy(
        positions,
        charges,
        cell,
        selected.alpha_inv_A,
        selected.reciprocal_cutoff_inv_A,
        coulomb_constant_eV_A=coulomb_constant_eV_A,
    )
    self_energy, background = ewald_corrections(
        charges,
        cell,
        selected.alpha_inv_A,
        neutralizing_background=neutralizing_background,
        coulomb_constant_eV_A=coulomb_constant_eV_A,
    )
    return EwaldResult(
        energy=real + reciprocal + self_energy + background,
        real_energy=real,
        reciprocal_energy=reciprocal,
        self_energy=self_energy,
        background_energy=background,
    )


def isolated_coulomb_energy(
    positions: Any,
    charges: Any,
    pbc: Any = None,
    *,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
    pair_block: int = 65_536,
) -> Any:
    """Evaluate the exact open-boundary point-charge pair energy."""

    validate_boundary(pbc, periodic=False)
    _validate_inputs(positions, charges)
    _require_reference_precision(positions)
    if pair_block < 1:
        raise ValueError("pair_block must be positive")
    count = int(positions.shape[0])
    if count < 2:
        return positions.new_zeros(())
    pairs = torch.triu_indices(count, count, offset=1, device=positions.device)
    pieces = []
    for start in range(0, pairs.shape[1], pair_block):
        local = pairs[:, start : start + pair_block]
        displacement = positions[local[0]] - positions[local[1]]
        squared = displacement.square().sum(-1)
        if bool(torch.any(squared <= torch.finfo(positions.dtype).eps).detach()):
            raise ValueError("distinct point charges occupy a coincident position")
        distance = torch.sqrt(squared)
        pieces.append(charges[local[0]] * charges[local[1]] / distance)
    return float(coulomb_constant_eV_A) * torch.cat(pieces).sum()


__all__ = [
    "COULOMB_CONSTANT_EV_A",
    "EwaldParameters",
    "EwaldResult",
    "ewald_corrections",
    "ewald_energy",
    "isolated_coulomb_energy",
    "plan_ewald",
    "real_space_ewald_energy",
    "reciprocal_space_ewald_energy",
]
