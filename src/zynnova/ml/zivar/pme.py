"""Differentiable monopole particle-mesh Ewald implemented with torch FFTs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .ewald_reference import (
    COULOMB_CONSTANT_EV_A,
    EwaldParameters,
    EwaldResult,
    _validate_inputs,
    ewald_corrections,
    plan_ewald,
    real_space_ewald_energy,
)
from .mesh import MeshPlan, plan_mesh, validate_boundary

torch = require_torch()


@dataclass(frozen=True, slots=True)
class PMEPlan:
    """Immutable real/mesh decomposition for monopole PME."""

    ewald: EwaldParameters
    mesh: MeshPlan

    def __post_init__(self) -> None:
        if self.mesh.error_target != self.ewald.error_target:
            raise ValueError("PME real-space and mesh error targets must match")


def plan_pme(
    cell: Any,
    error_target: float,
    *,
    real_cutoff_A: float | None = None,
    interpolation_order: int = 4,
    mesh_shape: tuple[int, int, int] | None = None,
) -> PMEPlan:
    """Create a real-space Ewald split and an FFT mesh for the same target."""

    ewald = plan_ewald(cell, error_target, real_cutoff_A=real_cutoff_A)
    mesh = plan_mesh(
        cell,
        error_target,
        ewald.alpha_inv_A,
        interpolation_order=interpolation_order,
        shape=mesh_shape,
    )
    return PMEPlan(ewald=ewald, mesh=mesh)


def _cardinal_bspline(argument: Any, order: int) -> Any:
    """Evaluate the compact cardinal B-spline ``M_order(argument)``."""

    value = torch.zeros_like(argument)
    power = order - 1
    for knot in range(order + 1):
        value = value + (
            (-1.0) ** knot
            * math.comb(order, knot)
            * torch.clamp_min(argument - float(knot), 0.0).pow(power)
        )
    return value / math.factorial(power)


def assign_charges(
    positions: Any,
    charges: Any,
    cell: Any,
    mesh: MeshPlan,
) -> Any:
    """Assign particle charges to a periodic grid with cardinal B-splines."""

    _validate_inputs(positions, charges, cell)
    order = mesh.interpolation_order
    shape = mesh.shape
    fractional = positions @ torch.linalg.inv(cell)
    fractional = fractional - torch.floor(fractional)
    scale = positions.new_tensor(shape)
    grid_position = fractional * scale
    base = torch.floor(grid_position).to(dtype=torch.long)
    fraction = grid_position - base.to(dtype=positions.dtype)
    offsets = torch.arange(
        1 - order // 2,
        order // 2 + 1,
        device=positions.device,
        dtype=torch.long,
    )

    indices = []
    weights = []
    for axis in range(3):
        local_index = torch.remainder(base[:, axis, None] + offsets[None, :], shape[axis])
        argument = (
            fraction[:, axis, None]
            - offsets.to(dtype=positions.dtype)[None, :]
            + float(order) / 2.0
        )
        local_weight = _cardinal_bspline(argument, order)
        # Enforce exact charge conservation against cancellation at high order.
        local_weight = local_weight / local_weight.sum(dim=1, keepdim=True)
        indices.append(local_index)
        weights.append(local_weight)

    ix = indices[0][:, :, None, None].expand(-1, order, order, order)
    iy = indices[1][:, None, :, None].expand(-1, order, order, order)
    iz = indices[2][:, None, None, :].expand(-1, order, order, order)
    assignment = (
        weights[0][:, :, None, None]
        * weights[1][:, None, :, None]
        * weights[2][:, None, None, :]
    )
    flat_index = ((ix * shape[1] + iy) * shape[2] + iz).reshape(-1)
    values = (charges[:, None, None, None] * assignment).reshape(-1)
    grid = positions.new_zeros(math.prod(shape)).index_add(0, flat_index, values)
    return grid.reshape(shape)


def reciprocal_mesh_energy(
    positions: Any,
    charges: Any,
    cell: Any,
    alpha_inv_A: float,
    mesh: MeshPlan,
    *,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
) -> Any:
    """Evaluate the reciprocal PME term by charge assignment and a 3D FFT."""

    _validate_inputs(positions, charges, cell)
    if alpha_inv_A <= 0.0:
        raise ValueError("alpha_inv_A must be positive")
    charge_grid = assign_charges(positions, charges, cell, mesh)
    spectrum = torch.fft.fftn(charge_grid)

    frequencies = tuple(
        torch.fft.fftfreq(size, device=positions.device, dtype=positions.dtype)
        for size in mesh.shape
    )
    integer_axes = tuple(
        frequencies[axis] * float(mesh.shape[axis]) for axis in range(3)
    )
    integers = torch.stack(
        torch.meshgrid(*integer_axes, indexing="ij"), dim=-1
    )
    reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
    k_vectors = integers @ reciprocal
    k2 = k_vectors.square().sum(-1)

    assignment_window = (
        torch.sinc(frequencies[0])[:, None, None].pow(mesh.interpolation_order)
        * torch.sinc(frequencies[1])[None, :, None].pow(mesh.interpolation_order)
        * torch.sinc(frequencies[2])[None, None, :].pow(mesh.interpolation_order)
    )
    window2 = assignment_window.square().clamp_min(torch.finfo(positions.dtype).eps)
    nonzero = k2 > 0.0
    safe_k2 = torch.where(nonzero, k2, torch.ones_like(k2))
    influence = (
        torch.exp(-safe_k2 / (4.0 * float(alpha_inv_A) ** 2))
        / safe_k2
        / window2
    )
    contribution = torch.where(
        nonzero,
        influence * spectrum.abs().square(),
        torch.zeros_like(influence),
    )
    volume = torch.linalg.det(cell).abs()
    return (
        float(coulomb_constant_eV_A)
        * (2.0 * math.pi / volume)
        * contribution.sum().real
    )


def pme_energy(
    positions: Any,
    charges: Any,
    cell: Any,
    pbc: Any,
    plan: PMEPlan | None = None,
    *,
    error_target: float = 1.0e-6,
    neutralizing_background: bool = True,
    coulomb_constant_eV_A: float = COULOMB_CONSTANT_EV_A,
) -> EwaldResult:
    """Evaluate real, FFT reciprocal, self and background PME terms."""

    validate_boundary(pbc, periodic=True)
    _validate_inputs(positions, charges, cell)
    selected = plan_pme(cell, error_target) if plan is None else plan
    real = real_space_ewald_energy(
        positions,
        charges,
        cell,
        selected.ewald.alpha_inv_A,
        selected.ewald.real_cutoff_A,
        coulomb_constant_eV_A=coulomb_constant_eV_A,
    )
    reciprocal = reciprocal_mesh_energy(
        positions,
        charges,
        cell,
        selected.ewald.alpha_inv_A,
        selected.mesh,
        coulomb_constant_eV_A=coulomb_constant_eV_A,
    )
    self_energy, background = ewald_corrections(
        charges,
        cell,
        selected.ewald.alpha_inv_A,
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


__all__ = [
    "PMEPlan",
    "assign_charges",
    "plan_pme",
    "pme_energy",
    "reciprocal_mesh_energy",
]
