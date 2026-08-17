"""Positive-definite direct charge equilibration.

This module implements a fourth-generation charge-transfer path without an
unrolled optimiser. Environment-dependent electronegativities and hardnesses
are read from the local backbone and a single Cholesky/KKT solve returns the
global charges. Gaussian Coulomb matrices include their analytic self term and
are positive semidefinite; the learned positive hardness makes every system
matrix strictly positive definite.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ._deps import require_torch
from .config import ElectronicConfig
from .electrostatics import gaussian_monopole_kernel_and_field

torch = require_torch()


@dataclass(slots=True)
class QEqResult:
    charges: Any
    atomic_energy: Any
    graph_energy: Any
    residual: Any
    coulomb_matrices: tuple[Any, ...]


def _graph_count(batch: Any) -> int:
    return int(batch.max().item()) + 1 if batch.numel() else 0


@lru_cache(maxsize=32)
def _integer_grid(kmax: int, device: str, dtype: Any) -> Any:
    values = tuple(
        item
        for item in itertools.product(range(-kmax, kmax + 1), repeat=3)
        if item != (0, 0, 0)
    )
    return torch.as_tensor(values, device=torch.device(device), dtype=dtype)


def gaussian_coulomb_matrix(
    positions: Any,
    *,
    width_A: float,
    coulomb_constant_eV_A: float,
    cell: Any | None = None,
    periodic: bool = False,
    reciprocal_kmax: int = 3,
) -> Any:
    """Return the PSD Coulomb matrix of equal-width Gaussian densities."""

    count = int(positions.shape[0])
    if count == 0:
        return positions.new_zeros((0, 0))
    sigma = float(width_A)
    constant = float(coulomb_constant_eV_A)
    if periodic:
        if cell is None or cell.shape != (3, 3):
            raise ValueError("periodic QEq requires one 3x3 cell")
        volume = torch.linalg.det(cell).abs()
        if float(volume.detach()) <= 1.0e-12:
            raise ValueError("periodic QEq requires a nonsingular cell")
        integer = _integer_grid(
            reciprocal_kmax, str(positions.device), positions.dtype
        )
        reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
        k_vectors = integer @ reciprocal
        k2 = k_vectors.square().sum(-1)
        phase = positions @ k_vectors.T
        weight = (
            4.0
            * math.pi
            * constant
            / volume
            * torch.exp(-sigma * sigma * k2)
            / k2
        )
        cosine, sine = torch.cos(phase), torch.sin(phase)
        # Sum of two Gram matrices: exactly symmetric PSD up to roundoff.
        matrix = (cosine * weight) @ cosine.T + (sine * weight) @ sine.T
        return 0.5 * (matrix + matrix.T)
    difference = positions[:, None, :] - positions[None, :, :]
    kernel, _ = gaussian_monopole_kernel_and_field(difference, sigma)
    matrix = constant * kernel
    self_value = constant / (math.sqrt(math.pi) * sigma)
    identity = torch.eye(count, device=positions.device, dtype=positions.dtype)
    return matrix * (1.0 - identity) + self_value * identity


def _condition_vector(
    value: Any | None,
    *,
    selection: Any,
    graph: int,
    graph_count: int,
    default: float = 0.0,
) -> Any:
    if value is None:
        return selection.new_full(
            (selection.numel(),), float(default), dtype=torch.get_default_dtype()
        )
    if value.ndim == 0 or value.shape == (1,):
        return value.reshape(1).expand(selection.numel())
    if value.shape == (graph_count,):
        return value[graph].expand(selection.numel())
    if value.ndim == 1 and value.shape[0] >= int(selection.max().item()) + 1:
        return value[selection]
    raise ValueError("condition must be scalar, per graph, or per atom")


def _graph_scalar(value: Any, graph: int, graph_count: int, dtype: Any) -> Any:
    value = value.to(dtype=dtype)
    if value.ndim == 0 or value.shape == (1,):
        return value.reshape(())
    if value.shape == (graph_count,):
        return value[graph]
    raise ValueError("charge constraint must be scalar or have shape [B]")


def _constraints_for_graph(
    selection: Any,
    graph: int,
    graph_count: int,
    conditions: dict[str, Any],
    config: ElectronicConfig,
    dtype: Any,
) -> tuple[Any | None, Any | None]:
    count = int(selection.numel())
    rows: list[Any] = []
    targets: list[Any] = []
    total_charge = conditions.get("total_charge")
    if config.boundary_mode == "fixed_charge":
        target = (
            selection.new_zeros((), dtype=dtype)
            if total_charge is None
            else _graph_scalar(
                total_charge.to(device=selection.device), graph, graph_count, dtype
            )
        )
        rows.append(torch.ones(count, device=selection.device, dtype=dtype))
        targets.append(target)
    elif config.boundary_mode == "mixed":
        reservoir = conditions.get("reservoir_mask")
        if reservoir is None or reservoir.ndim != 1:
            raise ValueError("mixed QEq requires reservoir_mask with shape [N]")
        closed = 1.0 - reservoir.to(device=selection.device, dtype=dtype)[selection]
        if bool(torch.any((closed < 0.0) | (closed > 1.0)).detach()):
            raise ValueError("reservoir_mask values must lie in [0, 1]")
        target = conditions.get("closed_region_charge")
        if target is None:
            raise ValueError("mixed QEq requires closed_region_charge")
        rows.append(closed)
        targets.append(
            _graph_scalar(
                target.to(device=selection.device), graph, graph_count, dtype
            )
        )
    membership = conditions.get("fragment_membership")
    fragment_charge = conditions.get("fragment_charge")
    if fragment_charge is not None and membership is None:
        raise ValueError("fragment_charge requires fragment_membership")
    if membership is not None:
        if membership.ndim != 2:
            raise ValueError("fragment_membership must have shape [N,F]")
        local = membership.to(device=selection.device, dtype=dtype)[selection]
        active = local.abs().sum(0) > 1.0e-12
        if bool(torch.any(active).detach()):
            if fragment_charge is None:
                raise ValueError("QEq fragment constraints require fragment_charge")
            for column in torch.nonzero(active, as_tuple=False).flatten().tolist():
                rows.append(local[:, column])
                targets.append(
                    fragment_charge.to(device=selection.device, dtype=dtype)[column]
                )
    fixed_mask = conditions.get("fixed_charge_mask")
    fixed_values = conditions.get("fixed_charges")
    if fixed_values is not None and fixed_mask is None:
        raise ValueError("fixed_charges requires fixed_charge_mask")
    if fixed_mask is not None:
        if fixed_values is None:
            raise ValueError("fixed_charge_mask requires fixed_charges")
        local_mask = fixed_mask.to(device=selection.device, dtype=torch.bool)[selection]
        prescribed = fixed_values.to(device=selection.device, dtype=dtype)[selection]
        identity = torch.eye(count, device=selection.device, dtype=dtype)
        for atom in torch.nonzero(local_mask, as_tuple=False).flatten().tolist():
            rows.append(identity[atom])
            targets.append(prescribed[atom])
    if not rows:
        return None, None
    matrix = torch.stack(rows)
    target = torch.stack(targets)
    # Remove only *consistent* redundant rows before forming the Schur
    # complement. This supports total + exhaustive fragment/fixed constraints
    # without using a pseudoinverse or hiding an inconsistent specification.
    detached = matrix.detach()
    rank = int(torch.linalg.matrix_rank(detached).item())
    augmented = torch.cat((detached, target.detach()[:, None]), dim=1)
    if int(torch.linalg.matrix_rank(augmented).item()) != rank:
        raise ValueError("QEq constraints are inconsistent")
    selected: list[int] = []
    current_rank = 0
    for row in range(matrix.shape[0]):
        candidate = detached[selected + [row]]
        candidate_rank = int(torch.linalg.matrix_rank(candidate).item())
        if candidate_rank > current_rank:
            selected.append(row)
            current_rank = candidate_rank
        if current_rank == rank:
            break
    index = torch.as_tensor(selected, device=matrix.device, dtype=torch.long)
    return matrix[index], target[index]


def solve_qeq(
    electronegativity: Any,
    hardness: Any,
    positions: Any,
    batch: Any,
    *,
    cell: Any | None,
    pbc: Any | None,
    conditions: dict[str, Any],
    config: ElectronicConfig,
) -> QEqResult:
    """Solve every graph independently with a Cholesky/Schur KKT factorisation."""

    if electronegativity.shape != hardness.shape or electronegativity.ndim != 1:
        raise ValueError("QEq coefficients must have shape [N]")
    graph_count = _graph_count(batch)
    charges = electronegativity.new_zeros(electronegativity.shape)
    atomic_energy = electronegativity.new_zeros(electronegativity.shape)
    graph_energy = electronegativity.new_zeros(graph_count)
    residual = electronegativity.new_zeros(graph_count)
    matrices: list[Any] = []
    if pbc is not None and pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    electrode_potential = conditions.get("electrode_potential")
    membership = conditions.get("fragment_membership")
    if membership is not None:
        if membership.shape[0] != positions.shape[0] or membership.ndim != 2:
            raise ValueError("fragment_membership must have shape [N,F]")
        local_membership = membership.to(device=positions.device, dtype=positions.dtype)
        tolerance = 100.0 * torch.finfo(positions.dtype).eps
        if bool(
            torch.any(
                (local_membership - local_membership.round()).abs() > tolerance
            ).detach()
        ):
            raise ValueError("fragment_membership must be binary")
        if bool(torch.any(local_membership.sum(-1) > 1).detach()):
            raise ValueError("fragment memberships must be disjoint")
        presence = positions.new_zeros((graph_count, local_membership.shape[1]))
        presence.index_add_(0, batch, local_membership)
        if bool(torch.any((presence > 0).sum(0) > 1).detach()):
            raise ValueError("each fragment must be confined to one graph")
        fragment_charge = conditions.get("fragment_charge")
        if fragment_charge is None or fragment_charge.shape != (local_membership.shape[1],):
            raise ValueError("fragment_charge must have shape [F]")
    fixed_mask = conditions.get("fixed_charge_mask")
    if fixed_mask is not None:
        if fixed_mask.shape != positions.shape[:1]:
            raise ValueError("fixed_charge_mask must have shape [N]")
        local_mask = fixed_mask.to(device=positions.device, dtype=positions.dtype)
        tolerance = 100.0 * torch.finfo(positions.dtype).eps
        if bool(torch.any((local_mask - local_mask.round()).abs() > tolerance).detach()):
            raise ValueError("fixed_charge_mask must be binary")
        fixed_values = conditions.get("fixed_charges")
        if fixed_values is None or fixed_values.shape != positions.shape[:1]:
            raise ValueError("fixed_charges must have shape [N]")
    for graph in range(graph_count):
        selection = torch.nonzero(batch == graph, as_tuple=False).flatten()
        count = int(selection.numel())
        if count > config.qeq_max_atoms:
            raise ValueError(
                f"direct QEq graph has {count} atoms, exceeding qeq_max_atoms="
                f"{config.qeq_max_atoms}; use method='polar' for large-scale MD"
            )
        periodic = bool(
            cell is not None
            and pbc is not None
            and bool(torch.all(pbc[graph]).detach())
        )
        if pbc is not None and bool(torch.any(pbc[graph]).detach()) and not periodic:
            raise ValueError("QEq does not approximate partial periodic boundaries")
        local_cell = None if cell is None else cell[graph]
        coulomb = gaussian_coulomb_matrix(
            positions[selection],
            width_A=config.gaussian_width_A,
            coulomb_constant_eV_A=config.coulomb_constant_eV_A,
            cell=local_cell,
            periodic=periodic,
            reciprocal_kmax=config.reciprocal_kmax,
        )
        matrices.append(coulomb)
        local_hardness = hardness[selection].clamp_min(config.hardness_floor_eV)
        system = coulomb + torch.diag(local_hardness + config.qeq_jitter_eV)
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.detach().max().item()) != 0:
            raise FloatingPointError("positive-definite QEq Cholesky factorisation failed")
        chi = electronegativity[selection]
        if electrode_potential is not None:
            potential = _condition_vector(
                electrode_potential,
                selection=selection,
                graph=graph,
                graph_count=graph_count,
            ).to(device=chi.device, dtype=chi.dtype)
            chi = chi - potential
        electric = conditions.get("external_electric_field")
        if electric is not None:
            electric = electric.to(device=chi.device, dtype=chi.dtype)
            if electric.shape != (graph_count, 3):
                raise ValueError("external_electric_field must have shape [B,3]")
            origin = conditions.get("electric_field_origin")
            if origin is None:
                local_origin = positions.new_zeros(3)
            else:
                origin = origin.to(device=chi.device, dtype=chi.dtype)
                if origin.shape != (graph_count, 3):
                    raise ValueError("electric_field_origin must have shape [B,3]")
                local_origin = origin[graph]
            applied_potential = -(
                (positions[selection] - local_origin) * electric[graph]
            ).sum(-1)
            chi = chi + applied_potential
        def solve(rhs: Any, local_factor: Any = factor) -> Any:
            return torch.cholesky_solve(rhs, local_factor)
        unconstrained = -solve(chi[:, None]).squeeze(-1)
        constraint, target = _constraints_for_graph(
            selection,
            graph,
            graph_count,
            conditions,
            config,
            chi.dtype,
        )
        multiplier = None
        if constraint is None:
            local_charge = unconstrained
        else:
            response = solve(constraint.T)
            schur = constraint @ response
            schur_factor, schur_info = torch.linalg.cholesky_ex(schur)
            if int(schur_info.detach().max().item()) != 0:
                raise ValueError("QEq constraints are singular or inconsistent")
            right = (constraint @ unconstrained - target)[:, None]
            multiplier = torch.cholesky_solve(right, schur_factor).squeeze(-1)
            local_charge = unconstrained - response @ multiplier
        stationarity = system @ local_charge + chi
        if constraint is not None and multiplier is not None:
            stationarity = stationarity + constraint.T @ multiplier
        epsilon = 32.0 * torch.finfo(stationarity.dtype).eps
        residual[graph] = (
            torch.sqrt(stationarity.square().mean() + epsilon * epsilon) - epsilon
        )
        # At the exact constrained minimum, the envelope theorem evaluates
        # energy/force derivatives at fixed q. Keeping the supervised charge
        # tensor differentiable while detaching only its energy view avoids
        # unnecessary derivatives through the Cholesky factorisation.
        energy_charge = (
            local_charge.detach()
            if config.variational_envelope_forces
            else local_charge
        )
        local_atomic = chi * energy_charge + 0.5 * energy_charge * (
            system @ energy_charge
        )
        charges[selection] = local_charge
        atomic_energy[selection] = local_atomic
        graph_energy[graph] = local_atomic.sum()
    return QEqResult(
        charges=charges,
        atomic_energy=atomic_energy,
        graph_energy=graph_energy,
        residual=residual,
        coulomb_matrices=tuple(matrices),
    )


__all__ = ["QEqResult", "gaussian_coulomb_matrix", "solve_qeq"]
