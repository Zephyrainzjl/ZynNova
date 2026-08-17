from __future__ import annotations

import math
from typing import Any

from ...common import require_torch
from .graph import normalize_cell_pbc, scatter_mean, scatter_sum, smooth_cutoff

torch = require_torch()
nn = torch.nn

COULOMB_EV_A = 14.3996454784255
BOHR_RADIUS_A = 0.529177210903
_ZBL_A = (0.1818, 0.5099, 0.2802, 0.02817)
_ZBL_B = (3.2, 0.9423, 0.4029, 0.2016)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1.0e-8)))


def smooth_switch(distance: Any, inner: float, outer: float) -> Any:
    if not 0 < inner < outer:
        raise ValueError("require 0 < inner < outer")
    coordinate = ((distance - inner) / (outer - inner)).clamp(0.0, 1.0)
    transition = 1.0 - (10.0 * coordinate**3 - 15.0 * coordinate**4 + 6.0 * coordinate**5)
    return torch.where(
        distance <= inner,
        torch.ones_like(distance),
        torch.where(distance < outer, transition, torch.zeros_like(distance)),
    )


def zbl_pair_energy(
    atomic_number_i: Any,
    atomic_number_j: Any,
    distance: Any,
    *,
    inner_A: float,
    outer_A: float,
) -> Any:
    """Universal screened nuclear repulsion in eV for one directed edge list."""

    z_i = atomic_number_i.to(distance.dtype).clamp_min(1.0)
    z_j = atomic_number_j.to(distance.dtype).clamp_min(1.0)
    screening_length = 0.8854 * BOHR_RADIUS_A / (torch.pow(z_i, 0.23) + torch.pow(z_j, 0.23))
    reduced = distance.clamp_min(1.0e-6) / screening_length
    screening = distance.new_zeros(distance.shape)
    for coefficient, exponent in zip(_ZBL_A, _ZBL_B, strict=True):
        screening = screening + coefficient * torch.exp(-exponent * reduced)
    energy = COULOMB_EV_A * z_i * z_j * screening / distance.clamp_min(1.0e-6)
    return energy * smooth_switch(distance, inner_A, outer_A)


class ZBLRepulsion(nn.Module):
    def __init__(
        self,
        *,
        inner_A: float,
        outer_A: float,
        learnable_scale: bool,
    ) -> None:
        super().__init__()
        self.inner_A = float(inner_A)
        self.outer_A = float(outer_A)
        raw = torch.tensor(_inverse_softplus(1.0))
        if learnable_scale:
            self.raw_scale = nn.Parameter(raw)
        else:
            self.register_buffer("raw_scale", raw)

    @property
    def scale(self) -> Any:
        return torch.nn.functional.softplus(self.raw_scale)

    def forward(self, z_i: Any, z_j: Any, distance: Any) -> Any:
        return self.scale * zbl_pair_energy(
            z_i,
            z_j,
            distance,
            inner_A=self.inner_A,
            outer_A=self.outer_A,
        )


class DampedDispersion(nn.Module):
    """Learned positive C6/R0 tables with a finite, C2 local pair energy."""

    def __init__(
        self,
        max_atomic_number: int,
        *,
        cutoff_A: float,
        initial_c6_eV_A6: float,
    ) -> None:
        super().__init__()
        self.cutoff_A = float(cutoff_A)
        amplitude = math.sqrt(initial_c6_eV_A6)
        raw_c6 = _inverse_softplus(amplitude)
        raw_r0 = _inverse_softplus(1.5)
        self.raw_c6 = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)
        self.raw_r0 = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)
        nn.init.constant_(self.raw_c6.weight, raw_c6)
        nn.init.constant_(self.raw_r0.weight, raw_r0)
        with torch.no_grad():
            self.raw_c6.weight[0].zero_()
            self.raw_r0.weight[0].zero_()

    def forward(self, z_i: Any, z_j: Any, distance: Any) -> Any:
        c6_i = torch.nn.functional.softplus(self.raw_c6(z_i.long()).squeeze(-1))
        c6_j = torch.nn.functional.softplus(self.raw_c6(z_j.long()).squeeze(-1))
        r0_i = torch.nn.functional.softplus(self.raw_r0(z_i.long()).squeeze(-1))
        r0_j = torch.nn.functional.softplus(self.raw_r0(z_j.long()).squeeze(-1))
        c6 = c6_i * c6_j
        r0 = r0_i + r0_j
        denominator = distance**6 + r0**6 + 1.0e-12
        return -c6 / denominator * smooth_cutoff(distance, self.cutoff_A)


def _minimum_image(displacement: Any, cell: Any, pbc: Any) -> Any:
    if not bool(torch.any(pbc).item()):
        return displacement
    determinant = torch.det(cell)
    if float(torch.abs(determinant).detach().cpu()) < 1.0e-12:
        raise ValueError("periodic charge equilibration requires a non-singular cell")
    fractional = displacement @ torch.linalg.inv(cell)
    fractional = fractional - torch.round(fractional) * pbc.to(displacement.dtype)
    return fractional @ cell


class ScreenedChargeEquilibration(nn.Module):
    """Differentiable constrained QEq solve with a smooth screened kernel.

    The solve enforces the graph-level total charge exactly. It is suitable for
    molecules and moderate single-domain cells. Distributed ML-IAP explicitly
    disables this global solve because a domain-local KKT system would not impose
    the correct global constraint.
    """

    def __init__(
        self,
        *,
        screening_A_inv: float,
        softening_A: float,
        max_atoms: int,
    ) -> None:
        super().__init__()
        self.screening_A_inv = float(screening_A_inv)
        self.softening_A = float(softening_A)
        self.max_atoms = int(max_atoms)

    def forward(
        self,
        electronegativity: Any,
        hardness: Any,
        positions: Any,
        batch: Any,
        cell: Any | None,
        pbc: Any | None,
        total_charge: Any | None,
    ) -> tuple[Any, Any, Any]:
        node_count = positions.shape[0]
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        cell, pbc = normalize_cell_pbc(
            cell,
            pbc,
            graph_count=graph_count,
            positions=positions,
        )
        if total_charge is None:
            total_charge = positions.new_zeros((graph_count,))
        elif not torch.is_tensor(total_charge):
            total_charge = torch.as_tensor(
                total_charge,
                device=positions.device,
                dtype=positions.dtype,
            )
        else:
            total_charge = total_charge.to(positions)
        total_charge = total_charge.reshape(-1)
        if total_charge.numel() == 1 and graph_count > 1:
            total_charge = total_charge.expand(graph_count)
        if total_charge.shape != (graph_count,):
            raise ValueError("total_charge must be scalar or have shape [graphs]")

        charges = positions.new_zeros((node_count,))
        atomic_energy = positions.new_zeros((node_count,))
        graph_energy = positions.new_zeros((graph_count,))
        for graph_index in range(graph_count):
            atom_index = torch.nonzero(batch == graph_index, as_tuple=False).reshape(-1)
            count = atom_index.numel()
            if count == 0:
                continue
            if count > self.max_atoms:
                raise ValueError(
                    f"QEq graph has {count} atoms, exceeding qeq_max_atoms={self.max_atoms}"
                )
            local_position = positions[atom_index]
            displacement = local_position[None, :, :] - local_position[:, None, :]
            displacement = _minimum_image(
                displacement,
                cell[graph_index],
                pbc[graph_index],
            )
            softened_distance = torch.sqrt(displacement.square().sum(dim=-1) + self.softening_A**2)
            coulomb = (
                COULOMB_EV_A
                * torch.exp(-self.screening_A_inv * softened_distance)
                / softened_distance
            )
            local_hardness = hardness[atom_index]
            interaction = coulomb + torch.diag(local_hardness)
            ones = positions.new_ones((count, 1))
            kkt = torch.cat(
                (
                    torch.cat((interaction, ones), dim=1),
                    torch.cat((ones.transpose(0, 1), positions.new_zeros((1, 1))), dim=1),
                ),
                dim=0,
            )
            rhs = torch.cat(
                (
                    -electronegativity[atom_index],
                    total_charge[graph_index : graph_index + 1],
                )
            )
            solution = torch.linalg.solve(kkt, rhs)
            local_charge = solution[:count]
            local_atomic_energy = electronegativity[
                atom_index
            ] * local_charge + 0.5 * local_charge * (interaction @ local_charge)
            charges[atom_index] = local_charge
            atomic_energy[atom_index] = local_atomic_energy
            graph_energy[graph_index] = local_atomic_energy.sum()
        return charges, atomic_energy, graph_energy


def dipole_from_charges(charges: Any, positions: Any, batch: Any) -> Any:
    """Origin-stable graph dipoles in e·Å using the geometric center."""

    graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
    center = scatter_mean(positions, batch, graph_count)
    centered = positions - center[batch]
    return scatter_sum(charges[:, None] * centered, batch, graph_count)


__all__ = [
    "BOHR_RADIUS_A",
    "COULOMB_EV_A",
    "DampedDispersion",
    "ScreenedChargeEquilibration",
    "ZBLRepulsion",
    "dipole_from_charges",
    "smooth_switch",
    "zbl_pair_energy",
]
