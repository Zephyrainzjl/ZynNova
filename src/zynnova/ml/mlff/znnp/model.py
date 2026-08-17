from __future__ import annotations

import math
from typing import Any

from ...common import require_torch
from .config import ZNNPModelConfig


torch = require_torch()
nn = torch.nn


def scatter_sum(values, index, dim_size: int):
    output = values.new_zeros((dim_size, *values.shape[1:]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff
        gamma = 1.0 / max(spacing * spacing, 1.0e-8)
        self.register_buffer("centers", centers)
        self.gamma = gamma
        self.cutoff = float(cutoff)

    def forward(self, distances):
        basis = torch.exp(-self.gamma * (distances[:, None] - self.centers[None, :]).square())
        cutoff = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        cutoff = cutoff * (distances < self.cutoff).to(distances.dtype)
        return basis * cutoff[:, None]


class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.sender = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.filter = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, receiver, sender, radial):
        message = self.sender(node_features[sender]) * self.filter(radial)
        aggregated = scatter_sum(message, receiver, node_features.shape[0])
        update = self.update(torch.cat((node_features, aggregated), dim=-1))
        return self.norm(node_features + update)


def _cell_for_graph(cell, graph_index: int):
    if cell.ndim == 2:
        return cell
    return cell[graph_index]


def _pbc_for_graph(pbc, graph_index: int):
    if pbc.ndim == 1:
        return pbc
    return pbc[graph_index]


def build_radius_graph(
    positions,
    batch,
    cell,
    pbc,
    *,
    cutoff: float,
    max_neighbors: int | None = None,
):
    """Differentiable O(N²) reference neighbor builder.

    It supports variable-size batches and triclinic minimum-image wrapping. The
    implementation is intentionally dependency-free and is appropriate for small
    and medium systems. Production deployments may replace it with a compiled
    neighbor-list adapter while keeping the model contract unchanged.
    """

    receivers: list[Any] = []
    senders: list[Any] = []
    vectors: list[Any] = []
    distances: list[Any] = []
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_index in range(graph_count):
        atom_indices = torch.nonzero(batch == graph_index, as_tuple=False).reshape(-1)
        local = positions[atom_indices]
        count = local.shape[0]
        if count <= 1:
            continue
        delta = local[None, :, :] - local[:, None, :]
        graph_cell = _cell_for_graph(cell, graph_index)
        graph_pbc = _pbc_for_graph(pbc, graph_index).to(dtype=positions.dtype)
        if bool(torch.any(graph_pbc)):
            determinant = torch.det(graph_cell)
            if float(torch.abs(determinant).detach().cpu()) < 1.0e-12:
                raise ValueError("periodic neighbor construction requires a non-singular cell")
            fractional = delta @ torch.linalg.inv(graph_cell)
            fractional = fractional - torch.round(fractional) * graph_pbc
            delta = fractional @ graph_cell
        distance = torch.linalg.vector_norm(delta, dim=-1)
        pair_mask = (~torch.eye(count, device=positions.device, dtype=torch.bool)) & (
            distance < cutoff
        )
        if max_neighbors is None:
            pair_indices = torch.nonzero(pair_mask, as_tuple=False)
        else:
            selected: list[Any] = []
            for receiver in range(count):
                candidates = torch.nonzero(pair_mask[receiver], as_tuple=False).reshape(-1)
                if candidates.numel() > max_neighbors:
                    order = torch.argsort(distance[receiver, candidates])[:max_neighbors]
                    candidates = candidates[order]
                if candidates.numel():
                    selected.append(
                        torch.stack(
                            (
                                torch.full_like(candidates, receiver),
                                candidates,
                            ),
                            dim=1,
                        )
                    )
            pair_indices = (
                torch.cat(selected, dim=0)
                if selected
                else torch.empty((0, 2), device=positions.device, dtype=torch.long)
            )
        if not pair_indices.numel():
            continue
        local_receiver, local_sender = pair_indices.unbind(dim=1)
        receivers.append(atom_indices[local_receiver])
        senders.append(atom_indices[local_sender])
        vectors.append(delta[local_receiver, local_sender])
        distances.append(distance[local_receiver, local_sender])
    if not receivers:
        empty_index = torch.empty(0, device=positions.device, dtype=torch.long)
        return (
            torch.stack((empty_index, empty_index)),
            positions.new_empty((0, 3)),
            positions.new_empty((0,)),
        )
    receiver = torch.cat(receivers)
    sender = torch.cat(senders)
    return (
        torch.stack((receiver, sender), dim=0),
        torch.cat(vectors, dim=0),
        torch.cat(distances, dim=0),
    )


class ZNNP(nn.Module):
    """ZynNova neural-neighbor potential.

    The model is invariant to translations and rotations and predicts an extensive
    total energy by summing learned atomic contributions. Forces are obtained from
    ``-dE/dR`` and are therefore energy-conserving by construction.
    """

    def __init__(self, config: ZNNPModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ZNNPModelConfig()
        self.embedding = nn.Embedding(
            self.config.max_atomic_number + 1,
            self.config.hidden_dim,
            padding_idx=0,
        )
        self.rbf = GaussianRBF(self.config.num_rbf, self.config.cutoff_A)
        self.interactions = nn.ModuleList(
            InteractionBlock(self.config.hidden_dim, self.config.num_rbf)
            for _ in range(self.config.num_interactions)
        )
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.register_buffer(
            "energy_shift_eV_per_atom",
            torch.tensor(float(self.config.energy_shift_eV_per_atom)),
        )
        self.register_buffer(
            "energy_scale_eV",
            torch.tensor(float(self.config.energy_scale_eV)),
        )

    def set_energy_normalization(self, shift_eV_per_atom: float, scale_eV: float) -> None:
        self.energy_shift_eV_per_atom.fill_(float(shift_eV_per_atom))
        self.energy_scale_eV.fill_(max(float(scale_eV), 1.0e-8))
        self.config.energy_shift_eV_per_atom = float(shift_eV_per_atom)
        self.config.energy_scale_eV = max(float(scale_eV), 1.0e-8)

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        z = inputs.get("z", inputs.get("atomic_numbers"))
        positions = inputs.get("pos", inputs.get("positions"))
        if z is None or positions is None:
            raise KeyError("ZNNP requires z/atomic_numbers and pos/positions")
        batch = inputs.get("batch")
        if batch is None:
            batch = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
        cell = inputs.get("cell")
        if cell is None:
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
            cell = positions.new_zeros((graph_count, 3, 3))
        pbc = inputs.get("pbc")
        if pbc is None:
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
            pbc = torch.zeros((graph_count, 3), device=z.device, dtype=torch.bool)
        edge_index, edge_vector, edge_distance = build_radius_graph(
            positions,
            batch,
            cell,
            pbc,
            cutoff=self.config.cutoff_A,
            max_neighbors=self.config.max_neighbors,
        )
        node_features = self.embedding(z.long())
        if edge_distance.numel():
            radial = self.rbf(edge_distance)
            receiver, sender = edge_index
            for interaction in self.interactions:
                node_features = interaction(node_features, receiver, sender, radial)
        atomic_raw = self.energy_head(node_features).squeeze(-1)
        atomic_energy = atomic_raw * self.energy_scale_eV + self.energy_shift_eV_per_atom
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        total_energy = scatter_sum(atomic_energy, batch, graph_count)
        return {
            "energy": total_energy,
            "atomic_energies": atomic_energy,
            "edge_index": edge_index,
            "edge_vector": edge_vector,
            "edge_distance": edge_distance,
        }

    def energy_and_forces(self, inputs: dict[str, Any], *, create_graph: bool = False):
        positions = inputs.get("pos", inputs.get("positions"))
        if positions is None:
            raise KeyError("positions are required")
        if not positions.requires_grad:
            positions = positions.clone().requires_grad_(True)
            inputs = dict(inputs)
            inputs["pos"] = positions
            inputs["positions"] = positions
        output = self(inputs)
        forces = -torch.autograd.grad(
            output["energy"].sum(),
            positions,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return output["energy"], forces, output


__all__ = ["ZNNP", "build_radius_graph", "scatter_sum"]
