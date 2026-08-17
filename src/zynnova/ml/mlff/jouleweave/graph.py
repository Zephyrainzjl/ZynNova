from __future__ import annotations

import math
from typing import Any

from ...common import require_torch

torch = require_torch()


def scatter_sum(values: Any, index: Any, dim_size: int) -> Any:
    """Dependency-free sum reduction over the leading dimension."""
    output = values.new_zeros((dim_size, *values.shape[1:]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


def scatter_mean(values: Any, index: Any, dim_size: int) -> Any:
    output = scatter_sum(values, index, dim_size)
    count = scatter_sum(
        values.new_ones((values.shape[0],)),
        index,
        dim_size,
    ).clamp_min_(1.0)
    return output / count.reshape((dim_size,) + (1,) * (values.ndim - 1))


def smooth_cutoff(distance: Any, cutoff: float) -> Any:
    """Quintic C2 cutoff whose value and first two derivatives vanish at ``cutoff``."""

    x = (distance / float(cutoff)).clamp(0.0, 1.0)
    envelope = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return envelope * (distance < cutoff).to(distance.dtype)


def normalize_cell_pbc(
    cell: Any | None,
    pbc: Any | None,
    *,
    graph_count: int,
    positions: Any,
) -> tuple[Any, Any]:
    if cell is None:
        cell = positions.new_zeros((graph_count, 3, 3))
    elif not torch.is_tensor(cell):
        cell = torch.as_tensor(cell, device=positions.device, dtype=positions.dtype)
    else:
        cell = cell.to(device=positions.device, dtype=positions.dtype)
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    if cell.shape != (graph_count, 3, 3):
        raise ValueError(f"cell must have shape [graphs, 3, 3] or [3, 3], got {tuple(cell.shape)}")

    if pbc is None:
        pbc = torch.zeros((graph_count, 3), device=positions.device, dtype=torch.bool)
    elif not torch.is_tensor(pbc):
        pbc = torch.as_tensor(pbc, device=positions.device, dtype=torch.bool)
    else:
        pbc = pbc.to(device=positions.device, dtype=torch.bool)
    if pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    if pbc.shape != (graph_count, 3):
        raise ValueError(f"pbc must have shape [graphs, 3] or [3], got {tuple(pbc.shape)}")
    return cell, pbc


def _translation_grid(cell: Any, pbc: Any, cutoff: float) -> Any:
    if not bool(torch.any(pbc).item()):
        return torch.zeros((1, 3), device=cell.device, dtype=torch.long)
    determinant = torch.det(cell)
    if float(torch.abs(determinant).detach().cpu()) < 1.0e-12:
        raise ValueError("periodic neighbor construction requires a non-singular cell")

    # For row-vector cells, columns of inv(cell) are reciprocal plane normals.
    inverse = torch.linalg.inv(cell.detach())
    reciprocal_norm = torch.linalg.vector_norm(inverse, dim=0)
    repeats = []
    for axis in range(3):
        if bool(pbc[axis].item()):
            repeats.append(int(math.ceil(cutoff * float(reciprocal_norm[axis].cpu()))) + 1)
        else:
            repeats.append(0)
    axes = [
        torch.arange(-repeat, repeat + 1, device=cell.device, dtype=torch.long)
        for repeat in repeats
    ]
    grid = torch.cartesian_prod(*axes)
    if grid.ndim == 1:
        grid = grid.reshape(-1, 3)
    return grid


def build_periodic_radius_graph(
    positions: Any,
    batch: Any | None = None,
    cell: Any | None = None,
    pbc: Any | None = None,
    *,
    cutoff: float,
    max_neighbors: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build a differentiable directed radius graph with explicit periodic images.

    Topology selection is discrete, as in every neighbor-list implementation, but
    returned edge vectors remain connected to ``positions`` and ``cell`` for force
    and virial differentiation. Unlike a minimum-image-only implementation, this
    function retains multiple images when the cutoff exceeds half a cell height.

    Returns ``edge_index``, ``edge_vector`` (sender minus receiver),
    ``edge_distance``, and integer lattice ``edge_shift``.
    """

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError("max_neighbors must be positive or None")
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [atoms, 3]")
    atom_count = positions.shape[0]
    if batch is None:
        batch = torch.zeros(atom_count, device=positions.device, dtype=torch.long)
    else:
        batch = batch.to(device=positions.device, dtype=torch.long)
    if batch.shape != (atom_count,):
        raise ValueError("batch must have shape [atoms]")
    if batch.numel() and int(batch.min().item()) < 0:
        raise ValueError("batch indices cannot be negative")
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
    cell, pbc = normalize_cell_pbc(
        cell,
        pbc,
        graph_count=graph_count,
        positions=positions,
    )

    receivers: list[Any] = []
    senders: list[Any] = []
    vectors: list[Any] = []
    distances: list[Any] = []
    shifts: list[Any] = []

    for graph_index in range(graph_count):
        atom_indices = torch.nonzero(batch == graph_index, as_tuple=False).reshape(-1)
        if not atom_indices.numel():
            continue
        local = positions[atom_indices]
        count = local.shape[0]
        lattice_shift = _translation_grid(cell[graph_index], pbc[graph_index], cutoff)
        cart_shift = lattice_shift.to(positions.dtype) @ cell[graph_index]
        base = local[None, :, :] - local[:, None, :]
        delta = base.unsqueeze(0) + cart_shift[:, None, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        mask = distance < cutoff
        zero_shift = torch.all(lattice_shift == 0, dim=1)
        if bool(torch.any(zero_shift).item()):
            mask[zero_shift] &= ~torch.eye(
                count,
                device=positions.device,
                dtype=torch.bool,
            )
        triplets = torch.nonzero(mask, as_tuple=False)
        if not triplets.numel():
            continue
        shift_index, local_receiver, local_sender = triplets.unbind(dim=1)

        if max_neighbors is not None:
            selected: list[Any] = []
            selected_distance = distance[shift_index, local_receiver, local_sender]
            for receiver in range(count):
                candidates = torch.nonzero(
                    local_receiver == receiver,
                    as_tuple=False,
                ).reshape(-1)
                if candidates.numel() > max_neighbors:
                    order = torch.argsort(selected_distance[candidates])[:max_neighbors]
                    candidates = candidates[order]
                if candidates.numel():
                    selected.append(candidates)
            if not selected:
                continue
            keep = torch.cat(selected)
            shift_index = shift_index[keep]
            local_receiver = local_receiver[keep]
            local_sender = local_sender[keep]

        receivers.append(atom_indices[local_receiver])
        senders.append(atom_indices[local_sender])
        vectors.append(delta[shift_index, local_receiver, local_sender])
        distances.append(distance[shift_index, local_receiver, local_sender])
        shifts.append(lattice_shift[shift_index])

    if not receivers:
        empty_index = torch.empty(0, device=positions.device, dtype=torch.long)
        return (
            torch.stack((empty_index, empty_index), dim=0),
            positions.new_empty((0, 3)),
            positions.new_empty((0,)),
            torch.empty((0, 3), device=positions.device, dtype=torch.long),
        )
    receiver = torch.cat(receivers)
    sender = torch.cat(senders)
    return (
        torch.stack((receiver, sender), dim=0),
        torch.cat(vectors, dim=0),
        torch.cat(distances, dim=0),
        torch.cat(shifts, dim=0),
    )


def edge_vectors_from_shifts(
    positions: Any,
    edge_index: Any,
    *,
    edge_shift: Any | None = None,
    batch: Any | None = None,
    cell: Any | None = None,
) -> Any:
    receiver, sender = edge_index
    vector = positions[sender] - positions[receiver]
    if edge_shift is None:
        return vector
    if batch is None:
        batch = torch.zeros(
            positions.shape[0],
            device=positions.device,
            dtype=torch.long,
        )
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
    cell, _ = normalize_cell_pbc(
        cell,
        None,
        graph_count=graph_count,
        positions=positions,
    )
    edge_cell = cell[batch[receiver]]
    return vector + torch.einsum(
        "ei,eij->ej",
        edge_shift.to(dtype=positions.dtype),
        edge_cell,
    )


__all__ = [
    "build_periodic_radius_graph",
    "edge_vectors_from_shifts",
    "normalize_cell_pbc",
    "scatter_mean",
    "scatter_sum",
    "smooth_cutoff",
]
