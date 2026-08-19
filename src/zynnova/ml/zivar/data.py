"""Backbone-neutral ASE graph bridge with electrochemical conditions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._deps import require_ase, require_torch
from .types import Conditions, ZIVARBatch

torch = require_torch()


def _parameter_device_dtype(model: Any) -> tuple[Any, Any]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _radius_graph(atoms: Any, cutoff_A: float) -> tuple[np.ndarray, ...]:
    from ase.neighborlist import neighbor_list

    sender, receiver, unit_shifts = neighbor_list(
        "ijS", atoms, cutoff_A, self_interaction=False
    )
    sender = np.asarray(sender, dtype=np.int64)
    receiver = np.asarray(receiver, dtype=np.int64)
    unit_shifts = np.asarray(unit_shifts, dtype=np.int64).reshape(-1, 3)
    if sender.size:
        order = np.lexsort(
            (
                unit_shifts[:, 2],
                unit_shifts[:, 1],
                unit_shifts[:, 0],
                receiver,
                sender,
            )
        )
        sender, receiver, unit_shifts = (
            sender[order],
            receiver[order],
            unit_shifts[order],
        )
    return sender, receiver, unit_shifts


def atoms_to_batch(atoms: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the common reciprocal directed graph used by every backbone."""

    require_ase()
    device, dtype = _parameter_device_dtype(model)
    numbers = np.asarray(atoms.numbers, dtype=np.int64)
    table = {int(number): index for index, number in enumerate(model.atomic_numbers)}
    missing = sorted(set(int(value) for value in numbers) - set(table))
    if missing:
        raise ValueError(f"model element table does not contain {missing}")
    indices = np.asarray([table[int(value)] for value in numbers], dtype=np.int64)
    node_attrs = np.zeros((len(atoms), len(table)), dtype=float)
    node_attrs[np.arange(len(atoms)), indices] = 1.0
    sender, receiver, unit_shifts = _radius_graph(atoms, float(model.cutoff_A))
    cell = np.asarray(atoms.cell, dtype=float).reshape(3, 3)
    shifts = unit_shifts @ cell
    payload: dict[str, Any] = {
        "positions": torch.as_tensor(
            np.asarray(atoms.positions), device=device, dtype=dtype
        ),
        "node_attrs": torch.as_tensor(node_attrs, device=device, dtype=dtype),
        "atomic_numbers": torch.as_tensor(numbers, device=device, dtype=torch.long),
        "edge_index": torch.as_tensor(
            np.stack((sender, receiver), axis=0), device=device, dtype=torch.long
        ),
        "shifts": torch.as_tensor(shifts, device=device, dtype=dtype),
        "unit_shifts": torch.as_tensor(
            unit_shifts, device=device, dtype=dtype
        ),
        "cell": torch.as_tensor(cell[None], device=device, dtype=dtype),
        "pbc": torch.as_tensor(
            np.asarray(atoms.pbc, dtype=bool)[None], device=device, dtype=torch.bool
        ),
        "batch": torch.zeros(len(atoms), device=device, dtype=torch.long),
        "ptr": torch.tensor((0, len(atoms)), device=device, dtype=torch.long),
        "head": torch.zeros(1, device=device, dtype=torch.long),
    }
    conditions = atoms_to_conditions(
        atoms,
        device=device,
        dtype=dtype,
    )
    return payload, conditions


def atoms_to_typed_batch(atoms: Any, model: Any) -> tuple[ZIVARBatch, Conditions]:
    """Return the authoritative typed contract used by new integrations."""

    payload, conditions = atoms_to_batch(atoms, model)
    return (
        ZIVARBatch(
            positions=payload["positions"],
            atomic_numbers=payload["atomic_numbers"],
            batch=payload["batch"],
            edge_index=payload["edge_index"],
            shifts=payload["shifts"],
            cell=payload["cell"],
            pbc=payload["pbc"],
            node_attrs=payload["node_attrs"],
            unit_shifts=payload["unit_shifts"],
            ptr=payload["ptr"],
            head=payload["head"],
        ),
        Conditions.from_mapping(conditions),
    )


def collate_zivar_batches(items: list[ZIVARBatch]) -> ZIVARBatch:
    """Collate graph objects without CPU round trips or label-dependent inputs."""

    if not items:
        raise ValueError("cannot collate an empty ZIVAR batch")
    reference = items[0].positions
    attr_width = None if items[0].node_attrs is None else items[0].node_attrs.shape[1]
    for item in items:
        if item.positions.device != reference.device or item.positions.dtype != reference.dtype:
            raise ValueError("all batches must share position device and dtype")
        if item.graph_count != 1:
            raise ValueError("collate_zivar_batches expects one graph per input item")
        width = None if item.node_attrs is None else item.node_attrs.shape[1]
        if width != attr_width:
            raise ValueError("node attribute widths differ")
        if (item.cell is None) != (items[0].cell is None) or (
            (item.pbc is None) != (items[0].pbc is None)
        ):
            raise ValueError("cell/PBC presence differs across inputs")
    positions = torch.cat([item.positions for item in items], dim=0)
    atomic_numbers = torch.cat([item.atomic_numbers for item in items], dim=0)
    atom_offset = 0
    edges = []
    batches = []
    pointers = [0]
    for graph, item in enumerate(items):
        edges.append(item.edge_index + atom_offset)
        batches.append(
            torch.full(
                (item.atom_count,),
                graph,
                device=item.batch.device,
                dtype=torch.long,
            )
        )
        atom_offset += item.atom_count
        pointers.append(atom_offset)
    return ZIVARBatch(
        positions=positions,
        atomic_numbers=atomic_numbers,
        batch=torch.cat(batches),
        edge_index=torch.cat(edges, dim=1),
        shifts=(
            None if items[0].shifts is None else torch.cat([item.shifts for item in items])
        ),
        cell=(None if items[0].cell is None else torch.cat([item.cell for item in items])),
        pbc=(None if items[0].pbc is None else torch.cat([item.pbc for item in items])),
        node_attrs=(
            None
            if items[0].node_attrs is None
            else torch.cat([item.node_attrs for item in items])
        ),
        unit_shifts=(
            None
            if items[0].unit_shifts is None
            else torch.cat([item.unit_shifts for item in items])
        ),
        ptr=torch.tensor(pointers, device=reference.device, dtype=torch.long),
        head=torch.cat(
            [
                item.head
                if item.head is not None
                else torch.zeros(1, device=reference.device, dtype=torch.long)
                for item in items
            ]
        ),
    )


def atoms_to_conditions(
    atoms: Any,
    *,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Map stable ASE arrays/info fields to the explicit boundary contract."""

    atom_count = len(atoms)
    initial_charge = np.asarray(atoms.get_initial_charges(), dtype=float)
    conditions: dict[str, Any] = {
        "total_charge": torch.as_tensor(
            [float(atoms.info.get("total_charge", initial_charge.sum()))],
            device=device,
            dtype=dtype,
        )
    }
    # ASE initial charges determine only the physical total-charge boundary.
    # They are never copied into per-atom model inputs or used as supervision
    # seeds, which prevents both label leakage and charge-path hysteresis.
    explicit_spin = "spin_vectors" in atoms.arrays or "initial_magmoms" in atoms.arrays
    moments = np.asarray(
        atoms.arrays.get("spin_vectors", atoms.get_initial_magnetic_moments()),
        dtype=float,
    )
    if moments.shape == (atom_count,):
        vector_moments = np.zeros((atom_count, 3), dtype=float)
        vector_moments[:, 2] = moments
        moments = vector_moments
    if moments.shape == (atom_count, 3) and (explicit_spin or np.any(moments)):
        conditions["spin_vectors"] = torch.as_tensor(
            moments, device=device, dtype=dtype
        )
    for source, destination, shape in (
        ("external_electric_field", "external_electric_field", (1, 3)),
        ("external_magnetic_field", "external_magnetic_field", (1, 3)),
        ("electric_field_origin", "electric_field_origin", (1, 3)),
        ("total_magnetization", "total_magnetization", (1, 3)),
    ):
        if source in atoms.info:
            value = torch.as_tensor(atoms.info[source], device=device, dtype=dtype).reshape(shape)
            conditions[destination] = value
    if "total_spin" in atoms.info:
        conditions["total_spin"] = torch.as_tensor(
            [float(atoms.info["total_spin"])], device=device, dtype=dtype
        )
    if "formal_total_charge" in atoms.info:
        conditions["formal_total_charge"] = torch.as_tensor(
            [float(atoms.info["formal_total_charge"])], device=device, dtype=dtype
        )
    if "electrode_potential" in atoms.arrays:
        value = np.asarray(atoms.arrays["electrode_potential"], dtype=float)
        conditions["electrode_potential"] = torch.as_tensor(
            value, device=device, dtype=dtype
        )
    elif "electrode_potential" in atoms.info:
        value = np.asarray(atoms.info["electrode_potential"], dtype=float)
        if value.ndim == 0:
            value = value.reshape(1)
        conditions["electrode_potential"] = torch.as_tensor(
            value, device=device, dtype=dtype
        )
    if "reservoir_mask" in atoms.arrays:
        value = np.asarray(atoms.arrays["reservoir_mask"], dtype=float)
        if value.shape != (atom_count,):
            raise ValueError("reservoir_mask must have one value per atom")
        conditions["reservoir_mask"] = torch.as_tensor(
            value, device=device, dtype=dtype
        )
    if "closed_region_charge" in atoms.info:
        value = np.asarray(atoms.info["closed_region_charge"], dtype=float).reshape(1)
        conditions["closed_region_charge"] = torch.as_tensor(
            value, device=device, dtype=dtype
        )
    if "fragment_id" in atoms.arrays:
        fragment_ids = np.asarray(atoms.arrays["fragment_id"])
        labels, inverse = np.unique(fragment_ids, return_inverse=True)
        membership = np.eye(len(labels), dtype=float)[inverse]
        conditions["fragment_membership"] = torch.as_tensor(
            membership, device=device, dtype=dtype
        )
        if "fragment_charge" in atoms.info:
            fragment_charge = np.asarray(atoms.info["fragment_charge"], dtype=float)
            if fragment_charge.shape != (len(labels),):
                raise ValueError("fragment_charge must match unique fragment_id values")
            conditions["fragment_charge"] = torch.as_tensor(
                fragment_charge, device=device, dtype=dtype
            )
    return conditions


__all__ = [
    "atoms_to_batch",
    "atoms_to_conditions",
    "atoms_to_typed_batch",
    "collate_zivar_batches",
]
