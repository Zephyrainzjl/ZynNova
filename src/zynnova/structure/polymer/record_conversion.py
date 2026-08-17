from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..common.io import load_structure, write_structure
from ..common.types import StructureData
from .core import (
    ArchitectureEdge,
    ArchitectureNode,
    ArchitectureType,
    ConnectionPort,
    EdgeKind,
    EnsembleStatistics,
    MolecularGraph,
    PeriodicBox,
    PolymerArchitecture,
    PolymerRecord,
    PolymerUnit,
    PropertyValue,
    Resolution,
    SpatialFrame,
    SpatialState,
    UnitRole,
)


def _as_unit(
    unit_id: str,
    value: PolymerUnit | MolecularGraph | StructureData,
    *,
    role: UnitRole = UnitRole.REPEAT,
    ports: Sequence[ConnectionPort] | None = None,
) -> PolymerUnit:
    if isinstance(value, PolymerUnit):
        return value
    if isinstance(value, MolecularGraph):
        graph = value
    elif isinstance(value, StructureData):
        graph = MolecularGraph.from_structure(value, ports=list(ports or []))
    else:
        raise TypeError(
            "unit_library values must be PolymerUnit, MolecularGraph, or StructureData"
        )
    if ports is not None:
        graph.ports = list(ports)
    return PolymerUnit(id=unit_id, role=role, graph=graph)


def _atom_labels_from_counts(
    num_atoms: int,
    unit_atom_counts: Sequence[int],
) -> np.ndarray:
    counts = np.asarray(unit_atom_counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError("unit_atom_counts must be a one-dimensional sequence of positive integers")
    if int(counts.sum()) != num_atoms:
        raise ValueError("sum(unit_atom_counts) must equal the number of atoms")
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)


def _normalize_occurrences(
    num_atoms: int,
    *,
    atom_to_unit: Sequence[int | str] | np.ndarray | None,
    unit_atom_counts: Sequence[int] | None,
) -> tuple[np.ndarray, list[str]]:
    if atom_to_unit is None:
        if unit_atom_counts is None:
            labels = np.zeros(num_atoms, dtype=np.int64)
        else:
            labels = _atom_labels_from_counts(num_atoms, unit_atom_counts)
    else:
        labels = np.asarray(atom_to_unit)
        if labels.shape != (num_atoms,):
            raise ValueError("atom_to_unit must have shape [num_atoms]")
    occurrence_values: list[str] = []
    seen: set[str] = set()
    normalized = np.empty(num_atoms, dtype=np.int64)
    for atom_index, raw in enumerate(labels.tolist()):
        key = str(raw)
        if key not in seen:
            seen.add(key)
            occurrence_values.append(key)
        normalized[atom_index] = occurrence_values.index(key)
    return normalized, occurrence_values


def stru2record(
    structure: Any,
    *,
    format: str | None = None,
    index: int | str = -1,
    record_id: str | None = None,
    atom_to_unit: Sequence[int | str] | np.ndarray | None = None,
    unit_atom_counts: Sequence[int] | None = None,
    unit_sequence: Sequence[str] | None = None,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    unit_roles: Mapping[str, UnitRole | str] | None = None,
    unit_ports: Mapping[str, Sequence[ConnectionPort]] | None = None,
    architecture_type: ArchitectureType | str = ArchitectureType.LINEAR,
    architecture_edges: Sequence[ArchitectureEdge | tuple[int, int] | tuple[str, str]] | None = None,
    state_id: str = "structure",
    properties: Mapping[str, PropertyValue | float | int | list[float]] | None = None,
    ensemble: EnsembleStatistics | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PolymerRecord:
    """Convert a real atomistic structure into the canonical PolymerRecord.

    The conversion is exact: atom ordering, coordinates, bonds, cell, PBC,
    charges, masses, tags, arrays, and source metadata are retained. When repeat
    unit boundaries are known, supply ``atom_to_unit`` or ``unit_atom_counts``;
    otherwise the whole molecule becomes one chemical unit.
    """
    data = load_structure(structure, format=format, index=index)
    occurrence_index, occurrence_labels = _normalize_occurrences(
        data.num_atoms,
        atom_to_unit=atom_to_unit,
        unit_atom_counts=unit_atom_counts,
    )
    num_occurrences = len(occurrence_labels)
    if unit_sequence is None:
        unit_sequence = ["U0"] * num_occurrences
    else:
        unit_sequence = list(unit_sequence)
        if len(unit_sequence) != num_occurrences:
            raise ValueError("unit_sequence length must equal the number of unit occurrences")

    roles = {key: UnitRole(value) for key, value in (unit_roles or {}).items()}
    units: dict[str, PolymerUnit] = {}
    if unit_library is not None:
        for unit_id, value in unit_library.items():
            units[unit_id] = _as_unit(
                unit_id,
                value,
                role=roles.get(unit_id, UnitRole.REPEAT),
                ports=(unit_ports or {}).get(unit_id),
            )
    for unit_id in dict.fromkeys(unit_sequence):
        if unit_id in units:
            continue
        occurrence = unit_sequence.index(unit_id)
        atom_indices = np.flatnonzero(occurrence_index == occurrence)
        units[unit_id] = PolymerUnit(
            id=unit_id,
            role=roles.get(unit_id, UnitRole.REPEAT),
            graph=MolecularGraph.from_structure(
                data,
                atom_indices=atom_indices,
                ports=list((unit_ports or {}).get(unit_id, [])),
                metadata={"derived_from_occurrence": occurrence},
            ),
        )

    nodes = [
        ArchitectureNode(id=f"u{occurrence}", unit_id=unit_id, occurrence=occurrence)
        for occurrence, unit_id in enumerate(unit_sequence)
    ]
    edges: list[ArchitectureEdge] = []
    if architecture_edges is None:
        for occurrence in range(num_occurrences - 1):
            edges.append(
                ArchitectureEdge(
                    source=f"u{occurrence}",
                    target=f"u{occurrence + 1}",
                    kind=EdgeKind.BACKBONE,
                    directed=False,
                )
            )
    else:
        for item in architecture_edges:
            if isinstance(item, ArchitectureEdge):
                edges.append(item)
            else:
                source, target = item
                source_id = f"u{source}" if isinstance(source, int) else str(source)
                target_id = f"u{target}" if isinstance(target, int) else str(target)
                edges.append(
                    ArchitectureEdge(
                        source=source_id,
                        target=target_id,
                        kind=EdgeKind.BACKBONE,
                    )
                )

    counts = Counter(unit_sequence)
    composition = {unit_id: count / num_occurrences for unit_id, count in counts.items()}
    if ensemble is None:
        ensemble = EnsembleStatistics(
            composition=composition,
            number_of_chains=1,
        )
    elif not ensemble.composition:
        ensemble.composition = composition

    property_values: dict[str, PropertyValue] = {}
    for name, value in (properties or {}).items():
        property_values[name] = (
            value if isinstance(value, PropertyValue) else PropertyValue(name=name, value=value)
        )

    box = None
    if np.any(data.pbc) and abs(float(np.linalg.det(data.cell))) > 1e-14:
        box = PeriodicBox(
            matrix=data.cell.copy(),
            periodic=tuple(bool(item) for item in data.pbc),
        )
    atom_to_occurrence_ids = [f"u{int(value)}" for value in occurrence_index]
    frame = SpatialFrame(
        resolution=Resolution.ATOMISTIC,
        node_ids=[f"atom:{index}" for index in range(data.num_atoms)],
        coordinates=data.positions.copy(),
        box=box,
        metadata={
            "atom_to_occurrence": atom_to_occurrence_ids,
            "atomic_numbers": data.atomic_numbers.tolist(),
            "bonds": None if data.bonds is None else data.bonds.tolist(),
            "bond_orders": None
            if data.bond_orders is None
            else data.bond_orders.tolist(),
            "charges": None if data.charges is None else data.charges.tolist(),
            "masses": None if data.masses is None else data.masses.tolist(),
            "tags": None if data.tags is None else data.tags.tolist(),
            "arrays": {key: np.asarray(value).tolist() for key, value in data.arrays.items()},
            "structure_info": dict(data.info),
            "source": data.source,
        },
    )
    record = PolymerRecord(
        id=record_id or str(data.info.get("name") or data.source or "polymer"),
        units=units,
        architecture=PolymerArchitecture(
            architecture_type=ArchitectureType(architecture_type),
            nodes=nodes,
            edges=edges,
            sequence=list(unit_sequence),
            head_node=nodes[0].id if nodes else None,
            tail_node=nodes[-1].id if nodes else None,
            metadata={"occurrence_labels": occurrence_labels},
        ),
        ensemble=ensemble,
        spatial_states=[SpatialState(id=state_id, frames=[frame])],
        properties=property_values,
        metadata=dict(metadata or {}),
    )
    record.validate()
    return record


def _atomistic_payload(
    record: PolymerRecord,
    *,
    state_id: str | None,
    frame_index: int,
) -> tuple[SpatialFrame, dict[str, Any]] | None:
    states = record.spatial_states
    if state_id is not None:
        states = [record.get_state(state_id)]
    for state in states:
        if not state.frames:
            continue
        frame = state.frames[frame_index]
        if frame.resolution is Resolution.ATOMISTIC and "atomic_numbers" in frame.metadata:
            return frame, frame.metadata
    return None


def _expand_units(
    record: PolymerRecord,
    *,
    state_id: str | None,
    frame_index: int,
    spacing: float,
) -> StructureData:
    nodes = sorted(
        record.architecture.nodes,
        key=lambda node: (
            node.occurrence is None,
            node.occurrence if node.occurrence is not None else 0,
            node.id,
        ),
    )
    if not nodes:
        nodes = [ArchitectureNode(id="u0", unit_id=next(iter(record.units)), occurrence=0)]

    centers = np.zeros((len(nodes), 3), dtype=float)
    if state_id is not None:
        frame = record.get_state(state_id).frames[frame_index]
        if frame.resolution in {Resolution.REPEAT_UNIT, Resolution.COARSE_GRAINED}:
            mapping = {node_id: frame.coordinates[i] for i, node_id in enumerate(frame.node_ids)}
            if all(node.id in mapping for node in nodes):
                centers = np.asarray([mapping[node.id] for node in nodes], dtype=float)
            elif len(frame.coordinates) == len(nodes):
                centers = frame.coordinates.copy()
        else:
            centers[:, 0] = np.arange(len(nodes), dtype=float) * spacing
    else:
        centers[:, 0] = np.arange(len(nodes), dtype=float) * spacing

    atomic_numbers: list[int] = []
    positions: list[np.ndarray] = []
    charges: list[float] = []
    masses: list[float] = []
    bonds: list[tuple[int, int]] = []
    bond_orders: list[float] = []
    occurrence_offsets: dict[str, int] = {}
    for occurrence, node in enumerate(nodes):
        graph = record.units[node.unit_id].graph
        local = (
            graph.coordinates.copy()
            if graph.coordinates is not None
            else np.zeros((graph.num_atoms, 3), dtype=float)
        )
        local -= local.mean(axis=0, keepdims=True)
        offset = len(atomic_numbers)
        occurrence_offsets[node.id] = offset
        atomic_numbers.extend(atom.atomic_number for atom in graph.atoms)
        charges.extend(
            np.nan if atom.partial_charge is None else atom.partial_charge for atom in graph.atoms
        )
        masses.extend(np.nan if atom.mass is None else atom.mass for atom in graph.atoms)
        positions.extend(local + centers[occurrence])
        for bond in graph.bonds:
            bonds.append((offset + bond.source, offset + bond.target))
            bond_orders.append(bond.order)

    node_map = {node.id: node for node in nodes}
    for edge in record.architecture.edges:
        if edge.source not in occurrence_offsets or edge.target not in occurrence_offsets:
            continue
        source_unit = record.units[node_map[edge.source].unit_id]
        target_unit = record.units[node_map[edge.target].unit_id]
        source_atom = (
            source_unit.graph.port(edge.source_port).atom_index
            if edge.source_port is not None
            else source_unit.graph.num_atoms - 1
        )
        target_atom = (
            target_unit.graph.port(edge.target_port).atom_index
            if edge.target_port is not None
            else 0
        )
        pair = (
            occurrence_offsets[edge.source] + source_atom,
            occurrence_offsets[edge.target] + target_atom,
        )
        if pair[0] != pair[1] and pair not in bonds and pair[::-1] not in bonds:
            bonds.append(pair)
            bond_orders.append(edge.bond_order)

    return StructureData(
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        positions=np.asarray(positions, dtype=float),
        charges=None
        if np.isnan(np.asarray(charges, dtype=float)).all()
        else np.nan_to_num(np.asarray(charges, dtype=float)),
        masses=None
        if np.isnan(np.asarray(masses, dtype=float)).all()
        else np.nan_to_num(np.asarray(masses, dtype=float)),
        bonds=np.asarray(bonds, dtype=np.int64) if bonds else None,
        bond_orders=np.asarray(bond_orders, dtype=float) if bonds else None,
        info={"record_id": record.id, "generated_from": "PolymerRecord"},
    )


def record2stru(
    record: PolymerRecord,
    *,
    state_id: str | None = None,
    frame_index: int = 0,
    output: Literal["structure", "ase"] = "structure",
    path: str | Path | None = None,
    format: str | None = None,
    spacing: float = 3.0,
) -> StructureData | Any:
    """Decode a PolymerRecord into a real atomistic structure.

    Source-derived records round-trip exactly through their atomistic payload.
    Generated unit-level records are expanded from unit templates and ports,
    then can be relaxed by a downstream geometry or MD backend.
    """
    record.validate()
    payload = _atomistic_payload(record, state_id=state_id, frame_index=frame_index)
    if payload is not None:
        frame, meta = payload
        box = frame.box
        structure = StructureData(
            atomic_numbers=meta["atomic_numbers"],
            positions=frame.coordinates,
            cell=np.zeros((3, 3), dtype=float) if box is None else box.matrix,
            pbc=np.zeros(3, dtype=bool) if box is None else np.asarray(box.periodic),
            charges=meta.get("charges"),
            masses=meta.get("masses"),
            tags=meta.get("tags"),
            bonds=meta.get("bonds"),
            bond_orders=meta.get("bond_orders"),
            arrays={key: np.asarray(value) for key, value in meta.get("arrays", {}).items()},
            info=dict(meta.get("structure_info", {})),
            source=meta.get("source"),
        )
    else:
        structure = _expand_units(
            record,
            state_id=state_id,
            frame_index=frame_index,
            spacing=spacing,
        )
    if path is not None:
        write_structure(structure, path, format=format)
    if output == "ase":
        return structure.to_ase()
    if output != "structure":
        raise ValueError("output must be 'structure' or 'ase'")
    return structure


# Naming aliases consistent with crystal/molecular modules.
stru2simple = stru2record
simple2stru = record2stru
