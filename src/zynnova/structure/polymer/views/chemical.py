from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.polymer import PolymerRecord
from ..io.json_codec import record_to_dict
from .common import GraphTensorView
from .features import atom_features, bond_features


@dataclass
class ChemicalStructureView:
    unit_graphs: dict[str, GraphTensorView]
    composition: np.ndarray
    unit_order: list[str]
    targets: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)
    record_payload: dict[str, object] | None = None


def to_chemical_structure_view(
    record: PolymerRecord,
    *,
    include_reconstruction: bool = True,
) -> ChemicalStructureView:
    """Encode only chemical unit structures and composition.

    This is the lightest polymer representation and is suitable for molecular
    GNNs, fingerprint models, and unit encoders. It deliberately omits explicit
    chain occurrences unless ``record_payload`` is retained.
    """
    record.validate()
    unit_graphs: dict[str, GraphTensorView] = {}
    unit_order = sorted(record.units)

    for unit_id in unit_order:
        unit = record.units[unit_id]
        graph = unit.graph
        x = np.stack([atom_features(atom) for atom in graph.atoms], axis=0)
        directed_edges: list[tuple[int, int]] = []
        attrs: list[np.ndarray] = []
        bond_orders: list[float] = []
        for bond in graph.bonds:
            feature = bond_features(bond)
            directed_edges.extend([(bond.source, bond.target), (bond.target, bond.source)])
            attrs.extend([feature, feature])
            bond_orders.extend([bond.order, bond.order])
        edge_index = (
            np.asarray(directed_edges, dtype=np.int64).T
            if directed_edges
            else np.empty((2, 0), dtype=np.int64)
        )
        edge_attr = (
            np.stack(attrs, axis=0)
            if attrs
            else np.empty((0, 3), dtype=np.float32)
        )
        view = GraphTensorView(
            node_features=x,
            edge_index=edge_index,
            edge_features=edge_attr,
            positions=graph.coordinates,
            node_ids=[f"{unit_id}:atom:{i}" for i in range(graph.num_atoms)],
            node_type_ids=np.asarray(
                [atom.atomic_number for atom in graph.atoms], dtype=np.int64
            ),
            edge_type_ids=np.asarray(
                [int(round(order * 10)) for order in bond_orders], dtype=np.int64
            ),
            metadata={
                "unit_id": unit_id,
                "unit_role": unit.role.value,
                "unit_name": unit.name,
                "ports": [
                    {
                        "id": port.id,
                        "atom_index": port.atom_index,
                        "port_type": port.port_type,
                        "direction": port.direction,
                        "valence": port.valence,
                        "leaving_atom_indices": list(port.leaving_atom_indices),
                        "allowed_partner_types": sorted(port.allowed_partner_types),
                        "features": dict(port.features),
                    }
                    for port in graph.ports
                ],
            },
        )
        view.validate()
        unit_graphs[unit_id] = view

    composition = np.asarray(
        [record.ensemble.composition.get(unit_id, 0.0) for unit_id in unit_order],
        dtype=np.float32,
    )
    targets = {
        name: np.asarray(value.value)
        for name, value in record.properties.items()
        if isinstance(value.value, (int, float, list))
    }
    return ChemicalStructureView(
        unit_graphs=unit_graphs,
        composition=composition,
        unit_order=unit_order,
        targets=targets,
        metadata={
            "record_id": record.id,
            "architecture_type": record.architecture.architecture_type.value,
        },
        record_payload=record_to_dict(record) if include_reconstruction else None,
    )
