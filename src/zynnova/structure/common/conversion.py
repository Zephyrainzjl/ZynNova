from __future__ import annotations

from typing import Any

import numpy as np

from .backends import BackendName, build_neighbor_graph
from .elements import covalent_radii
from .features import FeatureConfig, build_edge_features, build_node_features
from .types import GraphData, StructureData


def infer_bond_edges(
    structure: StructureData,
    *,
    directed: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    if structure.bonds is None:
        return None
    bonds = np.asarray(structure.bonds, dtype=np.int64)
    orders = np.asarray(structure.bond_orders, dtype=np.float64)
    if directed:
        edge_index = np.concatenate((bonds.T, bonds[:, ::-1].T), axis=1)
        bond_order = np.concatenate((orders, orders))
    else:
        edge_index = bonds.T.copy()
        bond_order = orders.copy()
    source, target = edge_index
    edge_vec = structure.positions[target] - structure.positions[source]
    edge_dist = np.linalg.norm(edge_vec, axis=1)
    edge_shift = np.zeros((edge_index.shape[1], 3), dtype=np.int64)
    return {
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_vec": edge_vec,
        "edge_dist": edge_dist,
    }, bond_order


def structure_to_graph(
    structure: StructureData,
    *,
    structure_kind: str,
    backend: BackendName = "auto",
    neighbor_mode: str = "cutoff",
    cutoff: float = 5.0,
    radius_scale: float = 1.2,
    max_neighbors: int | None = None,
    directed: bool = True,
    self_edges: bool = False,
    use_explicit_bonds: bool = False,
    feature_config: FeatureConfig | None = None,
    tolerance: float = 1.0e-8,
) -> GraphData:
    config = feature_config or FeatureConfig()
    bond_order = None
    explicit = infer_bond_edges(structure, directed=directed) if use_explicit_bonds else None
    if explicit is not None:
        edge_payload, bond_order = explicit
        selected_backend = "explicit-bonds"
    else:
        edge_payload, selected_backend = build_neighbor_graph(
            structure.positions,
            structure.cell,
            structure.pbc,
            covalent_radii(structure.atomic_numbers),
            backend=backend,
            mode=neighbor_mode,
            cutoff=cutoff,
            radius_scale=radius_scale,
            max_neighbors=max_neighbors,
            directed=directed,
            self_edges=self_edges,
            tolerance=tolerance,
        )

    node_features, node_names, node_attrs = build_node_features(structure, config)
    edge_features, edge_names, edge_attrs = build_edge_features(
        edge_payload["edge_vec"],
        edge_payload["edge_dist"],
        edge_payload["edge_shift"],
        bond_order=bond_order,
        config=config,
    )
    for key, value in structure.arrays.items():
        if np.asarray(value).shape[:1] == (structure.num_atoms,):
            node_attrs.setdefault(key, np.asarray(value))

    frac_pos = structure.fractional_positions if np.any(structure.pbc) else None
    return GraphData(
        atomic_numbers=structure.atomic_numbers,
        positions=structure.positions,
        edge_index=edge_payload["edge_index"],
        edge_shift=edge_payload["edge_shift"],
        edge_vec=edge_payload["edge_vec"],
        edge_dist=edge_payload["edge_dist"],
        node_features=node_features,
        edge_features=edge_features,
        cell=structure.cell,
        pbc=structure.pbc,
        fractional_positions=frac_pos,
        charges=structure.charges,
        masses=structure.masses,
        tags=structure.tags,
        node_feature_names=node_names,
        edge_feature_names=edge_names,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        graph_attrs={
            "structure_kind": structure_kind,
            "neighbor_mode": "explicit-bonds" if explicit is not None else neighbor_mode,
            "cutoff": cutoff,
            "radius_scale": radius_scale,
            "max_neighbors": max_neighbors,
            "directed": directed,
            "self_edges": self_edges,
            "backend": selected_backend,
            "source": structure.source,
            "structure_info": dict(structure.info),
            "length_unit": "angstrom",
        },
    )


def coerce_graph(graph: Any) -> GraphData:
    if isinstance(graph, GraphData):
        return graph
    if hasattr(graph, "edge_index") and hasattr(graph, "pos") and hasattr(graph, "z"):
        return GraphData.from_pyg(graph)
    if isinstance(graph, dict):
        return GraphData(**graph)
    raise TypeError("graph must be GraphData, PyTorch Geometric Data, or a GraphData mapping")
