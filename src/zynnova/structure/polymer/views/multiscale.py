from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.polymer import PolymerRecord
from ..io.json_codec import record_to_dict
from .chemical import ChemicalStructureView, to_chemical_structure_view
from .features import graph_level_features, stable_vocabulary, unit_scalar_features


@dataclass
class RelationTable:
    source_type: str
    relation: str
    target_type: str
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None


@dataclass
class MultiScaleView:
    node_features: dict[str, np.ndarray]
    node_ids: dict[str, list[str]]
    relations: list[RelationTable]
    local_unit_graphs: ChemicalStructureView
    graph_features: np.ndarray
    graph_feature_names: list[str]
    spatial: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_payload: dict[str, Any] | None = None


def to_multiscale_view(
    record: PolymerRecord,
    *,
    state_id: str | None = None,
    frame_index: int = 0,
    include_reconstruction: bool = True,
) -> MultiScaleView:
    record.validate()
    local = to_chemical_structure_view(record, include_reconstruction=include_reconstruction)
    unit_vocab = stable_vocabulary(record.units)
    architecture_nodes = record.architecture.nodes

    unit_node_features = np.stack(
        [
            unit_scalar_features(
                record.units[node.unit_id],
                record.ensemble.composition.get(node.unit_id, 0.0),
            )
            for node in architecture_nodes
        ],
        axis=0,
    ) if architecture_nodes else np.empty((0, 7), dtype=np.float32)

    architecture_id_to_index = {
        node.id: index for index, node in enumerate(architecture_nodes)
    }
    directed_edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for edge in record.architecture.edges:
        source = architecture_id_to_index[edge.source]
        target = architecture_id_to_index[edge.target]
        directed_edges.append((source, target))
        edge_features.append([edge.bond_order, edge.probability, float(edge.directed)])
        if not edge.directed:
            directed_edges.append((target, source))
            edge_features.append([edge.bond_order, edge.probability, float(edge.directed)])

    relations: list[RelationTable] = [
        RelationTable(
            source_type="unit",
            relation="connects",
            target_type="unit",
            edge_index=(
                np.asarray(directed_edges, dtype=np.int64).T
                if directed_edges
                else np.empty((2, 0), dtype=np.int64)
            ),
            edge_features=(
                np.asarray(edge_features, dtype=np.float32)
                if edge_features
                else np.empty((0, 3), dtype=np.float32)
            ),
        )
    ]

    # Each architecture occurrence belongs to one chemical unit type.
    membership_edges = np.asarray(
        [
            list(range(len(architecture_nodes))),
            [unit_vocab[node.unit_id] for node in architecture_nodes],
        ],
        dtype=np.int64,
    ) if architecture_nodes else np.empty((2, 0), dtype=np.int64)
    relations.append(
        RelationTable(
            source_type="unit",
            relation="instance_of",
            target_type="unit_type",
            edge_index=membership_edges,
        )
    )

    type_features = np.stack(
        [
            unit_scalar_features(
                record.units[unit_id], record.ensemble.composition.get(unit_id, 0.0)
            )
            for unit_id in sorted(unit_vocab, key=unit_vocab.get)
        ],
        axis=0,
    )

    spatial: dict[str, Any] = {}
    if state_id is not None:
        state = record.get_state(state_id)
        frame = state.frames[frame_index]
        spatial = {
            "state_id": state_id,
            "coordinates": frame.coordinates,
            "spatial_edge_index": frame.spatial_edge_index,
            "periodic_edge_shift": frame.periodic_edge_shift,
            "box": frame.box.matrix if frame.box is not None else None,
            "resolution": frame.resolution.value,
            "node_ids": list(frame.node_ids),
            "metadata": dict(frame.metadata),
        }

    graph_features, graph_feature_names = graph_level_features(record)
    targets = {
        name: np.asarray(value.value)
        for name, value in record.properties.items()
        if isinstance(value.value, (int, float, list))
    }
    return MultiScaleView(
        node_features={"unit": unit_node_features, "unit_type": type_features},
        node_ids={
            "unit": [node.id for node in architecture_nodes],
            "unit_type": sorted(unit_vocab, key=unit_vocab.get),
        },
        relations=relations,
        local_unit_graphs=local,
        graph_features=graph_features,
        graph_feature_names=graph_feature_names,
        spatial=spatial,
        targets=targets,
        metadata={"record_id": record.id, "architecture_type": record.architecture.architecture_type.value},
        record_payload=record_to_dict(record) if include_reconstruction else None,
    )
