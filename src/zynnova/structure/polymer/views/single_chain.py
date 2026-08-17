from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.enums import EdgeKind, Resolution
from ..core.polymer import PolymerRecord
from ..io.json_codec import record_to_dict
from ..schema import RepresentationSchema
from .common import GraphTensorView
from .features import graph_level_features, stable_vocabulary, unit_scalar_features


@dataclass
class SingleChainView(GraphTensorView):
    """Occurrence-level graph for one polymer molecule/chain.

    Nodes are repeat-unit occurrences by default. Linear, ring, graft, comb,
    star, and branched single molecules are all represented by the same sparse
    graph; ``chain_order`` is populated when a linear ordering is available.
    """

    chain_order: np.ndarray | None = None
    backbone_mask: np.ndarray | None = None
    unit_order_vocabulary: dict[str, int] | None = None


def _ordered_nodes(record: PolymerRecord):
    return sorted(
        record.architecture.nodes,
        key=lambda node: (
            node.occurrence is None,
            node.occurrence if node.occurrence is not None else 0,
            node.id,
        ),
    )


def _positions_for_nodes(
    record: PolymerRecord,
    node_ids: list[str],
    *,
    state_id: str | None,
    frame_index: int,
) -> np.ndarray | None:
    if state_id is None:
        return None
    frame = record.get_state(state_id).frames[frame_index]
    if frame.resolution in {Resolution.REPEAT_UNIT, Resolution.COARSE_GRAINED}:
        position_map = {
            node_id: frame.coordinates[index]
            for index, node_id in enumerate(frame.node_ids)
        }
        if all(node_id in position_map for node_id in node_ids):
            return np.asarray([position_map[node_id] for node_id in node_ids], dtype=float)
        if len(frame.node_ids) == len(node_ids):
            return frame.coordinates.copy()
    atom_to_occurrence = frame.metadata.get("atom_to_occurrence")
    if frame.resolution is Resolution.ATOMISTIC and atom_to_occurrence is not None:
        labels = np.asarray(atom_to_occurrence)
        centers = []
        for node_id in node_ids:
            mask = labels.astype(str) == str(node_id)
            if not mask.any():
                raise ValueError(f"atomistic frame has no atoms for occurrence {node_id!r}")
            centers.append(frame.coordinates[mask].mean(axis=0))
        return np.asarray(centers, dtype=float)
    raise ValueError(
        "selected spatial frame cannot be aligned to architecture occurrences; "
        "use repeat-unit coordinates or provide frame.metadata['atom_to_occurrence']"
    )


def to_single_chain_view(
    record: PolymerRecord,
    *,
    state_id: str | None = None,
    frame_index: int = 0,
    schema: RepresentationSchema | None = None,
    include_reconstruction: bool = True,
) -> SingleChainView:
    record.validate()
    nodes = _ordered_nodes(record)
    if not nodes:
        # A chemistry-only record is represented as one occurrence per known unit.
        nodes = [
            type("NodeProxy", (), {"id": f"unit:{index}", "unit_id": unit_id})()
            for index, unit_id in enumerate(sorted(record.units))
        ]
    vocabulary = schema.unit_vocabulary if schema is not None else stable_vocabulary(record.units)
    x = np.stack(
        [
            unit_scalar_features(
                record.units[node.unit_id],
                record.ensemble.composition.get(node.unit_id, 0.0),
            )
            for node in nodes
        ],
        axis=0,
    )
    node_index = {node.id: index for index, node in enumerate(nodes)}
    directed_edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    edge_type_ids: list[int] = []
    edge_vocab = (
        schema.edge_vocabulary
        if schema is not None
        else {
            value: index
            for index, value in enumerate(
                sorted({edge.kind.value for edge in record.architecture.edges} | {"none"})
            )
        }
    )
    for edge in record.architecture.edges:
        if edge.source not in node_index or edge.target not in node_index:
            continue
        source, target = node_index[edge.source], node_index[edge.target]
        edge_id = edge_vocab.get(edge.kind.value, edge_vocab.get("none", 0))
        directed_edges.append((source, target))
        edge_features.append([edge.bond_order, edge.probability, float(edge.directed)])
        edge_type_ids.append(edge_id)
        if not edge.directed:
            directed_edges.append((target, source))
            edge_features.append([edge.bond_order, edge.probability, float(edge.directed)])
            edge_type_ids.append(edge_id)
    if not directed_edges and len(nodes) > 1:
        # For a sequence-only record, create canonical backbone edges.
        for index in range(len(nodes) - 1):
            directed_edges.extend([(index, index + 1), (index + 1, index)])
            edge_features.extend([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            edge_id = edge_vocab.get(EdgeKind.BACKBONE.value, edge_vocab.get("none", 0))
            edge_type_ids.extend([edge_id, edge_id])
    edge_index = (
        np.asarray(directed_edges, dtype=np.int64).T
        if directed_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    graph_features, feature_names = graph_level_features(record)
    targets = {
        name: np.asarray(value.value)
        for name, value in record.properties.items()
        if isinstance(value.value, (int, float, list))
    }
    node_ids = [node.id for node in nodes]
    sequence = [node.unit_id for node in nodes]
    view = SingleChainView(
        node_features=x,
        edge_index=edge_index,
        edge_features=np.asarray(edge_features, dtype=np.float32).reshape(-1, 3),
        positions=_positions_for_nodes(
            record, node_ids, state_id=state_id, frame_index=frame_index
        ),
        node_ids=node_ids,
        node_type_ids=np.asarray(
            [
                schema.unit_index(item) if schema is not None else vocabulary[item]
                for item in sequence
            ],
            dtype=np.int64,
        ),
        edge_type_ids=np.asarray(edge_type_ids, dtype=np.int64),
        graph_features=graph_features,
        targets=targets,
        metadata={
            "record_id": record.id,
            "unit_sequence": sequence,
            "graph_feature_names": feature_names,
            "resolution": "repeat_unit",
            "architecture_type": record.architecture.architecture_type.value,
            "edge_vocabulary": edge_vocab,
            "state_id": state_id,
        },
        record_payload=record_to_dict(record) if include_reconstruction else None,
        chain_order=np.arange(len(sequence), dtype=np.int64),
        backbone_mask=np.ones(len(sequence), dtype=bool),
        unit_order_vocabulary=vocabulary,
    )
    view.validate()
    return view
