from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..core.polymer import PolymerRecord
from ..io.json_codec import record_to_dict
from ..record_conversion import record2stru
from ..schema import RepresentationSchema
from .features import unit_scalar_features


@dataclass
class GenerativeTensorView:
    """Dense mixed discrete/continuous representation for graph generation.

    ``level='unit'`` generates polymer architecture graphs and ensemble
    statistics. ``level='atom'`` generates complete atom/bond graphs with 3-D
    coordinates, suitable for DiGress-like diffusion, discrete flow matching,
    or hybrid atom-coordinate flow models.
    """

    level: Literal["unit", "atom"]
    node_type: np.ndarray
    node_mask: np.ndarray
    node_features: np.ndarray
    edge_type: np.ndarray
    edge_mask: np.ndarray
    composition_logits: np.ndarray
    composition_mask: np.ndarray
    transition_logits: np.ndarray
    transition_mask: np.ndarray
    continuous_features: np.ndarray
    continuous_feature_mask: np.ndarray
    coordinates: np.ndarray | None = None
    coordinate_mask: np.ndarray | None = None
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_payload: dict[str, Any] | None = None

    def validate(self) -> None:
        self.node_type = np.asarray(self.node_type, dtype=np.int64)
        self.node_mask = np.asarray(self.node_mask, dtype=bool)
        self.node_features = np.asarray(self.node_features, dtype=np.float32)
        self.edge_type = np.asarray(self.edge_type, dtype=np.int64)
        self.edge_mask = np.asarray(self.edge_mask, dtype=bool)
        n = self.node_type.shape[0]
        if self.node_mask.shape != (n,):
            raise ValueError("node_mask shape mismatch")
        if self.node_features.shape[0] != n:
            raise ValueError("node feature count mismatch")
        if self.edge_type.shape != (n, n) or self.edge_mask.shape != (n, n):
            raise ValueError("edge tensors must have shape [N, N]")
        k = self.composition_logits.shape[0]
        if self.composition_mask.shape != (k,):
            raise ValueError("composition_mask shape mismatch")
        if self.transition_logits.shape != (k, k):
            raise ValueError("transition_logits must have shape [K, K]")
        if self.transition_mask.shape != (k, k):
            raise ValueError("transition_mask must have shape [K, K]")
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
            if self.coordinates.shape != (n, 3):
                raise ValueError("coordinates must have shape [N, 3]")
            if self.coordinate_mask is None or np.asarray(self.coordinate_mask).shape != (n,):
                raise ValueError("coordinate_mask required when coordinates are present")
            self.coordinate_mask = np.asarray(self.coordinate_mask, dtype=bool)

    def active_node_count(self) -> int:
        return int(self.node_mask.sum())


def _safe_log(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return np.log(np.clip(values, epsilon, None))


def _logit(value: float, epsilon: float = 1e-8) -> float:
    value = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(np.log(value / (1.0 - value)))


def _ensemble_tensors(
    record: PolymerRecord,
    schema: RepresentationSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unit_vocab = schema.unit_vocabulary
    pad_index = unit_vocab[schema.PAD_UNIT]
    unknown_index = unit_vocab[schema.UNK_UNIT]
    k = len(unit_vocab)
    composition = np.zeros(k, dtype=np.float64)
    composition_mask = np.ones(k, dtype=bool)
    composition_mask[pad_index] = False
    for unit_id, fraction in record.ensemble.composition.items():
        composition[unit_vocab.get(unit_id, unknown_index)] += fraction
    active = composition_mask
    if active.any():
        if composition[active].sum() <= 0:
            composition[active] = 1.0 / active.sum()
        else:
            composition[active] /= composition[active].sum()
    composition_logits = _safe_log(composition)
    composition_logits[~composition_mask] = -20.0

    transition = np.zeros((k, k), dtype=np.float64)
    transition_mask = np.outer(composition_mask, composition_mask)
    source_order = record.ensemble.transition_unit_order
    source_index = {unit: i for i, unit in enumerate(source_order)}
    if record.ensemble.transition_matrix is not None:
        for source_unit in record.units:
            source_global = unit_vocab.get(source_unit, unknown_index)
            if source_unit not in source_index:
                continue
            for target_unit in record.units:
                target_global = unit_vocab.get(target_unit, unknown_index)
                if target_unit in source_index:
                    transition[source_global, target_global] += record.ensemble.transition_matrix[
                        source_index[source_unit], source_index[target_unit]
                    ]
    for row_index in range(k):
        valid = transition_mask[row_index]
        if not valid.any():
            continue
        if transition[row_index, valid].sum() <= 0:
            transition[row_index, valid] = 1.0 / valid.sum()
        else:
            transition[row_index, valid] /= transition[row_index, valid].sum()
    transition_logits = _safe_log(transition)
    transition_logits[~transition_mask] = -20.0

    feature_names = schema.continuous_feature_names
    feature_index = {name: index for index, name in enumerate(feature_names)}
    continuous = np.zeros(len(feature_names), dtype=np.float32)
    continuous_mask = np.zeros(len(feature_names), dtype=bool)

    def set_feature(name: str, value: float) -> None:
        if name in feature_index:
            continuous[feature_index[name]] = value
            continuous_mask[feature_index[name]] = True

    dp = record.ensemble.degree_of_polymerization
    if dp is not None and (value := dp.representative_value()) is not None and value > 0:
        set_feature("log_dp", float(np.log(value)))
    mw = record.ensemble.molecular_weight
    if mw is not None:
        mn = mw.parameters.get("Mn", mw.parameters.get("number_average"))
        if mn is not None and mn > 0:
            set_feature("log_mn", float(np.log(mn)))
        dispersity = mw.parameters.get("dispersity")
        if dispersity is None:
            mw_value = mw.parameters.get("Mw", mw.parameters.get("weight_average"))
            if mn is not None and mw_value is not None and mn > 0:
                dispersity = mw_value / mn
        if dispersity is not None and dispersity >= 1:
            set_feature(
                "log_dispersity_minus_one",
                float(np.log(max(dispersity - 1.0, 1e-8))),
            )
    if record.ensemble.crosslink_density is not None:
        set_feature(
            "crosslink_density_log1p",
            float(np.log1p(record.ensemble.crosslink_density)),
        )
    if record.ensemble.tacticity:
        set_feature(
            "tacticity_logit", _logit(next(iter(record.ensemble.tacticity.values())))
        )
    return (
        composition_logits.astype(np.float32),
        composition_mask,
        transition_logits.astype(np.float32),
        transition_mask,
        continuous,
        continuous_mask,
    )


def _unit_level(
    record: PolymerRecord,
    schema: RepresentationSchema,
    *,
    max_nodes: int,
    state_id: str | None,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    nodes = sorted(
        record.architecture.nodes,
        key=lambda node: (
            node.occurrence is None,
            node.occurrence if node.occurrence is not None else 0,
            node.id,
        ),
    )
    if not nodes:
        raise ValueError("unit-level generative view requires architecture nodes")
    if len(nodes) > max_nodes:
        raise ValueError(f"record has {len(nodes)} unit nodes but max_nodes={max_nodes}")
    pad_index = schema.unit_vocabulary[schema.PAD_UNIT]
    node_type = np.full(max_nodes, pad_index, dtype=np.int64)
    node_mask = np.zeros(max_nodes, dtype=bool)
    node_features = np.zeros((max_nodes, 7), dtype=np.float32)
    node_index: dict[str, int] = {}
    for index, node in enumerate(nodes):
        node_index[node.id] = index
        node_type[index] = schema.unit_index(node.unit_id)
        node_mask[index] = True
        node_features[index] = unit_scalar_features(
            record.units[node.unit_id], record.ensemble.composition.get(node.unit_id, 0.0)
        )
    edge_type = np.full(
        (max_nodes, max_nodes), schema.edge_vocabulary[schema.NO_EDGE], dtype=np.int64
    )
    edge_mask = np.outer(node_mask, node_mask)
    np.fill_diagonal(edge_mask, False)
    for edge in record.architecture.edges:
        source, target = node_index[edge.source], node_index[edge.target]
        edge_id = schema.edge_vocabulary.get(edge.kind.value)
        if edge_id is None:
            raise ValueError(f"edge kind {edge.kind.value!r} is absent from schema")
        edge_type[source, target] = edge_id
        if not edge.directed:
            edge_type[target, source] = edge_id
    coordinates = None
    if state_id is not None:
        frame = record.get_state(state_id).frames[frame_index]
        mapping = {node_id: frame.coordinates[i] for i, node_id in enumerate(frame.node_ids)}
        if all(node.id in mapping for node in nodes):
            coordinates = np.zeros((max_nodes, 3), dtype=np.float32)
            coordinates[: len(nodes)] = np.asarray([mapping[node.id] for node in nodes])
        elif len(frame.coordinates) == len(nodes):
            coordinates = np.zeros((max_nodes, 3), dtype=np.float32)
            coordinates[: len(nodes)] = frame.coordinates
    return node_type, node_mask, node_features, edge_type, edge_mask, coordinates


def _atom_level(
    record: PolymerRecord,
    schema: RepresentationSchema,
    *,
    max_atoms: int,
    state_id: str | None,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = record2stru(record, state_id=state_id, frame_index=frame_index)
    if structure.num_atoms > max_atoms:
        raise ValueError(f"record has {structure.num_atoms} atoms but max_atoms={max_atoms}")
    pad_index = schema.atom_vocabulary[schema.PAD_ATOM]
    node_type = np.full(max_atoms, pad_index, dtype=np.int64)
    node_mask = np.zeros(max_atoms, dtype=bool)
    node_features = np.zeros((max_atoms, 4), dtype=np.float32)
    for index, atomic_number in enumerate(structure.atomic_numbers):
        node_type[index] = schema.atom_index(int(atomic_number))
        node_mask[index] = True
        node_features[index, 0] = atomic_number
        if structure.charges is not None:
            node_features[index, 1] = structure.charges[index]
        if structure.masses is not None:
            node_features[index, 2] = structure.masses[index]
        if structure.tags is not None:
            node_features[index, 3] = structure.tags[index]
    edge_type = np.full(
        (max_atoms, max_atoms), schema.bond_vocabulary[schema.NO_EDGE], dtype=np.int64
    )
    edge_mask = np.outer(node_mask, node_mask)
    np.fill_diagonal(edge_mask, False)
    if structure.bonds is not None:
        orders = structure.bond_orders if structure.bond_orders is not None else np.ones(len(structure.bonds))
        for (source, target), order in zip(structure.bonds, orders, strict=True):
            label = schema.bond_label(float(order))
            bond_id = schema.bond_vocabulary.get(label, schema.bond_vocabulary["other"])
            edge_type[int(source), int(target)] = bond_id
            edge_type[int(target), int(source)] = bond_id
    coordinates = np.zeros((max_atoms, 3), dtype=np.float32)
    coordinates[: structure.num_atoms] = structure.positions
    return node_type, node_mask, node_features, edge_type, edge_mask, coordinates


def to_generative_view(
    record: PolymerRecord,
    *,
    schema: RepresentationSchema,
    level: Literal["unit", "atom"] = "unit",
    max_nodes: int | None = None,
    state_id: str | None = None,
    frame_index: int = 0,
    include_reconstruction: bool = True,
) -> GenerativeTensorView:
    record.validate()
    ensemble = _ensemble_tensors(record, schema)
    if level == "unit":
        capacity = max_nodes or schema.max_nodes
        node_type, node_mask, node_features, edge_type, edge_mask, coordinates = _unit_level(
            record,
            schema,
            max_nodes=capacity,
            state_id=state_id,
            frame_index=frame_index,
        )
        vocabularies = {
            "node": schema.unit_vocabulary,
            "edge": schema.edge_vocabulary,
            "unit": schema.unit_vocabulary,
        }
    elif level == "atom":
        capacity = max_nodes or int(schema.max_atoms or schema.max_nodes)
        node_type, node_mask, node_features, edge_type, edge_mask, coordinates = _atom_level(
            record,
            schema,
            max_atoms=capacity,
            state_id=state_id,
            frame_index=frame_index,
        )
        vocabularies = {
            "node": schema.atom_vocabulary,
            "edge": schema.bond_vocabulary,
            "unit": schema.unit_vocabulary,
        }
    else:
        raise ValueError("level must be 'unit' or 'atom'")
    view = GenerativeTensorView(
        level=level,
        node_type=node_type,
        node_mask=node_mask,
        node_features=node_features,
        edge_type=edge_type,
        edge_mask=edge_mask,
        composition_logits=ensemble[0],
        composition_mask=ensemble[1],
        transition_logits=ensemble[2],
        transition_mask=ensemble[3],
        continuous_features=ensemble[4],
        continuous_feature_mask=ensemble[5],
        coordinates=coordinates,
        coordinate_mask=node_mask.copy() if coordinates is not None else None,
        vocabularies=vocabularies,
        feature_names=list(schema.continuous_feature_names),
        metadata={
            "record_id": record.id,
            "schema_id": schema.schema_id,
            "dataset_batchable": True,
            "state_id": state_id,
        },
        record_payload=record_to_dict(record) if include_reconstruction else None,
    )
    view.validate()
    return view


def generative_view_from_logits(
    template: GenerativeTensorView,
    *,
    node_logits: Any,
    edge_logits: Any,
    coordinates: Any | None = None,
    continuous_features: Any | None = None,
    composition_logits: Any | None = None,
    transition_logits: Any | None = None,
    symmetrize_edges: bool = True,
) -> GenerativeTensorView:
    """Convert neural-network logits into a decodable generative graph view.

    Expected shapes are ``node_logits[N, K_node]`` and
    ``edge_logits[N, N, K_edge]``. PyTorch tensors are accepted without making
    the core package depend on torch.
    """

    def numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    node_scores = numpy(node_logits)
    edge_scores = numpy(edge_logits)
    n = template.node_type.shape[0]
    if node_scores.ndim != 2 or node_scores.shape[0] != n:
        raise ValueError("node_logits must have shape [N, K_node]")
    if edge_scores.ndim != 3 or edge_scores.shape[:2] != (n, n):
        raise ValueError("edge_logits must have shape [N, N, K_edge]")
    if symmetrize_edges:
        edge_scores = 0.5 * (edge_scores + edge_scores.transpose(1, 0, 2))
    node_type = np.argmax(node_scores, axis=-1).astype(np.int64)
    edge_type = np.argmax(edge_scores, axis=-1).astype(np.int64)
    no_edge = template.vocabularies.get("edge", {}).get("none", 0)
    edge_type = np.where(template.edge_mask, edge_type, no_edge)
    np.fill_diagonal(edge_type, no_edge)
    result = GenerativeTensorView(
        level=template.level,
        node_type=node_type,
        node_mask=template.node_mask.copy(),
        node_features=template.node_features.copy(),
        edge_type=edge_type,
        edge_mask=template.edge_mask.copy(),
        composition_logits=(
            template.composition_logits.copy()
            if composition_logits is None
            else numpy(composition_logits).astype(np.float32)
        ),
        composition_mask=template.composition_mask.copy(),
        transition_logits=(
            template.transition_logits.copy()
            if transition_logits is None
            else numpy(transition_logits).astype(np.float32)
        ),
        transition_mask=template.transition_mask.copy(),
        continuous_features=(
            template.continuous_features.copy()
            if continuous_features is None
            else numpy(continuous_features).astype(np.float32)
        ),
        continuous_feature_mask=template.continuous_feature_mask.copy(),
        coordinates=(
            None
            if template.coordinates is None and coordinates is None
            else (
                template.coordinates.copy()
                if coordinates is None
                else numpy(coordinates).astype(np.float32)
            )
        ),
        coordinate_mask=(
            None if template.coordinate_mask is None else template.coordinate_mask.copy()
        ),
        vocabularies={key: dict(value) for key, value in template.vocabularies.items()},
        feature_names=list(template.feature_names),
        metadata={**template.metadata, "decoded_from_logits": True},
        record_payload=None,
    )
    result.validate()
    return result
