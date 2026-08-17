from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..common.types import StructureData
from .core import (
    ArchitectureEdge,
    ArchitectureNode,
    ArchitectureType,
    Atom,
    Bond,
    ConnectionPort,
    Distribution,
    DistributionKind,
    EdgeKind,
    EnsembleStatistics,
    MolecularGraph,
    PolymerArchitecture,
    PolymerRecord,
    PolymerUnit,
    Resolution,
    SpatialFrame,
    SpatialState,
    UnitRole,
)
from .io.json_codec import record_from_dict
from .record_conversion import record2stru
from .schema import RepresentationSchema
from .views import (
    ChemicalStructureView,
    GenerativeTensorView,
    MultiScaleView,
    SingleChainView,
    TransformerInputView,
)


def _softmax(values: np.ndarray, mask: np.ndarray | None = None, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if mask is not None:
        array = np.where(mask, array, -1.0e30)
    array = array - np.max(array, axis=axis, keepdims=True)
    result = np.exp(array)
    if mask is not None:
        result = np.where(mask, result, 0.0)
    denominator = result.sum(axis=axis, keepdims=True)
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator > 0)


def _unit_copy(value: PolymerUnit | MolecularGraph | StructureData, unit_id: str) -> PolymerUnit:
    if isinstance(value, PolymerUnit):
        return value
    if isinstance(value, MolecularGraph):
        return PolymerUnit(unit_id, UnitRole.REPEAT, value)
    if isinstance(value, StructureData):
        return PolymerUnit(unit_id, UnitRole.REPEAT, MolecularGraph.from_structure(value))
    raise TypeError("unit library values must be PolymerUnit, MolecularGraph, or StructureData")


def _ports_from_metadata(items: list[dict[str, Any]]) -> list[ConnectionPort]:
    return [
        ConnectionPort(
            id=item["id"],
            atom_index=int(item["atom_index"]),
            port_type=item.get("port_type", "generic"),
            direction=item.get("direction"),
            valence=int(item.get("valence", 1)),
            leaving_atom_indices=list(item.get("leaving_atom_indices", [])),
            allowed_partner_types=set(item.get("allowed_partner_types", [])),
            features=dict(item.get("features", {})),
        )
        for item in items
    ]


def chemical2record(view: ChemicalStructureView, *, record_id: str | None = None) -> PolymerRecord:
    if view.record_payload is not None:
        return record_from_dict(view.record_payload)
    units: dict[str, PolymerUnit] = {}
    for unit_id in view.unit_order:
        graph = view.unit_graphs[unit_id]
        graph.validate()
        atomic_numbers = (
            graph.node_type_ids
            if graph.node_type_ids is not None
            else np.rint(graph.node_features[:, 0]).astype(np.int64)
        )
        atoms = [Atom(int(number)) for number in atomic_numbers]
        bonds: list[Bond] = []
        seen: set[tuple[int, int]] = set()
        for edge_index in range(graph.edge_index.shape[1]):
            source, target = map(int, graph.edge_index[:, edge_index])
            pair = tuple(sorted((source, target)))
            if source == target or pair in seen:
                continue
            seen.add(pair)
            order = (
                float(graph.edge_features[edge_index, 0])
                if graph.edge_features is not None and graph.edge_features.shape[1]
                else 1.0
            )
            bonds.append(Bond(pair[0], pair[1], order))
        ports = _ports_from_metadata(list(graph.metadata.get("ports", [])))
        units[unit_id] = PolymerUnit(
            id=unit_id,
            role=UnitRole(graph.metadata.get("unit_role", UnitRole.REPEAT.value)),
            graph=MolecularGraph(atoms, bonds, ports, graph.positions),
            name=graph.metadata.get("unit_name"),
        )
    nodes = [ArchitectureNode(f"u{i}", unit_id, i) for i, unit_id in enumerate(view.unit_order)]
    composition = {
        unit_id: float(view.composition[index])
        for index, unit_id in enumerate(view.unit_order)
        if float(view.composition[index]) > 0
    }
    total = sum(composition.values())
    if total > 0:
        composition = {key: value / total for key, value in composition.items()}
    elif units:
        composition = {unit_id: 1.0 / len(units) for unit_id in units}
    record = PolymerRecord(
        id=record_id or str(view.metadata.get("record_id", "generated_polymer")),
        units=units,
        architecture=PolymerArchitecture(
            ArchitectureType(view.metadata.get("architecture_type", "unknown")),
            nodes=nodes,
            sequence=list(view.unit_order),
        ),
        ensemble=EnsembleStatistics(composition=composition),
    )
    record.validate()
    return record


def single_chain2record(
    view: SingleChainView,
    *,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    schema: RepresentationSchema | None = None,
    record_id: str | None = None,
) -> PolymerRecord:
    if view.record_payload is not None:
        return record_from_dict(view.record_payload)
    inverse = (
        schema.inverse_unit_vocabulary
        if schema is not None
        else {value: key for key, value in (view.unit_order_vocabulary or {}).items()}
    )
    if view.node_type_ids is None:
        sequence = list(view.metadata.get("unit_sequence", []))
    else:
        sequence = [inverse.get(int(index), "[UNK]") for index in view.node_type_ids]
    if not sequence:
        raise ValueError("single-chain view does not contain a decodable unit sequence")
    if unit_library is None:
        raise ValueError("unit_library is required to decode a generated single-chain view")
    units = {
        unit_id: _unit_copy(unit_library[unit_id], unit_id)
        for unit_id in dict.fromkeys(sequence)
        if unit_id in unit_library
    }
    missing = set(sequence) - set(units)
    if missing:
        raise KeyError(f"unit_library is missing templates: {sorted(missing)}")
    nodes = [ArchitectureNode(f"u{i}", unit_id, i) for i, unit_id in enumerate(sequence)]
    edge_inverse = (
        schema.inverse_edge_vocabulary
        if schema is not None
        else {
            value: key
            for key, value in dict(view.metadata.get("edge_vocabulary", {})).items()
        }
    )
    edges: list[ArchitectureEdge] = []
    seen: set[tuple[int, int]] = set()
    for edge_index in range(view.edge_index.shape[1]):
        source, target = map(int, view.edge_index[:, edge_index])
        pair = tuple(sorted((source, target)))
        if source == target or pair in seen:
            continue
        seen.add(pair)
        kind_name = EdgeKind.BACKBONE.value
        if view.edge_type_ids is not None:
            kind_name = edge_inverse.get(int(view.edge_type_ids[edge_index]), kind_name)
        if kind_name == "none":
            continue
        edge_features = view.edge_features[edge_index] if view.edge_features is not None else [1, 1, 0]
        edges.append(
            ArchitectureEdge(
                source=f"u{pair[0]}",
                target=f"u{pair[1]}",
                kind=EdgeKind(kind_name),
                bond_order=float(edge_features[0]),
                probability=float(edge_features[1]) if len(edge_features) > 1 else 1.0,
                directed=bool(edge_features[2]) if len(edge_features) > 2 else False,
            )
        )
    counts = {unit_id: sequence.count(unit_id) / len(sequence) for unit_id in set(sequence)}
    states: list[SpatialState] = []
    if view.positions is not None:
        states.append(
            SpatialState(
                id=str(view.metadata.get("state_id") or "generated_chain"),
                frames=[
                    SpatialFrame(
                        Resolution.REPEAT_UNIT,
                        [node.id for node in nodes],
                        view.positions,
                    )
                ],
            )
        )
    record = PolymerRecord(
        id=record_id or str(view.metadata.get("record_id", "generated_polymer")),
        units=units,
        architecture=PolymerArchitecture(
            ArchitectureType(view.metadata.get("architecture_type", "linear")),
            nodes=nodes,
            edges=edges,
            sequence=sequence,
            head_node=nodes[0].id,
            tail_node=nodes[-1].id,
        ),
        ensemble=EnsembleStatistics(composition=counts, number_of_chains=1),
        spatial_states=states,
    )
    record.validate()
    return record


def _decode_continuous(view: GenerativeTensorView) -> dict[str, Any]:
    values = {
        name: float(view.continuous_features[index])
        for index, name in enumerate(view.feature_names)
        if view.continuous_feature_mask[index]
    }
    result: dict[str, Any] = {}
    if "log_dp" in values:
        result["dp"] = float(np.exp(values["log_dp"]))
    if "log_mn" in values:
        result["mn"] = float(np.exp(values["log_mn"]))
    if "log_dispersity_minus_one" in values:
        result["dispersity"] = 1.0 + float(np.exp(values["log_dispersity_minus_one"]))
    if "crosslink_density_log1p" in values:
        result["crosslink_density"] = float(np.expm1(values["crosslink_density_log1p"]))
    if "tacticity_logit" in values:
        result["tacticity"] = 1.0 / (1.0 + np.exp(-values["tacticity_logit"]))
    return result


def generative2record(
    view: GenerativeTensorView,
    *,
    schema: RepresentationSchema,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    record_id: str | None = None,
    prefer_payload: bool = True,
) -> PolymerRecord:
    view.validate()
    if prefer_payload and view.record_payload is not None:
        return record_from_dict(view.record_payload)
    active = np.flatnonzero(view.node_mask)
    continuous = _decode_continuous(view)
    composition_values = _softmax(view.composition_logits, view.composition_mask)
    composition = {
        unit_id: float(composition_values[index])
        for index, unit_id in schema.inverse_unit_vocabulary.items()
        if unit_id not in {schema.PAD_UNIT, schema.UNK_UNIT}
        and composition_values[index] > 1e-8
    }
    if view.level == "atom":
        atoms: list[Atom] = []
        for index in active:
            label = schema.inverse_atom_vocabulary.get(int(view.node_type[index]), "[UNK]")
            if not label.startswith("Z"):
                raise ValueError(f"cannot decode atom class {label!r}")
            atoms.append(
                Atom(
                    int(label[1:]),
                    partial_charge=float(view.node_features[index, 1])
                    if view.node_features.shape[1] > 1
                    else None,
                    mass=float(view.node_features[index, 2])
                    if view.node_features.shape[1] > 2
                    else None,
                )
            )
        bonds: list[Bond] = []
        order_map = {"single": 1.0, "aromatic": 1.5, "double": 2.0, "triple": 3.0, "other": 1.0}
        for local_i, source in enumerate(active):
            for local_j, target in enumerate(active):
                if local_j <= local_i:
                    continue
                label = schema.inverse_bond_vocabulary.get(int(view.edge_type[source, target]), "none")
                if label != "none":
                    bonds.append(Bond(local_i, local_j, order_map.get(label, 1.0), aromatic=label == "aromatic"))
        coordinates = None if view.coordinates is None else view.coordinates[active]
        unit = PolymerUnit("generated", UnitRole.REPEAT, MolecularGraph(atoms, bonds, coordinates=coordinates))
        meta = {
            "atomic_numbers": [atom.atomic_number for atom in atoms],
            "bonds": [[bond.source, bond.target] for bond in bonds],
            "bond_orders": [bond.order for bond in bonds],
            "charges": [atom.partial_charge or 0.0 for atom in atoms],
            "masses": [atom.mass or 0.0 for atom in atoms],
            "atom_to_occurrence": ["u0"] * len(atoms),
        }
        states = []
        if coordinates is not None:
            states = [SpatialState("generated", [SpatialFrame(Resolution.ATOMISTIC, [f"atom:{i}" for i in range(len(atoms))], coordinates, metadata=meta)])]
        record = PolymerRecord(
            id=record_id or "generated_atom_polymer",
            units={"generated": unit},
            architecture=PolymerArchitecture(ArchitectureType.LINEAR, [ArchitectureNode("u0", "generated", 0)], sequence=["generated"]),
            ensemble=EnsembleStatistics(composition={"generated": 1.0}, number_of_chains=1),
            spatial_states=states,
        )
        record.validate()
        return record

    if unit_library is None:
        raise ValueError("unit_library is required to decode a generated unit-level graph")
    sequence: list[str] = []
    for index in active:
        unit_id = schema.inverse_unit_vocabulary.get(int(view.node_type[index]), schema.UNK_UNIT)
        if unit_id in {schema.PAD_UNIT, schema.UNK_UNIT}:
            raise ValueError(f"cannot decode generated unit class {unit_id!r}")
        sequence.append(unit_id)
    units = {unit_id: _unit_copy(unit_library[unit_id], unit_id) for unit_id in set(sequence)}
    nodes = [ArchitectureNode(f"u{i}", unit_id, i) for i, unit_id in enumerate(sequence)]
    edges: list[ArchitectureEdge] = []
    for i, source in enumerate(active):
        for j, target in enumerate(active):
            if j <= i:
                continue
            label = schema.inverse_edge_vocabulary.get(int(view.edge_type[source, target]), "none")
            if label != "none":
                edges.append(ArchitectureEdge(f"u{i}", f"u{j}", kind=EdgeKind(label)))
    transition_probabilities = _softmax(view.transition_logits, view.transition_mask, axis=1)
    order = [unit_id for unit_id in schema.unit_vocabulary if unit_id not in {schema.PAD_UNIT, schema.UNK_UNIT} and unit_id in units]
    indices = [schema.unit_vocabulary[unit_id] for unit_id in order]
    transition = transition_probabilities[np.ix_(indices, indices)] if indices else None
    if transition is not None and transition.size:
        transition = transition / np.clip(transition.sum(axis=1, keepdims=True), 1e-12, None)
    ensemble = EnsembleStatistics(
        composition={unit_id: composition.get(unit_id, sequence.count(unit_id) / len(sequence)) for unit_id in units},
        transition_matrix=transition,
        transition_unit_order=order,
        degree_of_polymerization=Distribution(DistributionKind.DELTA, {"value": continuous["dp"]}) if "dp" in continuous else None,
        molecular_weight=Distribution(DistributionKind.DELTA, {"Mn": continuous.get("mn", 0.0), "dispersity": continuous.get("dispersity", 1.0)}) if "mn" in continuous else None,
        crosslink_density=continuous.get("crosslink_density"),
        tacticity={"generated": continuous["tacticity"]} if "tacticity" in continuous else {},
        number_of_chains=1,
    )
    states = []
    if view.coordinates is not None:
        states = [SpatialState("generated", [SpatialFrame(Resolution.REPEAT_UNIT, [node.id for node in nodes], view.coordinates[active])])]
    record = PolymerRecord(
        id=record_id or "generated_unit_polymer",
        units=units,
        architecture=PolymerArchitecture(ArchitectureType.LINEAR, nodes, edges, sequence, nodes[0].id if nodes else None, nodes[-1].id if nodes else None),
        ensemble=ensemble,
        spatial_states=states,
    )
    record.validate()
    return record


def transformer2record(
    view: TransformerInputView,
    *,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    record_id: str | None = None,
) -> PolymerRecord:
    if view.record_payload is not None:
        return record_from_dict(view.record_payload)
    tokens = [token for token, keep in zip(view.tokens, view.attention_mask, strict=True) if keep]
    unit_ids: list[str] = []
    sequence: list[str] = []
    architecture_type = ArchitectureType.UNKNOWN
    for index, token in enumerate(tokens):
        if token == "[UNIT]" and index + 1 < len(tokens):
            unit_ids.append(tokens[index + 1])
        elif token.startswith("SEQ="):
            sequence.append(token.split("=", 1)[1])
        elif token.startswith("TYPE="):
            architecture_type = ArchitectureType(token.split("=", 1)[1])
    if not sequence:
        sequence = unit_ids
    if unit_library is None:
        raise ValueError("unit_library is required to decode generated Transformer tokens")
    units = {unit_id: _unit_copy(unit_library[unit_id], unit_id) for unit_id in set(sequence)}
    nodes = [ArchitectureNode(f"u{i}", unit_id, i) for i, unit_id in enumerate(sequence)]
    edges = [ArchitectureEdge(f"u{i}", f"u{i+1}", kind=EdgeKind.BACKBONE) for i in range(max(0, len(nodes)-1))]
    counts = {unit_id: sequence.count(unit_id) / len(sequence) for unit_id in set(sequence)}
    record = PolymerRecord(
        id=record_id or str(view.metadata.get("record_id", "generated_transformer_polymer")),
        units=units,
        architecture=PolymerArchitecture(architecture_type, nodes, edges, sequence),
        ensemble=EnsembleStatistics(composition=counts, number_of_chains=1),
    )
    record.validate()
    return record


def multiscale2record(
    view: MultiScaleView,
    *,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    record_id: str | None = None,
) -> PolymerRecord:
    if view.record_payload is not None:
        return record_from_dict(view.record_payload)
    # The local chemical graphs recover unit templates; the occurrence graph then
    # follows the same decoder as SingleChainView.
    chemical_record = chemical2record(view.local_unit_graphs, record_id=record_id)
    unit_library = unit_library or chemical_record.units
    unit_type_ids = view.node_ids.get("unit_type", [])
    membership = next((relation for relation in view.relations if relation.relation == "instance_of"), None)
    if membership is None:
        raise ValueError("multiscale view has no unit-instance membership relation")
    sequence = [""] * len(view.node_ids.get("unit", []))
    for source, target in membership.edge_index.T:
        sequence[int(source)] = unit_type_ids[int(target)]
    connection = next((relation for relation in view.relations if relation.relation == "connects"), None)
    edge_index = np.empty((2, 0), dtype=np.int64) if connection is None else connection.edge_index
    positions = view.spatial.get("coordinates")
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        if view.spatial.get("resolution") == "atomistic":
            atom_to_occurrence = view.spatial.get("metadata", {}).get("atom_to_occurrence")
            if atom_to_occurrence is not None:
                labels = np.asarray(atom_to_occurrence).astype(str)
                centers = []
                for node_id in view.node_ids.get("unit", []):
                    mask = labels == str(node_id)
                    if not mask.any():
                        raise ValueError(f"atomistic multiscale data has no atoms for {node_id!r}")
                    centers.append(positions[mask].mean(axis=0))
                positions = np.asarray(centers, dtype=float)
        if len(positions) != len(sequence):
            positions = None
    chain = SingleChainView(
        node_features=view.node_features["unit"],
        edge_index=edge_index,
        edge_features=None if connection is None else connection.edge_features,
        positions=positions,
        node_ids=view.node_ids.get("unit", []),
        node_type_ids=None,
        metadata={"unit_sequence": sequence, "architecture_type": view.metadata.get("architecture_type", "unknown")},
        unit_order_vocabulary={unit_id: index for index, unit_id in enumerate(sequence)},
    )
    return single_chain2record(chain, unit_library=unit_library, record_id=record_id)


def view2record(
    view: Any,
    *,
    schema: RepresentationSchema | None = None,
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] | None = None,
    record_id: str | None = None,
    prefer_payload: bool = True,
) -> PolymerRecord:
    if isinstance(view, dict) and view and all(hasattr(item, "zynnova_record") for item in view.values()):
        return record_from_dict(next(iter(view.values())).zynnova_record)
    payload = getattr(view, "zynnova_record", None)
    if payload is not None:
        return record_from_dict(payload)
    pyg_kind = getattr(view, "zynnova_view_kind", None)
    if pyg_kind == "generative":
        from .adapters.pyg import generative_view_from_pyg
        view = generative_view_from_pyg(view)
    elif pyg_kind == "single_chain":
        from .adapters.pyg import single_chain_view_from_pyg
        view = single_chain_view_from_pyg(view)
    if isinstance(view, GenerativeTensorView):
        if schema is None:
            raise ValueError("schema is required for GenerativeTensorView decoding")
        return generative2record(view, schema=schema, unit_library=unit_library, record_id=record_id, prefer_payload=prefer_payload)
    if isinstance(view, SingleChainView):
        return single_chain2record(view, unit_library=unit_library, schema=schema, record_id=record_id)
    if isinstance(view, ChemicalStructureView):
        return chemical2record(view, record_id=record_id)
    if isinstance(view, MultiScaleView):
        return multiscale2record(view, unit_library=unit_library, record_id=record_id)
    if isinstance(view, TransformerInputView):
        return transformer2record(view, unit_library=unit_library, record_id=record_id)
    if isinstance(view, PolymerRecord):
        return view
    raise TypeError(f"unsupported polymer view type: {type(view).__name__}")


def view2stru(view: Any, **kwargs: Any) -> StructureData | Any:
    record_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in {"schema", "unit_library", "record_id", "prefer_payload"}
    }
    return record2stru(view2record(view, **record_kwargs), **kwargs)
