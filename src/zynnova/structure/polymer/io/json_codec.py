from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..core import (
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
    PeriodicBox,
    PolymerArchitecture,
    PolymerRecord,
    PolymerUnit,
    ProcessHistory,
    ProcessStep,
    PropertyValue,
    Provenance,
    Resolution,
    SpatialFrame,
    SpatialState,
    UnitRole,
)


def _primitive(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def record_to_dict(record: PolymerRecord) -> dict[str, Any]:
    record.validate()
    return _primitive(record)


def _distribution(data: dict[str, Any] | None) -> Distribution | None:
    if data is None:
        return None
    return Distribution(
        kind=DistributionKind(data["kind"]),
        parameters=dict(data.get("parameters", {})),
        samples=np.asarray(data["samples"], dtype=float) if data.get("samples") is not None else None,
        bin_edges=np.asarray(data["bin_edges"], dtype=float) if data.get("bin_edges") is not None else None,
        probabilities=np.asarray(data["probabilities"], dtype=float)
        if data.get("probabilities") is not None
        else None,
        unit=data.get("unit"),
        uncertainty=dict(data.get("uncertainty", {})),
        metadata=dict(data.get("metadata", {})),
    )


def _molecular_graph(data: dict[str, Any]) -> MolecularGraph:
    atoms = [
        Atom(
            atomic_number=item["atomic_number"],
            formal_charge=item.get("formal_charge", 0),
            isotope=item.get("isotope"),
            aromatic=item.get("aromatic", False),
            chirality=item.get("chirality"),
            mass=item.get("mass"),
            partial_charge=item.get("partial_charge"),
            name=item.get("name"),
            features=dict(item.get("features", {})),
        )
        for item in data.get("atoms", [])
    ]
    bonds = [
        Bond(
            source=item["source"],
            target=item["target"],
            order=item.get("order", 1.0),
            aromatic=item.get("aromatic", False),
            conjugated=item.get("conjugated", False),
            stereo=item.get("stereo"),
            kind=item.get("kind", "covalent"),
            features=dict(item.get("features", {})),
        )
        for item in data.get("bonds", [])
    ]
    ports = [
        ConnectionPort(
            id=item["id"],
            atom_index=item["atom_index"],
            port_type=item.get("port_type", "generic"),
            direction=item.get("direction"),
            valence=item.get("valence", 1),
            leaving_atom_indices=list(item.get("leaving_atom_indices", [])),
            allowed_partner_types=set(item.get("allowed_partner_types", [])),
            features=dict(item.get("features", {})),
        )
        for item in data.get("ports", [])
    ]
    return MolecularGraph(
        atoms=atoms,
        bonds=bonds,
        ports=ports,
        coordinates=np.asarray(data["coordinates"], dtype=float)
        if data.get("coordinates") is not None
        else None,
        metadata=dict(data.get("metadata", {})),
    )


def record_from_dict(data: dict[str, Any]) -> PolymerRecord:
    units = {
        unit_id: PolymerUnit(
            id=item["id"],
            role=UnitRole(item["role"]),
            graph=_molecular_graph(item["graph"]),
            name=item.get("name"),
            aliases=list(item.get("aliases", [])),
            descriptors=dict(item.get("descriptors", {})),
            metadata=dict(item.get("metadata", {})),
        )
        for unit_id, item in data["units"].items()
    }

    architecture_data = data["architecture"]
    architecture = PolymerArchitecture(
        architecture_type=ArchitectureType(architecture_data["architecture_type"]),
        nodes=[
            ArchitectureNode(
                id=item["id"],
                unit_id=item["unit_id"],
                occurrence=item.get("occurrence"),
                role=item.get("role"),
                features=dict(item.get("features", {})),
            )
            for item in architecture_data.get("nodes", [])
        ],
        edges=[
            ArchitectureEdge(
                source=item["source"],
                target=item["target"],
                source_port=item.get("source_port"),
                target_port=item.get("target_port"),
                kind=EdgeKind(item.get("kind", EdgeKind.POLYMER_CONNECTION.value)),
                bond_order=item.get("bond_order", 1.0),
                probability=item.get("probability", 1.0),
                directed=item.get("directed", False),
                features=dict(item.get("features", {})),
            )
            for item in architecture_data.get("edges", [])
        ],
        sequence=list(architecture_data["sequence"])
        if architecture_data.get("sequence") is not None
        else None,
        head_node=architecture_data.get("head_node"),
        tail_node=architecture_data.get("tail_node"),
        metadata=dict(architecture_data.get("metadata", {})),
    )

    ensemble_data = data.get("ensemble", {})
    ensemble = EnsembleStatistics(
        composition=dict(ensemble_data.get("composition", {})),
        transition_matrix=np.asarray(ensemble_data["transition_matrix"], dtype=float)
        if ensemble_data.get("transition_matrix") is not None
        else None,
        transition_unit_order=list(ensemble_data.get("transition_unit_order", [])),
        degree_of_polymerization=_distribution(
            ensemble_data.get("degree_of_polymerization")
        ),
        molecular_weight=_distribution(ensemble_data.get("molecular_weight")),
        branch_length=_distribution(ensemble_data.get("branch_length")),
        block_length=_distribution(ensemble_data.get("block_length")),
        tacticity=dict(ensemble_data.get("tacticity", {})),
        end_group_fraction=dict(ensemble_data.get("end_group_fraction", {})),
        crosslink_density=ensemble_data.get("crosslink_density"),
        number_of_chains=ensemble_data.get("number_of_chains"),
        metadata=dict(ensemble_data.get("metadata", {})),
    )

    spatial_states: list[SpatialState] = []
    for state_data in data.get("spatial_states", []):
        frames: list[SpatialFrame] = []
        for frame_data in state_data.get("frames", []):
            box_data = frame_data.get("box")
            box = None
            if box_data is not None:
                box = PeriodicBox(
                    matrix=np.asarray(box_data["matrix"], dtype=float),
                    periodic=tuple(box_data.get("periodic", [True, True, True])),
                    unit=box_data.get("unit", "angstrom"),
                )
            frames.append(
                SpatialFrame(
                    resolution=Resolution(frame_data["resolution"]),
                    node_ids=list(frame_data["node_ids"]),
                    coordinates=np.asarray(frame_data["coordinates"], dtype=float),
                    box=box,
                    velocities=np.asarray(frame_data["velocities"], dtype=float)
                    if frame_data.get("velocities") is not None
                    else None,
                    forces=np.asarray(frame_data["forces"], dtype=float)
                    if frame_data.get("forces") is not None
                    else None,
                    orientations=np.asarray(frame_data["orientations"], dtype=float)
                    if frame_data.get("orientations") is not None
                    else None,
                    spatial_edge_index=np.asarray(
                        frame_data["spatial_edge_index"], dtype=np.int64
                    )
                    if frame_data.get("spatial_edge_index") is not None
                    else None,
                    periodic_edge_shift=np.asarray(
                        frame_data["periodic_edge_shift"], dtype=np.int64
                    )
                    if frame_data.get("periodic_edge_shift") is not None
                    else None,
                    phase_labels=np.asarray(frame_data["phase_labels"])
                    if frame_data.get("phase_labels") is not None
                    else None,
                    time=frame_data.get("time"),
                    units=dict(frame_data.get("units", {"length": "angstrom"})),
                    metadata=dict(frame_data.get("metadata", {})),
                )
            )
        spatial_states.append(
            SpatialState(
                id=state_data["id"],
                frames=frames,
                temperature=state_data.get("temperature"),
                pressure=state_data.get("pressure"),
                solvent=state_data.get("solvent"),
                density=state_data.get("density"),
                crystallinity=state_data.get("crystallinity"),
                metadata=dict(state_data.get("metadata", {})),
            )
        )

    process_data = data.get("process_history", {})
    process_history = ProcessHistory(
        steps=[
            ProcessStep(
                id=item["id"],
                operation=item["operation"],
                parameters=dict(item.get("parameters", {})),
                input_state_ids=list(item.get("input_state_ids", [])),
                output_state_ids=list(item.get("output_state_ids", [])),
                metadata=dict(item.get("metadata", {})),
            )
            for item in process_data.get("steps", [])
        ],
        dependencies=[tuple(item) for item in process_data.get("dependencies", [])],
    )

    properties = {
        name: PropertyValue(
            name=item["name"],
            value=item["value"],
            unit=item.get("unit"),
            uncertainty=item.get("uncertainty"),
            conditions=dict(item.get("conditions", {})),
            method=item.get("method"),
            metadata=dict(item.get("metadata", {})),
        )
        for name, item in data.get("properties", {}).items()
    }
    provenance_data = data.get("provenance", {})
    provenance = Provenance(
        source_type=provenance_data.get("source_type"),
        dataset_name=provenance_data.get("dataset_name"),
        record_id=provenance_data.get("record_id"),
        reference=provenance_data.get("reference"),
        software=dict(provenance_data.get("software", {})),
        metadata=dict(provenance_data.get("metadata", {})),
    )

    record = PolymerRecord(
        id=data["id"],
        units=units,
        architecture=architecture,
        ensemble=ensemble,
        spatial_states=spatial_states,
        properties=properties,
        process_history=process_history,
        provenance=provenance,
        tags=set(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
        schema_version=data.get("schema_version", "0.1.0"),
    )
    record.validate()
    return record
