from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..record import MaterialType
from ..source import DatasetSource


class PolymerDatasetSource(DatasetSource):
    material_type = MaterialType.POLYMER


def polymer_record_from_psmiles(
    psmiles: str,
    *,
    record_id: str = "polymer",
    name: str | None = None,
    properties: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    sequence: Sequence[str] | None = None,
):
    """Build a chemistry-level :class:`PolymerRecord` from PSMILES/SMILES.

    Dummy atoms such as ``[*:1]`` are converted to connection ports by the
    existing :mod:`zynnova.structure.polymer` RDKit adapter.  The resulting
    record is intentionally valid even when a dataset only provides one repeat
    unit and no atomistic chain conformation.
    """

    from ...structure.polymer import (
        ArchitectureEdge,
        ArchitectureNode,
        ArchitectureType,
        EdgeKind,
        EnsembleStatistics,
        PolymerArchitecture,
        PolymerRecord,
        PolymerUnit,
        PropertyValue,
        Provenance,
        UnitRole,
        molecular_graph_from_smiles,
    )

    graph = molecular_graph_from_smiles(psmiles)
    unit_id = "U0"
    unit = PolymerUnit(
        id=unit_id,
        role=UnitRole.REPEAT,
        graph=graph,
        name=name,
        metadata={"psmiles": psmiles},
    )
    unit_sequence = list(sequence or [unit_id])
    if any(value != unit_id for value in unit_sequence):
        raise ValueError("a single-PSMILES record can only use unit id 'U0'")
    nodes = [
        ArchitectureNode(id=f"u{index}", unit_id=unit_id, occurrence=index)
        for index in range(len(unit_sequence))
    ]
    ports = [port.id for port in graph.ports]
    source_port = ports[-1] if ports else None
    target_port = ports[0] if ports else None
    edges = [
        ArchitectureEdge(
            source=nodes[index].id,
            target=nodes[index + 1].id,
            source_port=source_port,
            target_port=target_port,
            kind=EdgeKind.BACKBONE,
        )
        for index in range(max(0, len(nodes) - 1))
    ]
    property_values = {
        key: value
        if isinstance(value, PropertyValue)
        else PropertyValue(name=key, value=value)
        for key, value in (properties or {}).items()
    }
    record = PolymerRecord(
        id=str(record_id),
        units={unit_id: unit},
        architecture=PolymerArchitecture(
            architecture_type=ArchitectureType.LINEAR,
            nodes=nodes,
            edges=edges,
            sequence=unit_sequence,
            head_node=nodes[0].id if nodes else None,
            tail_node=nodes[-1].id if nodes else None,
        ),
        ensemble=EnsembleStatistics(
            composition={unit_id: 1.0},
            number_of_chains=1,
        ),
        properties=property_values,
        provenance=Provenance(
            source_type="dataset",
            record_id=str(record_id),
        ),
        metadata={"psmiles": psmiles, **dict(metadata or {})},
    )
    record.validate()
    return record
