from __future__ import annotations

from typing import Any

from ..core.polymer import PolymerRecord


def architecture_to_networkx(record: PolymerRecord) -> Any:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for this adapter") from exc

    graph = nx.MultiDiGraph()
    for node in record.architecture.nodes:
        graph.add_node(
            node.id,
            unit_id=node.unit_id,
            occurrence=node.occurrence,
            role=node.role,
            **node.features,
        )
    for index, edge in enumerate(record.architecture.edges):
        attributes = {
            "kind": edge.kind.value,
            "source_port": edge.source_port,
            "target_port": edge.target_port,
            "bond_order": edge.bond_order,
            "probability": edge.probability,
            **edge.features,
        }
        graph.add_edge(edge.source, edge.target, key=index, **attributes)
        if not edge.directed:
            graph.add_edge(edge.target, edge.source, key=f"{index}:reverse", **attributes)
    return graph
