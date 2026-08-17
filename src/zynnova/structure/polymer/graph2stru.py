from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..common.conversion import coerce_graph
from ..common.io import write_structure
from ..common.types import GraphData, StructureData
from .decoder import view2stru
from .io.json_codec import record_from_dict
from .record_conversion import record2stru


def graph2stru(
    graph: Any,
    *,
    output: Literal["structure", "ase"] = "structure",
    include_edges_as_bonds: bool = True,
    path: str | Path | None = None,
    format: str | None = None,
    **decode_kwargs: Any,
) -> StructureData | Any:
    """Decode atom graphs, PyG objects, or polymer-specific training views."""
    if isinstance(graph, GraphData) or (
        hasattr(graph, "edge_index") and hasattr(graph, "pos") and hasattr(graph, "z")
    ):
        rich_graph = coerce_graph(graph)
        payload = rich_graph.graph_attrs.get("polymer_record")
        if payload is not None:
            structure = record2stru(record_from_dict(payload))
        else:
            rich_graph.validate_geometry()
            structure = rich_graph.to_structure(include_edges_as_bonds=include_edges_as_bonds)
    else:
        structure = view2stru(graph, output="structure", **decode_kwargs)
    if path is not None:
        write_structure(structure, path, format=format)
    if output == "ase":
        return structure.to_ase()
    if output != "structure":
        raise ValueError("output must be 'structure' or 'ase'")
    return structure


to_structure = graph2stru
