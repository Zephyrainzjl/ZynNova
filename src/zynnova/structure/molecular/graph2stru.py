from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..common.backends import BackendName, resolve_backend, validate_graph_cpp
from ..common.conversion import coerce_graph
from ..common.io import write_structure
from ..common.types import StructureData


def graph2stru(
    graph: Any,
    *,
    backend: BackendName = "auto",
    output: Literal["structure", "ase"] = "structure",
    include_edges_as_bonds: bool = True,
    path: str | Path | None = None,
    format: str | None = None,
) -> StructureData | Any:
    """Reconstruct a molecule and optionally derive unique bonds from graph edges."""
    rich_graph = coerce_graph(graph)
    selected = resolve_backend(backend)
    if selected == "cpp":
        validate_graph_cpp(rich_graph.num_nodes, rich_graph.edge_index, rich_graph.edge_shift)
    rich_graph.validate_geometry()
    if rich_graph.pbc.any():
        raise ValueError("Molecular graph2stru expects a non-periodic graph")
    structure = rich_graph.to_structure(include_edges_as_bonds=include_edges_as_bonds)
    if path is not None:
        write_structure(structure, path, format=format)
    if output == "ase":
        return structure.to_ase()
    if output != "structure":
        raise ValueError("output must be 'structure' or 'ase'")
    return structure


to_structure = graph2stru
