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
    path: str | Path | None = None,
    format: str | None = None,
) -> StructureData | Any:
    """Reconstruct the central crystal structure from GraphData or PyG Data.

    Graph edges are neighborhood information; the lossless reconstruction uses
    atomic numbers, Cartesian positions, cell, PBC, charges, masses, and retained
    per-node arrays.  Periodic image edges do not create duplicate atoms.
    """
    rich_graph = coerce_graph(graph)
    selected = resolve_backend(backend)
    if selected == "cpp":
        validate_graph_cpp(rich_graph.num_nodes, rich_graph.edge_index, rich_graph.edge_shift)
    rich_graph.validate_geometry()
    structure = rich_graph.to_structure(include_edges_as_bonds=False)
    if not structure.pbc.any():
        raise ValueError("The graph does not describe a periodic crystal")
    if path is not None:
        write_structure(structure, path, format=format)
    if output == "ase":
        return structure.to_ase()
    if output != "structure":
        raise ValueError("output must be 'structure' or 'ase'")
    return structure


to_structure = graph2stru
