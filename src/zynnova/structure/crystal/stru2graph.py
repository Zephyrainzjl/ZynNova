from __future__ import annotations

from typing import Any

from ..common.backends import BackendName
from ..common.conversion import structure_to_graph
from ..common.features import FeatureConfig
from ..common.io import load_structure
from ..common.types import GraphData


def stru2graph(
    structure: Any,
    *,
    format: str | None = None,
    index: int | str = -1,
    backend: BackendName = "auto",
    neighbor_mode: str = "cutoff",
    cutoff: float = 5.0,
    radius_scale: float = 1.2,
    max_neighbors: int | None = None,
    directed: bool = True,
    self_edges: bool = False,
    feature_config: FeatureConfig | None = None,
    tolerance: float = 1.0e-8,
    as_pyg: bool = False,
) -> GraphData | Any:
    """Convert a crystal/CIF/ASE object to a rich periodic atom graph.

    ``neighbor_mode`` supports:

    * ``"cutoff"``: every periodic image within ``cutoff`` Å;
    * ``"radius"``: pair cutoff ``radius_scale * (r_i + r_j)``;
    * ``"knn"``: nearest ``max_neighbors`` candidates, with ``cutoff`` used as
      a finite candidate radius for periodic structures.

    The default directed COO graph is directly suitable for message passing.
    Periodic image offsets are retained in ``edge_shift``.
    """
    data = load_structure(structure, format=format, index=index, kind="crystal")
    graph = structure_to_graph(
        data,
        structure_kind="crystal",
        backend=backend,
        neighbor_mode=neighbor_mode,
        cutoff=cutoff,
        radius_scale=radius_scale,
        max_neighbors=max_neighbors,
        directed=directed,
        self_edges=self_edges,
        feature_config=feature_config,
        tolerance=tolerance,
    )
    graph.validate_geometry(atol=max(tolerance * 10.0, 1e-7))
    return graph.to_pyg() if as_pyg else graph


def stru2pyg(structure: Any, **kwargs: Any):
    kwargs["as_pyg"] = True
    return stru2graph(structure, **kwargs)


to_graph = stru2graph

# Paired inverse is re-exported here for users who prefer one conversion module.
from .graph2stru import graph2stru  # noqa: E402

__all__ = ["stru2graph", "stru2pyg", "to_graph", "graph2stru"]
