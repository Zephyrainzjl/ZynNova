from __future__ import annotations

from typing import Any, Literal

from ..common.backends import BackendName
from ..common.conversion import structure_to_graph
from ..common.features import FeatureConfig
from ..common.io import load_structure
from ..common.types import GraphData
from .core import PolymerRecord
from .factory import ViewKind, make_view
from .io.json_codec import record_to_dict
from .record_conversion import record2stru, stru2record
from .schema import RepresentationSchema


def stru2graph(
    structure: Any,
    *,
    representation: Literal[
        "atom", "single_chain", "multiscale", "generative", "generative_atom"
    ] = "atom",
    record: PolymerRecord | None = None,
    schema: RepresentationSchema | None = None,
    state_id: str | None = None,
    frame_index: int = 0,
    format: str | None = None,
    index: int | str = -1,
    backend: BackendName = "auto",
    neighbor_mode: str = "radius",
    cutoff: float = 3.0,
    radius_scale: float = 1.2,
    max_neighbors: int | None = None,
    directed: bool = True,
    self_edges: bool = False,
    use_explicit_bonds: bool = True,
    feature_config: FeatureConfig | None = None,
    tolerance: float = 1.0e-8,
    as_pyg: bool = False,
    include_reconstruction: bool = True,
    **record_kwargs: Any,
) -> GraphData | Any:
    """Convert a polymer structure to an atom graph or a polymer ML view.

    The default ``representation='atom'`` matches ``molecular.stru2graph`` and
    ``crystal.stru2graph``. Other representations expose unit-level, multiscale,
    and dense graph-generative tensors through the same namespace.
    """
    if record is None:
        if isinstance(structure, PolymerRecord):
            record = structure
        else:
            record = stru2record(
                structure,
                format=format,
                index=index,
                **record_kwargs,
            )
    if representation == "atom":
        data = record2stru(record, state_id=state_id, frame_index=frame_index)
        graph = structure_to_graph(
            data,
            structure_kind="polymer",
            backend=backend,
            neighbor_mode=neighbor_mode,
            cutoff=cutoff,
            radius_scale=radius_scale,
            max_neighbors=max_neighbors,
            directed=directed,
            self_edges=self_edges,
            use_explicit_bonds=use_explicit_bonds,
            feature_config=feature_config,
            tolerance=tolerance,
        )
        if include_reconstruction:
            graph.graph_attrs["polymer_record"] = record_to_dict(record)
        graph.graph_attrs["polymer_representation"] = "atom"
        return graph.to_pyg(include_metadata=include_reconstruction) if as_pyg else graph
    if representation == "single_chain":
        view = make_view(
            record,
            ViewKind.SINGLE_CHAIN,
            schema=schema,
            state_id=state_id,
            frame_index=frame_index,
            include_reconstruction=include_reconstruction,
        )
    elif representation == "multiscale":
        view = make_view(
            record,
            ViewKind.MULTISCALE,
            state_id=state_id,
            frame_index=frame_index,
            include_reconstruction=include_reconstruction,
        )
    elif representation in {"generative", "generative_atom"}:
        if schema is None:
            raise ValueError("schema is required for generative representations")
        view = make_view(
            record,
            ViewKind.GENERATIVE,
            schema=schema,
            level="atom" if representation == "generative_atom" else "unit",
            state_id=state_id,
            frame_index=frame_index,
            include_reconstruction=include_reconstruction,
        )
    else:
        raise ValueError(f"unsupported polymer representation: {representation}")
    if not as_pyg:
        return view
    from .adapters.pyg import view_to_pyg

    return view_to_pyg(view, include_reconstruction=include_reconstruction)


def stru2pyg(structure: Any, **kwargs: Any):
    kwargs["as_pyg"] = True
    return stru2graph(structure, **kwargs)


to_graph = stru2graph

from .graph2stru import graph2stru  # noqa: E402

__all__ = ["stru2graph", "stru2pyg", "to_graph", "graph2stru"]
