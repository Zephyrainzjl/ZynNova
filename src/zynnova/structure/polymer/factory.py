from __future__ import annotations

from enum import Enum
from typing import Any

from .core.polymer import PolymerRecord
from .views import (
    to_chemical_structure_view,
    to_generative_view,
    to_multiscale_view,
    to_single_chain_view,
    to_transformer_view,
)


class ViewKind(str, Enum):
    CHEMICAL = "chemical"
    SINGLE_CHAIN = "single_chain"
    MULTISCALE = "multiscale"
    GENERATIVE = "generative"
    GRAPH_GENERATIVE = "graph_generative"
    ATOM_GENERATIVE = "atom_generative"
    TRANSFORMER = "transformer"


def make_view(record: PolymerRecord, kind: ViewKind | str, **kwargs: Any) -> Any:
    selected = ViewKind(kind)
    if selected is ViewKind.CHEMICAL:
        return to_chemical_structure_view(record, **kwargs)
    if selected is ViewKind.SINGLE_CHAIN:
        return to_single_chain_view(record, **kwargs)
    if selected is ViewKind.MULTISCALE:
        return to_multiscale_view(record, **kwargs)
    if selected in {ViewKind.GENERATIVE, ViewKind.GRAPH_GENERATIVE}:
        kwargs.setdefault("level", "unit")
        return to_generative_view(record, **kwargs)
    if selected is ViewKind.ATOM_GENERATIVE:
        kwargs.setdefault("level", "atom")
        return to_generative_view(record, **kwargs)
    if selected is ViewKind.TRANSFORMER:
        return to_transformer_view(record, **kwargs)
    raise AssertionError("unreachable")


record2view = make_view
