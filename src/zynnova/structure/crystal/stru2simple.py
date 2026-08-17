from __future__ import annotations

from typing import Any

from ..common.io import load_structure
from ..common.types import StructureData


def stru2simple(
    structure: Any,
    *,
    format: str | None = None,
    index: int | str = -1,
    as_dict: bool = False,
) -> StructureData | dict[str, Any]:
    """Convert a crystal source to the canonical lightweight StructureData."""
    data = load_structure(structure, format=format, index=index, kind="crystal")
    if not as_dict:
        return data
    return {
        "atomic_numbers": data.atomic_numbers.copy(),
        "positions": data.positions.copy(),
        "cell": data.cell.copy(),
        "pbc": data.pbc.copy(),
        "charges": None if data.charges is None else data.charges.copy(),
        "masses": None if data.masses is None else data.masses.copy(),
        "tags": None if data.tags is None else data.tags.copy(),
        "arrays": {k: v.copy() for k, v in data.arrays.items()},
        "info": dict(data.info),
        "source": data.source,
    }

from .simple2stru import simple2stru  # noqa: E402

__all__ = ["stru2simple", "simple2stru"]
