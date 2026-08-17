from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..common.io import write_structure
from ..common.types import StructureData


def simple2stru(
    simple: StructureData | dict[str, Any],
    *,
    output: Literal["structure", "ase"] = "structure",
    path: str | Path | None = None,
    format: str | None = None,
) -> StructureData | Any:
    structure = simple.copy() if isinstance(simple, StructureData) else StructureData(**simple)
    if not structure.pbc.any():
        raise ValueError("Crystal simple data must have at least one periodic direction")
    if path is not None:
        write_structure(structure, path, format=format)
    return structure.to_ase() if output == "ase" else structure
