from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import StructureData


def load_structure(
    source: Any,
    *,
    format: str | None = None,
    index: int | str = -1,
    kind: str | None = None,
) -> StructureData:
    """Normalize a path, ASE Atoms, mapping, or StructureData.

    CIF/POSCAR/PDB/XYZ and other file formats are delegated to ASE so the core
    graph code stays independent from file-parser details.
    """
    if isinstance(source, StructureData):
        structure = source.copy()
    elif isinstance(source, dict):
        structure = StructureData(**source)
    elif isinstance(source, (str, Path)):
        try:
            from ase.io import read
        except ImportError as exc:
            raise ImportError("Reading structure files requires ASE; install zynnova[io]") from exc
        path = Path(source)
        atoms = read(path, index=index, format=format)
        if isinstance(atoms, list):
            raise ValueError("A single structure is required; choose a scalar ASE index")
        structure = StructureData.from_ase(atoms, source=str(path))
    elif hasattr(source, "get_atomic_numbers") and hasattr(source, "get_positions"):
        structure = StructureData.from_ase(source)
    else:
        raise TypeError(
            "source must be a StructureData, ASE Atoms, mapping, or structure file path"
        )

    if kind == "crystal":
        if not np.any(structure.pbc):
            if abs(np.linalg.det(structure.cell)) < 1e-14:
                raise ValueError("Crystal conversion requires a non-singular cell")
            structure.pbc = np.ones(3, dtype=bool)
    elif kind == "molecular":
        structure.pbc = np.zeros(3, dtype=bool)
        structure.cell = np.zeros((3, 3), dtype=np.float64)
    return structure


def write_structure(
    structure: StructureData,
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    try:
        from ase.io import write
    except ImportError as exc:
        raise ImportError("Writing structure files requires ASE; install zynnova[io]") from exc
    write(path, structure.to_ase(), format=format, **kwargs)
