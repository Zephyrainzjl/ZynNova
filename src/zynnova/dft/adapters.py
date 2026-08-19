from __future__ import annotations

from pathlib import Path
from typing import Any

from zynnova.structure import StructureData

from .exceptions import MissingElectronicBackendError


def require_ase():
    try:
        from ase.io import read
    except ImportError as exc:
        raise MissingElectronicBackendError(
            "ASE is required for electronic-structure and AIMD workflows; install zynnova[dft]"
        ) from exc
    return read


def to_ase_atoms(
    structure: Any,
    *,
    format: str | None = None,
    index: int | str = -1,
):
    """Convert a supported input into a detached ASE ``Atoms`` object."""
    read = require_ase()
    if isinstance(structure, StructureData):
        return structure.to_ase()
    if hasattr(structure, "get_positions") and hasattr(structure, "get_atomic_numbers"):
        return structure.copy()
    if isinstance(structure, (str, Path)):
        return read(str(structure), format=format, index=index)
    raise TypeError("structure must be StructureData, ASE Atoms, or an ASE-readable path")


def to_structure_data(atoms: Any) -> StructureData:
    return StructureData.from_ase(atoms)
