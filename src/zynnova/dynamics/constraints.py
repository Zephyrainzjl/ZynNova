from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .adapters import to_ase_atoms
from .exceptions import MissingBackendError


def fix_atoms(
    structure: Any,
    *,
    indices: Iterable[int] | None = None,
    mask: Iterable[bool] | None = None,
    symbols: Iterable[str] | None = None,
    tags: Iterable[int] | None = None,
):
    """Return an ASE structure with a ``FixAtoms`` constraint attached."""
    atoms = to_ase_atoms(structure)
    try:
        from ase.constraints import FixAtoms
    except ImportError as exc:
        raise MissingBackendError("ASE constraints are required") from exc
    selected = np.zeros(len(atoms), dtype=bool)
    if indices is not None:
        selected[np.asarray(list(indices), dtype=int)] = True
    if mask is not None:
        given = np.asarray(list(mask), dtype=bool)
        if given.shape != (len(atoms),):
            raise ValueError("mask must have one value per atom")
        selected |= given
    if symbols is not None:
        selected |= np.isin(atoms.get_chemical_symbols(), list(symbols))
    if tags is not None:
        selected |= np.isin(atoms.get_tags(), list(tags))
    if not np.any(selected):
        raise ValueError("No atoms were selected for fixing")
    atoms.set_constraint(FixAtoms(mask=selected))
    return atoms


def fix_bonds(structure: Any, pairs: Iterable[tuple[int, int]]):
    atoms = to_ase_atoms(structure)
    try:
        from ase.constraints import FixBondLengths
    except ImportError as exc:
        raise MissingBackendError("ASE constraints are required") from exc
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs cannot be empty")
    atoms.set_constraint(FixBondLengths(pairs))
    return atoms
