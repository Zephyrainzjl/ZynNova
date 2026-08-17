from __future__ import annotations

from typing import Any

from .adapters import infer_structure_kind
from .crystal import visualize_crystal
from .molecule import visualize_molecule
from .polymer import visualize_polymer
from .trajectory import visualize_trajectory


def visualize_structure(
    source: Any,
    *,
    kind: str = "auto",
    trajectory: bool = False,
    **kwargs: Any,
) -> Any:
    """Unified notebook visualization entry point.

    Parameters
    ----------
    source:
        StructureData, ASE Atoms, PolymerRecord, MolecularGraph, RDKit Mol,
        supported structure path, or a sequence/path containing trajectory frames.
    kind:
        ``"auto"``, ``"molecule"``, ``"polymer"``, or ``"crystal"``.
    trajectory:
        Route the input through the shared trajectory renderer before
        molecule/polymer/crystal dispatch.
    **kwargs:
        Forwarded to the selected specialized visualizer.
    """
    if trajectory:
        return visualize_trajectory(source, kind=kind, **kwargs)

    resolved = (
        infer_structure_kind(source, format=kwargs.get("format"))
        if kind == "auto"
        else kind
    )
    if resolved == "molecule":
        return visualize_molecule(source, **kwargs)
    if resolved == "polymer":
        return visualize_polymer(source, **kwargs)
    if resolved == "crystal":
        return visualize_crystal(source, **kwargs)
    raise ValueError(f"unknown structure kind: {resolved}")


visualize = visualize_structure
view = visualize_structure
