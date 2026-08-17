from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from zynnova.structure import StructureData

from .exceptions import MissingBackendError


def _require_ase():
    try:
        import ase  # noqa: F401
        from ase.io import read
    except ImportError as exc:
        raise MissingBackendError(
            "ASE is required for molecular dynamics; install zynnova[dynamics]"
        ) from exc
    return read


def to_ase_atoms(structure: Any, *, format: str | None = None, index: int | str = -1):
    """Convert supported ZynNova structures to a detached ASE ``Atoms`` object."""
    read = _require_ase()
    if isinstance(structure, StructureData):
        return structure.to_ase()
    if hasattr(structure, "get_positions") and hasattr(structure, "get_atomic_numbers"):
        return structure.copy()
    try:
        from zynnova.structure.polymer import PolymerRecord, record2stru
    except ImportError:
        PolymerRecord = ()  # type: ignore[assignment]
    if PolymerRecord and isinstance(structure, PolymerRecord):
        return record2stru(structure).to_ase()
    if isinstance(structure, (str, Path)):
        return read(str(structure), format=format, index=index)
    raise TypeError(
        "structure must be StructureData, ASE Atoms, PolymerRecord, or an ASE-readable path"
    )


def to_structure_data(atoms: Any) -> StructureData:
    return StructureData.from_ase(atoms)


def restore_like(template: Any, atoms: Any) -> Any:
    """Restore a relaxed/simulated structure to the input object's high-level type."""
    data = to_structure_data(atoms)
    if isinstance(template, StructureData):
        return data
    if hasattr(template, "get_positions") and hasattr(template, "get_atomic_numbers"):
        return atoms.copy()
    try:
        from zynnova.structure.polymer import PolymerRecord
        from zynnova.structure.polymer.core import PeriodicBox, Resolution, SpatialFrame
    except ImportError:
        PolymerRecord = ()  # type: ignore[assignment]
    if PolymerRecord and isinstance(template, PolymerRecord):
        record = deepcopy(template)
        state = next(
            (
                item
                for item in record.spatial_states
                if item.frames and item.frames[-1].resolution is Resolution.ATOMISTIC
            ),
            None,
        )
        box = None
        if data.pbc.any():
            box = PeriodicBox(
                matrix=data.cell.copy(),
                periodic=tuple(bool(value) for value in data.pbc),
            )
        frame = SpatialFrame(
            resolution=Resolution.ATOMISTIC,
            node_ids=[f"atom:{index}" for index in range(data.num_atoms)],
            coordinates=data.positions.copy(),
            box=box,
            velocities=atoms.get_velocities(),
            forces=None,
            metadata={
                "atomic_numbers": data.atomic_numbers.tolist(),
                "structure_info": dict(data.info),
            },
        )
        if state is None:
            from zynnova.structure.polymer.core import SpatialState

            record.spatial_states.append(SpatialState(id="dynamics", frames=[frame]))
        else:
            state.frames.append(frame)
        return record
    return data
