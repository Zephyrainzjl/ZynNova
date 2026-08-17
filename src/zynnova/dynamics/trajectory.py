from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from zynnova.structure import StructureData

from .exceptions import MissingBackendError


def iter_trajectory(
    path: str | Path,
    *,
    index: str | slice | int = ":",
    format: str | None = None,
) -> Iterator[StructureData]:
    try:
        from ase.io import iread
    except ImportError as exc:
        raise MissingBackendError("ASE is required to read trajectories") from exc
    for atoms in iread(str(path), index=index, format=format):
        yield StructureData.from_ase(atoms, source=str(path))


def load_trajectory(
    path: str | Path,
    *,
    index: str | slice | int = ":",
    format: str | None = None,
) -> list[StructureData]:
    return list(iter_trajectory(path, index=index, format=format))


def write_trajectory(
    path: str | Path,
    frames: list[Any],
    *,
    format: str | None = None,
    append: bool = False,
) -> None:
    try:
        from ase.io import write
    except ImportError as exc:
        raise MissingBackendError("ASE is required to write trajectories") from exc
    atoms_frames = [
        frame.to_ase() if isinstance(frame, StructureData) else frame
        for frame in frames
    ]
    write(str(path), atoms_frames, format=format, append=append)
