from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import acos, degrees

import numpy as np

from zynnova.structure.common.elements import symbols_from_numbers
from zynnova.structure.common.types import StructureData


def cell_lengths_angles(cell: np.ndarray) -> tuple[float, float, float, float, float, float]:
    matrix = np.asarray(cell, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3]")
    a, b, c = (float(np.linalg.norm(vector)) for vector in matrix)
    if min(a, b, c) <= 1e-12:
        return a, b, c, 90.0, 90.0, 90.0

    def angle(left: np.ndarray, right: np.ndarray) -> float:
        cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
        return degrees(acos(float(np.clip(cosine, -1.0, 1.0))))

    alpha = angle(matrix[1], matrix[2])
    beta = angle(matrix[0], matrix[2])
    gamma = angle(matrix[0], matrix[1])
    return a, b, c, alpha, beta, gamma


def structure_to_xyz(structure: StructureData, *, comment: str = "ZynNova") -> str:
    symbols = symbols_from_numbers(structure.atomic_numbers)
    lines = [str(structure.num_atoms), comment]
    for symbol, position in zip(symbols, structure.positions, strict=True):
        lines.append(
            f"{symbol:<3s} {position[0]: .10f} {position[1]: .10f} {position[2]: .10f}"
        )
    return "\n".join(lines) + "\n"


def structure_to_sdf(structure: StructureData, *, title: str = "ZynNova") -> str:
    """Serialize a finite atomistic structure to an SDF V2000 block.

    The serializer preserves explicit bond orders and is used for ordinary
    molecules and moderate-sized atomistic polymer chains.  Very large systems
    should use :func:`structure_to_pdb` because V2000 count fields are limited.
    """
    if structure.num_atoms > 999:
        raise ValueError("SDF V2000 supports at most 999 atoms")
    bonds = np.empty((0, 2), dtype=np.int64) if structure.bonds is None else structure.bonds
    if len(bonds) > 999:
        raise ValueError("SDF V2000 supports at most 999 bonds")
    orders = (
        np.ones(len(bonds), dtype=float)
        if structure.bond_orders is None
        else np.asarray(structure.bond_orders, dtype=float)
    )
    symbols = symbols_from_numbers(structure.atomic_numbers)
    lines = [title, "  ZynNova", ""]
    lines.append(f"{structure.num_atoms:>3d}{len(bonds):>3d}  0  0  0  0            999 V2000")
    for symbol, position in zip(symbols, structure.positions, strict=True):
        lines.append(
            f"{position[0]:10.4f}{position[1]:10.4f}{position[2]:10.4f} "
            f"{symbol:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for (source, target), order in zip(bonds, orders, strict=True):
        integer_order = int(np.clip(round(float(order)), 1, 3))
        lines.append(f"{int(source) + 1:>3d}{int(target) + 1:>3d}{integer_order:>3d}  0  0  0  0")
    lines.extend(["M  END", "$$$$"])
    return "\n".join(lines) + "\n"


def _normalize_residue_ids(
    count: int,
    residue_ids: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if residue_ids is None:
        return np.ones(count, dtype=np.int64)
    values = np.asarray(residue_ids, dtype=np.int64)
    if values.shape != (count,):
        raise ValueError("residue_ids must have shape [N]")
    return values


def _normalize_chain_ids(
    count: int,
    chain_ids: Sequence[str] | None,
) -> list[str]:
    if chain_ids is None:
        return ["A"] * count
    values = [str(value)[:1] or "A" for value in chain_ids]
    if len(values) != count:
        raise ValueError("chain_ids must have length N")
    return values


def structure_to_pdb(
    structure: StructureData,
    *,
    title: str = "ZynNova",
    residue_ids: Sequence[int] | np.ndarray | None = None,
    residue_names: Sequence[str] | None = None,
    chain_ids: Sequence[str] | None = None,
    model_index: int | None = None,
    include_conect: bool = True,
) -> str:
    """Serialize a structure to PDB with CRYST1 and explicit CONECT records."""
    if structure.num_atoms > 99999:
        raise ValueError("PDB serializer supports at most 99999 atoms")
    symbols = symbols_from_numbers(structure.atomic_numbers)
    residues = _normalize_residue_ids(structure.num_atoms, residue_ids)
    chains = _normalize_chain_ids(structure.num_atoms, chain_ids)
    if residue_names is None:
        names = ["MOL"] * structure.num_atoms
    else:
        names = [str(value).upper()[:3].ljust(3) for value in residue_names]
        if len(names) != structure.num_atoms:
            raise ValueError("residue_names must have length N")

    lines = [f"HEADER    {title[:66]}"]
    if np.any(structure.pbc) and abs(float(np.linalg.det(structure.cell))) > 1e-14:
        a, b, c, alpha, beta, gamma = cell_lengths_angles(structure.cell)
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
        )
    if model_index is not None:
        lines.append(f"MODEL     {model_index:4d}")
    for index, (symbol, position, residue, residue_name, chain) in enumerate(
        zip(symbols, structure.positions, residues, names, chains, strict=True),
        start=1,
    ):
        atom_name = symbol if len(symbol) > 1 else f" {symbol}"
        lines.append(
            f"HETATM{index:5d} {atom_name:<4s} {residue_name:>3s} {chain:1s}"
            f"{int(residue) % 10000:4d}    {position[0]:8.3f}{position[1]:8.3f}"
            f"{position[2]:8.3f}{1.00:6.2f}{0.00:6.2f}          {symbol:>2s}  "
        )
    if include_conect and structure.bonds is not None:
        neighbors: dict[int, list[int]] = defaultdict(list)
        for source, target in np.asarray(structure.bonds, dtype=np.int64):
            left, right = int(source) + 1, int(target) + 1
            if right not in neighbors[left]:
                neighbors[left].append(right)
            if left not in neighbors[right]:
                neighbors[right].append(left)
        for source in sorted(neighbors):
            targets = sorted(neighbors[source])
            for offset in range(0, len(targets), 4):
                group = targets[offset : offset + 4]
                lines.append(f"CONECT{source:5d}" + "".join(f"{target:5d}" for target in group))
    if model_index is not None:
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def structures_to_multimodel_pdb(
    frames: Sequence[StructureData],
    *,
    title: str = "ZynNova trajectory",
    residue_ids: Sequence[int] | np.ndarray | None = None,
    residue_names: Sequence[str] | None = None,
    chain_ids: Sequence[str] | None = None,
) -> str:
    """Serialize trajectory frames as one standards-friendly multi-model PDB.

    PDB stores one global ``CRYST1`` record, so an NPT trajectory with a
    changing cell displays the first-frame unit cell in py3Dmol.  Atomic
    coordinates still animate correctly for every frame.
    """
    if not frames:
        raise ValueError("at least one frame is required")
    atom_count = frames[0].num_atoms
    atomic_numbers = frames[0].atomic_numbers
    for index, frame in enumerate(frames[1:], start=1):
        if frame.num_atoms != atom_count:
            raise ValueError(
                f"trajectory frame {index} has {frame.num_atoms} atoms; "
                f"expected {atom_count}"
            )
        if not np.array_equal(frame.atomic_numbers, atomic_numbers):
            raise ValueError(
                f"trajectory frame {index} has a different atom ordering or composition"
            )

    lines = [f"HEADER    {title[:66]}"]
    first = frames[0]
    if np.any(first.pbc) and abs(float(np.linalg.det(first.cell))) > 1e-14:
        a, b, c, alpha, beta, gamma = cell_lengths_angles(first.cell)
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
            f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
        )

    for model_index, frame in enumerate(frames, start=1):
        block = structure_to_pdb(
            frame,
            title=title,
            residue_ids=residue_ids,
            residue_names=residue_names,
            chain_ids=chain_ids,
            model_index=model_index,
            include_conect=model_index == 1,
        )
        for line in block.splitlines():
            if line.startswith("HEADER") or line.startswith("CRYST1") or line == "END":
                continue
            lines.append(line)
    lines.append("END")
    return "\n".join(lines) + "\n"
