from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ....structure import StructureData


# Covalent radii and approximate neutral valence limits for QM9 elements.
_COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_MAX_VALENCE = {1: 1, 6: 4, 7: 4, 8: 2, 9: 1}


@dataclass(slots=True)
class GeometryReport:
    minimum_distance_A: float
    connected: bool
    approximate_valence_valid: bool
    inferred_bond_count: int
    collision_count: int
    component_count: int
    atom_valences: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair_distances(positions: np.ndarray) -> np.ndarray:
    displacement = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(displacement, axis=-1)


def infer_bonds(
    structure: StructureData,
    *,
    bond_scale: float = 1.25,
    minimum_bond_distance_A: float = 0.35,
) -> np.ndarray:
    z = np.asarray(structure.atomic_numbers, dtype=np.int64)
    distance = _pair_distances(np.asarray(structure.positions, dtype=np.float64))
    bonds: list[tuple[int, int]] = []
    for first in range(len(z)):
        radius_first = _COVALENT_RADII.get(int(z[first]), 0.75)
        for second in range(first + 1, len(z)):
            radius_second = _COVALENT_RADII.get(int(z[second]), 0.75)
            cutoff = bond_scale * (radius_first + radius_second)
            value = float(distance[first, second])
            if minimum_bond_distance_A <= value <= cutoff:
                bonds.append((first, second))
    if not bonds:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(bonds, dtype=np.int64)


def _component_count(atom_count: int, bonds: np.ndarray) -> int:
    if atom_count == 0:
        return 0
    adjacency: list[list[int]] = [[] for _ in range(atom_count)]
    for first, second in bonds:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    visited: set[int] = set()
    components = 0
    for start in range(atom_count):
        if start in visited:
            continue
        components += 1
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def analyze_generated_structure(
    structure: StructureData,
    *,
    minimum_distance_A: float = 0.55,
    bond_scale: float = 1.25,
) -> tuple[GeometryReport, np.ndarray]:
    positions = np.asarray(structure.positions, dtype=np.float64)
    if len(positions) < 2:
        minimum_distance = float("inf")
        collision_count = 0
    else:
        distance = _pair_distances(positions)
        upper = distance[np.triu_indices(len(positions), k=1)]
        minimum_distance = float(np.min(upper))
        collision_count = int(np.sum(upper < minimum_distance_A))
    bonds = infer_bonds(structure, bond_scale=bond_scale)
    valences = np.zeros(structure.num_atoms, dtype=np.int64)
    for first, second in bonds:
        valences[int(first)] += 1
        valences[int(second)] += 1
    valid = all(
        int(valences[index]) <= _MAX_VALENCE.get(int(z), 6)
        for index, z in enumerate(structure.atomic_numbers)
    )
    components = _component_count(structure.num_atoms, bonds)
    return (
        GeometryReport(
            minimum_distance_A=minimum_distance,
            connected=components <= 1,
            approximate_valence_valid=bool(valid),
            inferred_bond_count=int(len(bonds)),
            collision_count=collision_count,
            component_count=components,
            atom_valences=tuple(int(value) for value in valences),
        ),
        bonds,
    )


__all__ = [
    "GeometryReport",
    "analyze_generated_structure",
    "infer_bonds",
]
