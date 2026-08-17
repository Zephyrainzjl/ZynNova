from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...common.types import StructureData


@dataclass(slots=True)
class Atom:
    atomic_number: int
    formal_charge: int = 0
    isotope: int | None = None
    aromatic: bool = False
    chirality: str | None = None
    mass: float | None = None
    partial_charge: float | None = None
    name: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.atomic_number <= 0:
            raise ValueError("atomic_number must be positive")


@dataclass(slots=True)
class Bond:
    source: int
    target: int
    order: float = 1.0
    aromatic: bool = False
    conjugated: bool = False
    stereo: str | None = None
    kind: str = "covalent"
    features: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_atoms: int) -> None:
        if self.source == self.target:
            raise ValueError("self-bond is not allowed")
        if not (0 <= self.source < num_atoms and 0 <= self.target < num_atoms):
            raise IndexError("bond atom index out of range")
        if self.order <= 0:
            raise ValueError("bond order must be positive")


@dataclass(slots=True)
class ConnectionPort:
    id: str
    atom_index: int
    port_type: str = "generic"
    direction: str | None = None
    valence: int = 1
    leaving_atom_indices: list[int] = field(default_factory=list)
    allowed_partner_types: set[str] = field(default_factory=set)
    features: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_atoms: int) -> None:
        if not self.id:
            raise ValueError("port id cannot be empty")
        if not (0 <= self.atom_index < num_atoms):
            raise IndexError("port atom index out of range")
        if self.valence < 1:
            raise ValueError("port valence must be >= 1")
        for index in self.leaving_atom_indices:
            if not (0 <= index < num_atoms):
                raise IndexError("leaving atom index out of range")


@dataclass
class MolecularGraph:
    """A unit-level atom graph used by the polymer semantic layer.

    It deliberately remains convertible to and from the package-wide
    :class:`zynnova.structure.StructureData`, so the polymer module reuses the
    existing molecular and C++/PyG conversion pipeline instead of creating an
    incompatible atom representation.
    """

    atoms: list[Atom]
    bonds: list[Bond]
    ports: list[ConnectionPort] = field(default_factory=list)
    coordinates: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.atoms:
            raise ValueError("molecular graph must contain at least one atom")
        for atom in self.atoms:
            atom.validate()
        for bond in self.bonds:
            bond.validate(len(self.atoms))
        seen_ports: set[str] = set()
        for port in self.ports:
            port.validate(len(self.atoms))
            if port.id in seen_ports:
                raise ValueError(f"duplicate port id: {port.id}")
            seen_ports.add(port.id)
        if self.coordinates is not None:
            array = np.asarray(self.coordinates, dtype=float)
            if array.shape != (len(self.atoms), 3):
                raise ValueError(
                    f"coordinates must have shape ({len(self.atoms)}, 3), got {array.shape}"
                )
            self.coordinates = np.ascontiguousarray(array)

    def port(self, port_id: str) -> ConnectionPort:
        for port in self.ports:
            if port.id == port_id:
                return port
        raise KeyError(f"unknown port: {port_id}")

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @classmethod
    def from_structure(
        cls,
        structure: StructureData,
        *,
        atom_indices: np.ndarray | list[int] | None = None,
        ports: list[ConnectionPort] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MolecularGraph":
        """Build a molecular graph from all atoms or a selected atom subset."""
        indices = (
            np.arange(structure.num_atoms, dtype=np.int64)
            if atom_indices is None
            else np.asarray(atom_indices, dtype=np.int64)
        )
        if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= structure.num_atoms)):
            raise ValueError("atom_indices contains invalid indices")
        old_to_new = {int(old): new for new, old in enumerate(indices.tolist())}
        atoms = [
            Atom(
                atomic_number=int(structure.atomic_numbers[index]),
                mass=(None if structure.masses is None else float(structure.masses[index])),
                partial_charge=(
                    None if structure.charges is None else float(structure.charges[index])
                ),
                features={
                    key: np.asarray(value)[index].item()
                    if np.asarray(value)[index].ndim == 0
                    else np.asarray(value)[index].tolist()
                    for key, value in structure.arrays.items()
                    if len(np.asarray(value)) == structure.num_atoms
                },
            )
            for index in indices
        ]
        bonds: list[Bond] = []
        if structure.bonds is not None:
            orders = (
                structure.bond_orders
                if structure.bond_orders is not None
                else np.ones(len(structure.bonds), dtype=float)
            )
            for (source, target), order in zip(structure.bonds, orders, strict=True):
                source_int, target_int = int(source), int(target)
                if source_int in old_to_new and target_int in old_to_new:
                    bonds.append(
                        Bond(old_to_new[source_int], old_to_new[target_int], float(order))
                    )
        graph = cls(
            atoms=atoms,
            bonds=bonds,
            ports=list(ports or []),
            coordinates=structure.positions[indices].copy(),
            metadata={
                "source_atom_indices": indices.tolist(),
                **dict(metadata or {}),
            },
        )
        graph.validate()
        return graph

    def to_structure(
        self,
        *,
        positions: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        pbc: np.ndarray | tuple[bool, bool, bool] | None = None,
        info: dict[str, Any] | None = None,
    ) -> StructureData:
        self.validate()
        coordinates = self.coordinates if positions is None else np.asarray(positions, dtype=float)
        if coordinates is None:
            coordinates = np.zeros((self.num_atoms, 3), dtype=float)
        if coordinates.shape != (self.num_atoms, 3):
            raise ValueError("positions must have shape [num_atoms, 3]")
        masses = np.asarray(
            [np.nan if atom.mass is None else atom.mass for atom in self.atoms], dtype=float
        )
        charges = np.asarray(
            [
                np.nan if atom.partial_charge is None else atom.partial_charge
                for atom in self.atoms
            ],
            dtype=float,
        )
        return StructureData(
            atomic_numbers=np.asarray([atom.atomic_number for atom in self.atoms]),
            positions=coordinates,
            cell=np.zeros((3, 3), dtype=float) if cell is None else cell,
            pbc=np.zeros(3, dtype=bool) if pbc is None else pbc,
            charges=None if np.isnan(charges).all() else np.nan_to_num(charges),
            masses=None if np.isnan(masses).all() else np.nan_to_num(masses),
            bonds=np.asarray([[bond.source, bond.target] for bond in self.bonds], dtype=np.int64)
            if self.bonds
            else None,
            bond_orders=np.asarray([bond.order for bond in self.bonds], dtype=float)
            if self.bonds
            else None,
            info={**dict(self.metadata), **dict(info or {})},
        )
