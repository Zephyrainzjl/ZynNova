from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

Array = np.ndarray


def _array(value: Any, dtype: Any, shape_tail: tuple[int, ...] | None = None) -> Array:
    out = np.asarray(value, dtype=dtype)
    if shape_tail is not None and out.shape[-len(shape_tail):] != shape_tail:
        raise ValueError(f"Expected trailing shape {shape_tail}, got {out.shape}")
    return np.ascontiguousarray(out)


@dataclass(slots=True)
class StructureData:
    """Backend-neutral atomistic structure.

    Positions and cell vectors use Å.  ``cell`` stores lattice vectors as rows,
    matching ASE's convention.  Optional arrays are copied so callers can safely
    mutate their original objects after conversion.
    """

    atomic_numbers: Array
    positions: Array
    cell: Array = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    pbc: Array = field(default_factory=lambda: np.zeros(3, dtype=bool))
    charges: Array | None = None
    masses: Array | None = None
    tags: Array | None = None
    bonds: Array | None = None
    bond_orders: Array | None = None
    arrays: dict[str, Array] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        self.atomic_numbers = _array(self.atomic_numbers, np.int64)
        self.positions = _array(self.positions, np.float64, (3,))
        self.cell = _array(self.cell, np.float64)
        self.pbc = _array(self.pbc, bool)
        if self.atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must have shape [N]")
        if self.positions.shape != (len(self.atomic_numbers), 3):
            raise ValueError("positions must have shape [N, 3]")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must have shape [3, 3]")
        if self.pbc.shape != (3,):
            raise ValueError("pbc must have shape [3]")
        for name in ("charges", "masses", "tags"):
            value = getattr(self, name)
            if value is not None:
                dtype = np.int64 if name == "tags" else np.float64
                value = _array(value, dtype)
                if value.shape != (len(self.atomic_numbers),):
                    raise ValueError(f"{name} must have shape [N]")
                setattr(self, name, value)
        if self.bonds is not None:
            self.bonds = _array(self.bonds, np.int64)
            if self.bonds.ndim != 2 or self.bonds.shape[1] != 2:
                raise ValueError("bonds must have shape [M, 2]")
            if self.bonds.size and (self.bonds.min() < 0 or self.bonds.max() >= self.num_atoms):
                raise ValueError("bonds contain invalid atom indices")
            if self.bond_orders is None:
                self.bond_orders = np.ones(len(self.bonds), dtype=np.float64)
        if self.bond_orders is not None:
            self.bond_orders = _array(self.bond_orders, np.float64)
            if self.bonds is None or self.bond_orders.shape != (len(self.bonds),):
                raise ValueError("bond_orders must have shape [M] and require bonds")
        self.arrays = {k: np.array(v, copy=True) for k, v in self.arrays.items()}
        self.info = dict(self.info)

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def fractional_positions(self) -> Array:
        if not np.any(self.pbc):
            return np.zeros_like(self.positions)
        if abs(np.linalg.det(self.cell)) < 1e-14:
            raise ValueError("Cannot compute fractional positions from a singular cell")
        return self.positions @ np.linalg.inv(self.cell)

    def copy(self) -> "StructureData":
        return StructureData(
            atomic_numbers=self.atomic_numbers.copy(),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
            charges=None if self.charges is None else self.charges.copy(),
            masses=None if self.masses is None else self.masses.copy(),
            tags=None if self.tags is None else self.tags.copy(),
            bonds=None if self.bonds is None else self.bonds.copy(),
            bond_orders=None if self.bond_orders is None else self.bond_orders.copy(),
            arrays={k: v.copy() for k, v in self.arrays.items()},
            info=dict(self.info),
            source=self.source,
        )

    def to_ase(self):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError("ASE is required; install zynnova[io]") from exc
        atoms = Atoms(
            numbers=self.atomic_numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            info=dict(self.info),
        )
        if self.charges is not None:
            atoms.set_initial_charges(self.charges)
        if self.masses is not None:
            atoms.set_masses(self.masses)
        if self.tags is not None:
            atoms.set_tags(self.tags)
        reserved = {"numbers", "positions", "initial_charges", "masses", "tags"}
        for name, value in self.arrays.items():
            if name not in reserved and len(value) == self.num_atoms:
                atoms.set_array(name, np.array(value, copy=True))
        if self.bonds is not None:
            atoms.info["zynnova_bonds"] = self.bonds.tolist()
            atoms.info["zynnova_bond_orders"] = self.bond_orders.tolist()
        return atoms

    @classmethod
    def from_ase(cls, atoms: Any, *, source: str | None = None) -> "StructureData":
        arrays = {
            key: np.array(value, copy=True)
            for key, value in atoms.arrays.items()
            if key not in {"numbers", "positions", "initial_charges", "masses", "tags"}
        }
        bonds = atoms.info.get("zynnova_bonds")
        bond_orders = atoms.info.get("zynnova_bond_orders")
        charges = atoms.get_initial_charges()
        if not np.any(charges):
            charges = None
        return cls(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            cell=np.asarray(atoms.cell.array, dtype=np.float64),
            pbc=np.asarray(atoms.pbc, dtype=bool),
            charges=charges,
            masses=atoms.get_masses(),
            tags=atoms.get_tags(),
            bonds=bonds,
            bond_orders=bond_orders,
            arrays=arrays,
            info=dict(atoms.info),
            source=source,
        )


@dataclass(slots=True)
class GraphData:
    """Rich homogeneous atom graph with lossless structure metadata.

    The edge convention is
    ``edge_vec[e] = pos[target] + edge_shift[e] @ cell - pos[source]``.
    It makes periodic-image information explicit and supports exact reconstruction
    of the central structure even after conversion to PyTorch Geometric.
    """

    atomic_numbers: Array
    positions: Array
    edge_index: Array
    edge_shift: Array
    edge_vec: Array
    edge_dist: Array
    node_features: Array
    edge_features: Array
    cell: Array
    pbc: Array
    fractional_positions: Array | None = None
    charges: Array | None = None
    masses: Array | None = None
    tags: Array | None = None
    node_feature_names: tuple[str, ...] = ()
    edge_feature_names: tuple[str, ...] = ()
    node_attrs: dict[str, Array] = field(default_factory=dict)
    edge_attrs: dict[str, Array] = field(default_factory=dict)
    graph_attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atomic_numbers = _array(self.atomic_numbers, np.int64)
        self.positions = _array(self.positions, np.float64, (3,))
        self.edge_index = _array(self.edge_index, np.int64)
        self.edge_shift = _array(self.edge_shift, np.int64)
        self.edge_vec = _array(self.edge_vec, np.float64)
        self.edge_dist = _array(self.edge_dist, np.float64)
        self.node_features = _array(self.node_features, np.float32)
        self.edge_features = _array(self.edge_features, np.float32)
        self.cell = _array(self.cell, np.float64)
        self.pbc = _array(self.pbc, bool)
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        edge_count = self.edge_index.shape[1]
        if self.edge_shift.shape != (edge_count, 3):
            raise ValueError("edge_shift must have shape [E, 3]")
        if self.edge_vec.shape != (edge_count, 3):
            raise ValueError("edge_vec must have shape [E, 3]")
        if self.edge_dist.shape not in {(edge_count,), (edge_count, 1)}:
            raise ValueError("edge_dist must have shape [E] or [E, 1]")
        self.edge_dist = self.edge_dist.reshape(edge_count)
        if self.node_features.shape[0] != self.num_nodes:
            raise ValueError("node_features first dimension must equal num_nodes")
        if self.edge_features.shape[0] != edge_count:
            raise ValueError("edge_features first dimension must equal num_edges")
        if self.cell.shape != (3, 3) or self.pbc.shape != (3,):
            raise ValueError("cell and pbc must have shapes [3,3] and [3]")
        if self.fractional_positions is not None:
            self.fractional_positions = _array(self.fractional_positions, np.float64)
            if self.fractional_positions.shape != (self.num_nodes, 3):
                raise ValueError("fractional_positions must have shape [N,3]")
        for name in ("charges", "masses", "tags"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (self.num_nodes,):
                    raise ValueError(f"{name} must have shape [N]")
                setattr(self, name, np.ascontiguousarray(value))
        if edge_count and (self.edge_index.min() < 0 or self.edge_index.max() >= self.num_nodes):
            raise ValueError("edge_index contains invalid node indices")
        self.node_attrs = {k: np.array(v, copy=True) for k, v in self.node_attrs.items()}
        self.edge_attrs = {k: np.array(v, copy=True) for k, v in self.edge_attrs.items()}
        self.graph_attrs = dict(self.graph_attrs)

    @property
    def num_nodes(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def z(self) -> Array:
        return self.atomic_numbers

    @property
    def pos(self) -> Array:
        return self.positions

    @property
    def x(self) -> Array:
        return self.node_features

    @property
    def edge_attr(self) -> Array:
        return self.edge_features

    def validate_geometry(self, *, atol: float = 1e-8) -> None:
        source, target = self.edge_index
        expected = self.positions[target] + self.edge_shift @ self.cell - self.positions[source]
        if not np.allclose(expected, self.edge_vec, atol=atol, rtol=0):
            raise ValueError("edge_vec is inconsistent with positions, cell, and edge_shift")
        if not np.allclose(np.linalg.norm(self.edge_vec, axis=1), self.edge_dist, atol=atol, rtol=0):
            raise ValueError("edge_dist is inconsistent with edge_vec")

    def to_pyg(self, *, include_metadata: bool = False):
        """Return a batch-safe ``torch_geometric.data.Data`` object.

        Geometry is stored as float64 to keep periodic round-trips numerically
        stable. Set ``include_metadata=True`` only for single-graph inspection;
        arbitrary source metadata is intentionally omitted by default because it
        may not be collatable by :class:`torch_geometric.data.Batch`.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("PyTorch Geometric is required; install zynnova[graph]") from exc

        data = Data(
            x=torch.as_tensor(self.node_features, dtype=torch.float32),
            z=torch.as_tensor(self.atomic_numbers, dtype=torch.long),
            pos=torch.as_tensor(self.positions, dtype=torch.float64),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_features, dtype=torch.float32),
        )
        data.edge_shift = torch.as_tensor(self.edge_shift, dtype=torch.long)
        data.edge_vec = torch.as_tensor(self.edge_vec, dtype=torch.float64)
        data.edge_dist = torch.as_tensor(self.edge_dist[:, None], dtype=torch.float64)
        data.cell = torch.as_tensor(self.cell[None, :, :], dtype=torch.float64)
        data.pbc = torch.as_tensor(self.pbc[None, :], dtype=torch.bool)
        data.natoms = torch.tensor([self.num_nodes], dtype=torch.long)
        if self.fractional_positions is not None:
            data.frac_pos = torch.as_tensor(
                self.fractional_positions, dtype=torch.float64
            )
        if self.charges is not None:
            data.charge = torch.as_tensor(self.charges, dtype=torch.float64)
        if self.masses is not None:
            data.mass = torch.as_tensor(self.masses, dtype=torch.float64)
        if self.tags is not None:
            data.tags = torch.as_tensor(self.tags, dtype=torch.long)
        for name, value in self.node_attrs.items():
            if not hasattr(data, name) and np.asarray(value).shape[0] == self.num_nodes:
                setattr(data, name, torch.as_tensor(value))
        for name, value in self.edge_attrs.items():
            if not hasattr(data, name) and np.asarray(value).shape[0] == self.num_edges:
                setattr(data, name, torch.as_tensor(value))
        if include_metadata:
            # Useful for inspection and round-tripping a single graph, but not
            # guaranteed to batch when source parsers retain custom objects.
            data.node_feature_names = self.node_feature_names
            data.edge_feature_names = self.edge_feature_names
            data.zynnova_meta = dict(self.graph_attrs)
        return data

    @classmethod
    def from_pyg(cls, data: Any) -> "GraphData":
        def cpu_numpy(value: Any) -> Array:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        cell = cpu_numpy(data.cell)
        if cell.ndim == 3:
            if cell.shape[0] != 1:
                raise ValueError("GraphData.from_pyg expects one graph, not a batched Data object")
            cell = cell[0]
        pbc = cpu_numpy(data.pbc)
        if pbc.ndim == 2:
            if pbc.shape[0] != 1:
                raise ValueError("GraphData.from_pyg expects one graph, not a batch")
            pbc = pbc[0]
        edge_dist = cpu_numpy(data.edge_dist).reshape(-1)
        known = {
            "x", "z", "pos", "edge_index", "edge_attr", "edge_shift", "edge_vec",
            "edge_dist", "cell", "pbc", "natoms", "frac_pos", "charge", "mass", "tags",
            "node_feature_names", "edge_feature_names", "zynnova_meta",
        }
        node_attrs: dict[str, Array] = {}
        edge_attrs: dict[str, Array] = {}
        for key, value in data.to_dict().items():
            if key in known or not hasattr(value, "shape"):
                continue
            array = cpu_numpy(value)
            if array.ndim and array.shape[0] == len(data.z):
                node_attrs[key] = array
            elif array.ndim and array.shape[0] == data.edge_index.shape[1]:
                edge_attrs[key] = array
        return cls(
            atomic_numbers=cpu_numpy(data.z),
            positions=cpu_numpy(data.pos),
            edge_index=cpu_numpy(data.edge_index),
            edge_shift=cpu_numpy(data.edge_shift),
            edge_vec=cpu_numpy(data.edge_vec),
            edge_dist=edge_dist,
            node_features=cpu_numpy(data.x),
            edge_features=cpu_numpy(data.edge_attr),
            cell=cell,
            pbc=pbc,
            fractional_positions=(
                cpu_numpy(data.frac_pos) if hasattr(data, "frac_pos") else None
            ),
            charges=cpu_numpy(data.charge) if hasattr(data, "charge") else None,
            masses=cpu_numpy(data.mass) if hasattr(data, "mass") else None,
            tags=cpu_numpy(data.tags) if hasattr(data, "tags") else None,
            node_feature_names=tuple(getattr(data, "node_feature_names", ())),
            edge_feature_names=tuple(getattr(data, "edge_feature_names", ())),
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            graph_attrs=dict(getattr(data, "zynnova_meta", {})),
        )

    def to_structure(self, *, include_edges_as_bonds: bool = False) -> StructureData:
        bonds = None
        bond_orders = None
        if include_edges_as_bonds and self.num_edges:
            source, target = self.edge_index
            zero_shift = np.all(self.edge_shift == 0, axis=1)
            keep = zero_shift & (source < target)
            bonds = np.column_stack((source[keep], target[keep]))
            if "bond_order" in self.edge_attrs:
                bond_orders = np.asarray(self.edge_attrs["bond_order"])[keep]
            else:
                bond_orders = np.ones(len(bonds), dtype=np.float64)
        return StructureData(
            atomic_numbers=self.atomic_numbers.copy(),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
            charges=None if self.charges is None else self.charges.copy(),
            masses=None if self.masses is None else self.masses.copy(),
            tags=None if self.tags is None else self.tags.copy(),
            bonds=bonds,
            bond_orders=bond_orders,
            arrays={k: v.copy() for k, v in self.node_attrs.items()},
            info=dict(self.graph_attrs.get("structure_info", {})),
            source=self.graph_attrs.get("source"),
        )

    def save_npz(self, path: str | Path) -> None:
        """Store the numeric graph payload in a portable compressed NPZ file."""
        path = Path(path)
        np.savez_compressed(
            path,
            atomic_numbers=self.atomic_numbers,
            positions=self.positions,
            edge_index=self.edge_index,
            edge_shift=self.edge_shift,
            edge_vec=self.edge_vec,
            edge_dist=self.edge_dist,
            node_features=self.node_features,
            edge_features=self.edge_features,
            cell=self.cell,
            pbc=self.pbc,
            fractional_positions=(
                np.empty((0, 3)) if self.fractional_positions is None
                else self.fractional_positions
            ),
            node_feature_names=np.asarray(self.node_feature_names, dtype=str),
            edge_feature_names=np.asarray(self.edge_feature_names, dtype=str),
        )
