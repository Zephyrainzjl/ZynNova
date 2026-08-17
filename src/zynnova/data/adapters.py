from __future__ import annotations

from typing import Any

import numpy as np


def as_structure_data(value: Any, *, kind: str | None = None):
    from ..structure import StructureData

    if isinstance(value, StructureData):
        return value
    if hasattr(value, "get_atomic_numbers") and hasattr(value, "get_positions"):
        return StructureData.from_ase(value)
    if hasattr(value, "lattice") and hasattr(value, "frac_coords"):
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise ImportError("pymatgen is required to convert this crystal object") from exc
        return StructureData.from_ase(AseAtomsAdaptor.get_atoms(value))
    if hasattr(value, "ase_converter"):
        return StructureData.from_ase(value.ase_converter())
    if isinstance(value, dict):
        if {"atomic_numbers", "positions"}.issubset(value):
            return StructureData(**value)
        if {"numbers", "positions"}.issubset(value):
            return StructureData(
                atomic_numbers=value["numbers"],
                positions=value["positions"],
                cell=value.get("cell", np.zeros((3, 3))),
                pbc=value.get("pbc", [False, False, False]),
            )
    from ..structure.common.io import load_structure

    return load_structure(value, kind=kind)


def pyg_atoms_to_structure(data: Any, *, pbc: bool = False):
    from ..structure import StructureData

    def array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    z = array(data.z)
    positions = array(data.pos)
    cell = array(data.cell) if hasattr(data, "cell") else np.zeros((3, 3))
    if cell.ndim == 3:
        cell = cell[0]
    periodic = array(data.pbc) if hasattr(data, "pbc") else np.repeat(pbc, 3)
    if periodic.ndim == 2:
        periodic = periodic[0]
    bonds = None
    bond_orders = None
    if hasattr(data, "edge_index"):
        edge_index = array(data.edge_index)
        source, target = edge_index
        keep = source < target
        bonds = np.column_stack((source[keep], target[keep]))
        if hasattr(data, "edge_attr"):
            edge_attr = array(data.edge_attr)
            if edge_attr.ndim == 2 and edge_attr.shape[1] > 0:
                bond_orders = np.ones(int(keep.sum()), dtype=np.float64)
    return StructureData(
        atomic_numbers=z,
        positions=positions,
        cell=cell,
        pbc=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )


def smiles_to_structure(
    smiles: str,
    *,
    add_hydrogens: bool = True,
    embed_3d: bool = True,
    seed: int = 0,
):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("RDKit is required to convert SMILES structures") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    if add_hydrogens:
        molecule = Chem.AddHs(molecule)
    if embed_3d:
        status = AllChem.EmbedMolecule(molecule, randomSeed=int(seed))
        if status != 0:
            status = AllChem.EmbedMolecule(molecule, useRandomCoords=True, randomSeed=int(seed))
        if status != 0:
            raise ValueError(f"RDKit could not embed SMILES: {smiles!r}")
        try:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
        except (RuntimeError, ValueError):
            pass
    else:
        AllChem.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()]
    bond_orders = [bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()]
    from ..structure import StructureData

    return StructureData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        bonds=bonds,
        bond_orders=bond_orders,
        info={"smiles": smiles},
    )
