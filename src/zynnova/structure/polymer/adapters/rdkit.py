from __future__ import annotations

from ..core.chemistry import Atom, Bond, ConnectionPort, MolecularGraph


def molecular_graph_from_smiles(smiles: str) -> MolecularGraph:
    """Convert SMILES/PSMILES-like text to MolecularGraph.

    Dummy atoms ``[*:1]`` are converted into connection ports and removed from the
    returned atom graph. The neighboring real atom becomes the port anchor.
    """

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RDKit is required for SMILES conversion") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    dummy_indices = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0}
    old_to_new: dict[int, int] = {}
    atoms: list[Atom] = []
    for rd_atom in molecule.GetAtoms():
        if rd_atom.GetIdx() in dummy_indices:
            continue
        old_to_new[rd_atom.GetIdx()] = len(atoms)
        atoms.append(
            Atom(
                atomic_number=rd_atom.GetAtomicNum(),
                formal_charge=rd_atom.GetFormalCharge(),
                isotope=rd_atom.GetIsotope() or None,
                aromatic=rd_atom.GetIsAromatic(),
                chirality=str(rd_atom.GetChiralTag()),
                mass=rd_atom.GetMass(),
            )
        )

    ports: list[ConnectionPort] = []
    for dummy_index in sorted(dummy_indices):
        dummy = molecule.GetAtomWithIdx(dummy_index)
        neighbors = list(dummy.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError("each dummy connection atom must have exactly one neighbor")
        anchor_old = neighbors[0].GetIdx()
        map_number = dummy.GetAtomMapNum()
        port_id = f"port_{map_number or len(ports) + 1}"
        ports.append(
            ConnectionPort(
                id=port_id,
                atom_index=old_to_new[anchor_old],
                port_type="mapped" if map_number else "generic",
                features={"atom_map_number": map_number},
            )
        )

    bonds: list[Bond] = []
    for rd_bond in molecule.GetBonds():
        source_old = rd_bond.GetBeginAtomIdx()
        target_old = rd_bond.GetEndAtomIdx()
        if source_old in dummy_indices or target_old in dummy_indices:
            continue
        bonds.append(
            Bond(
                source=old_to_new[source_old],
                target=old_to_new[target_old],
                order=float(rd_bond.GetBondTypeAsDouble()),
                aromatic=rd_bond.GetIsAromatic(),
                conjugated=rd_bond.GetIsConjugated(),
                stereo=str(rd_bond.GetStereo()),
            )
        )

    graph = MolecularGraph(atoms=atoms, bonds=bonds, ports=ports, metadata={"smiles": smiles})
    graph.validate()
    return graph
