from __future__ import annotations

from typing import Literal

import numpy as np

GenerationRepresentation = Literal["psmiles", "polymer_selfies"]

POLYMER_SELFIES_PORT_ATOMIC_NUMBER = 54
POLYMER_SELFIES_PORT_TOKEN = "[Xe]"


def _require_selfies():
    try:
        import selfies
    except ImportError as exc:
        raise ImportError(
            "Polymer-SELFIES generation requires the 'selfies' package; "
            "install zynnova[ml-generation]"
        ) from exc
    return selfies


def _require_rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "Polymer-SELFIES generation requires RDKit; install zynnova[ml-generation]"
        ) from exc
    return Chem


def psmiles_to_polymer_selfies(psmiles: str) -> str:
    """Encode a two-port PSMILES repeat unit as robust Polymer-SELFIES.

    SELFIES does not natively encode dummy atoms. The two degree-one polymer
    ports are therefore replaced reversibly by xenon atoms before encoding.
    Xenon is reserved exclusively as a port surrogate in this representation.
    """

    Chem = _require_rdkit()
    selfies = _require_selfies()
    molecule = Chem.MolFromSmiles(str(psmiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse PSMILES: {psmiles!r}")
    if any(
        atom.GetAtomicNum() == POLYMER_SELFIES_PORT_ATOMIC_NUMBER for atom in molecule.GetAtoms()
    ):
        raise ValueError("native xenon atoms are reserved for Polymer-SELFIES port encoding")
    ports = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(ports) != 2:
        raise ValueError(
            f"Polymer-SELFIES requires exactly two polymerization ports; found {len(ports)}"
        )
    if any(atom.GetDegree() != 1 for atom in ports):
        raise ValueError("each Polymer-SELFIES port must be bonded to exactly one atom")

    editable = Chem.RWMol(molecule)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        atom.SetAtomicNum(POLYMER_SELFIES_PORT_ATOMIC_NUMBER)
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
    surrogate_smiles = Chem.MolToSmiles(editable, canonical=True, isomericSmiles=True)
    try:
        return str(selfies.encoder(surrogate_smiles))
    except Exception as exc:
        raise ValueError(
            f"SELFIES could not encode the port-surrogate SMILES {surrogate_smiles!r}"
        ) from exc


def _substitution_candidates(molecule, *, excluded: set[int]) -> list[int]:
    candidates = []
    for atom in molecule.GetAtoms():
        if atom.GetIdx() in excluded:
            continue
        if atom.GetAtomicNum() in {0, POLYMER_SELFIES_PORT_ATOMIC_NUMBER}:
            continue
        if int(atom.GetTotalNumHs(includeNeighbors=True)) > 0:
            candidates.append(atom.GetIdx())
    return candidates


def _farthest_pair(molecule, candidates: list[int]) -> tuple[int, int]:
    if len(candidates) >= 2:
        distances = np.asarray(_require_rdkit().GetDistanceMatrix(molecule), dtype=float)
        return max(
            (
                (left, right)
                for position, left in enumerate(candidates)
                for right in candidates[position + 1 :]
            ),
            key=lambda pair: (distances[pair[0], pair[1]], -pair[0], -pair[1]),
        )
    if len(candidates) == 1:
        atom = molecule.GetAtomWithIdx(candidates[0])
        if int(atom.GetTotalNumHs(includeNeighbors=True)) >= 2:
            return candidates[0], candidates[0]
    raise ValueError("decoded molecule has fewer than two chemically attachable port sites")


def _farthest_from(molecule, anchor: int, candidates: list[int]) -> int:
    distances = np.asarray(_require_rdkit().GetDistanceMatrix(molecule), dtype=float)
    if candidates:
        return max(candidates, key=lambda index: (distances[anchor, index], -index))
    atom = molecule.GetAtomWithIdx(anchor)
    if int(atom.GetTotalNumHs(includeNeighbors=True)) > 0:
        return anchor
    raise ValueError("decoded molecule has no second chemically attachable port site")


def _restore_two_ports(molecule):
    Chem = _require_rdkit()
    xenon_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == POLYMER_SELFIES_PORT_ATOMIC_NUMBER
    ]
    if len(xenon_atoms) > 2:
        raise ValueError("decoded Polymer-SELFIES contains more than two port-surrogate atoms")
    if any(atom.GetDegree() != 1 for atom in xenon_atoms):
        raise ValueError("a decoded port surrogate is not terminal")

    existing_anchors = [atom.GetNeighbors()[0].GetIdx() for atom in xenon_atoms]
    excluded = {atom.GetIdx() for atom in xenon_atoms}
    candidates = _substitution_candidates(molecule, excluded=excluded)
    missing = 2 - len(xenon_atoms)
    new_anchors: list[int] = []
    if missing == 2:
        new_anchors.extend(_farthest_pair(molecule, candidates))
    elif missing == 1:
        alternatives = [index for index in candidates if index != existing_anchors[0]]
        new_anchors.append(
            _farthest_from(
                molecule,
                existing_anchors[0],
                alternatives,
            )
        )

    editable = Chem.RWMol(molecule)
    map_number = 1
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != POLYMER_SELFIES_PORT_ATOMIC_NUMBER:
            continue
        atom.SetAtomicNum(0)
        atom.SetAtomMapNum(map_number)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        map_number += 1
    for anchor in new_anchors:
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(map_number)
        dummy.SetNoImplicit(True)
        dummy_index = editable.AddAtom(dummy)
        editable.AddBond(int(anchor), dummy_index, Chem.BondType.SINGLE)
        map_number += 1

    restored = editable.GetMol()
    Chem.SanitizeMol(restored)
    ports = [atom for atom in restored.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(ports) != 2 or any(atom.GetDegree() != 1 for atom in ports):
        raise ValueError("failed to restore exactly two terminal polymerization ports")
    return restored


def polymer_selfies_to_psmiles(
    polymer_selfies: str,
    *,
    repair_missing_ports: bool = True,
) -> str:
    """Decode Polymer-SELFIES to a valid, exactly two-port PSMILES string."""

    Chem = _require_rdkit()
    selfies = _require_selfies()
    try:
        surrogate_smiles = str(selfies.decoder(str(polymer_selfies)))
    except Exception as exc:
        raise ValueError("SELFIES decoder rejected the generated token sequence") from exc
    molecule = Chem.MolFromSmiles(surrogate_smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected SELFIES-decoded SMILES: {surrogate_smiles!r}")
    xenon_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == POLYMER_SELFIES_PORT_ATOMIC_NUMBER
    ]
    xenon_count = len(xenon_atoms)
    nonterminal_port = any(atom.GetDegree() != 1 for atom in xenon_atoms)
    if (xenon_count > 2 or nonterminal_port) and repair_missing_ports:
        # Under SELFIES branch semantics, a forced port control token can become
        # internal. Removing those tokens still leaves a valid SELFIES core; two
        # terminal ports are then placed at graph-distant substitutable sites.
        core_selfies = str(polymer_selfies).replace(POLYMER_SELFIES_PORT_TOKEN, "")
        if not core_selfies:
            raise ValueError("generated Polymer-SELFIES contains no molecular core")
        try:
            core_smiles = str(selfies.decoder(core_selfies))
        except Exception as exc:
            raise ValueError("SELFIES decoder rejected the port-free molecular core") from exc
        molecule = Chem.MolFromSmiles(core_smiles)
        if molecule is None:
            raise ValueError("RDKit rejected the port-free SELFIES molecular core")
        xenon_count = 0
    if xenon_count != 2 and not repair_missing_ports:
        raise ValueError(
            f"decoded Polymer-SELFIES must retain exactly two port surrogates; found {xenon_count}"
        )
    restored = _restore_two_ports(molecule)
    psmiles = Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True)
    verified = Chem.MolFromSmiles(psmiles)
    if verified is None:
        raise ValueError("RDKit could not reparse restored PSMILES")
    return psmiles


def encode_polymer_generation_sequence(
    psmiles: str,
    representation: GenerationRepresentation,
) -> str:
    if representation == "psmiles":
        return str(psmiles).strip()
    if representation == "polymer_selfies":
        return psmiles_to_polymer_selfies(psmiles)
    raise ValueError(f"unsupported polymer generation representation: {representation!r}")


def decode_polymer_generation_sequence(
    sequence: str,
    representation: GenerationRepresentation,
    *,
    repair_missing_ports: bool = True,
) -> str:
    if representation == "psmiles":
        return str(sequence).strip()
    if representation == "polymer_selfies":
        return polymer_selfies_to_psmiles(
            sequence,
            repair_missing_ports=repair_missing_ports,
        )
    raise ValueError(f"unsupported polymer generation representation: {representation!r}")


__all__ = [
    "GenerationRepresentation",
    "POLYMER_SELFIES_PORT_ATOMIC_NUMBER",
    "POLYMER_SELFIES_PORT_TOKEN",
    "decode_polymer_generation_sequence",
    "encode_polymer_generation_sequence",
    "polymer_selfies_to_psmiles",
    "psmiles_to_polymer_selfies",
]
