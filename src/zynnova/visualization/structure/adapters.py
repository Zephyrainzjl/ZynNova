from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zynnova.structure.common.io import load_structure
from zynnova.structure.common.types import StructureData


def _polymer_types() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from zynnova.structure.polymer import MolecularGraph, PolymerRecord, SpatialState
    except ImportError:
        return (), (), ()  # type: ignore[return-value]
    return PolymerRecord, MolecularGraph, SpatialState


def is_polymer_record(value: Any) -> bool:
    polymer_record, _, _ = _polymer_types()
    return bool(polymer_record) and isinstance(value, polymer_record)


def is_polymer_spatial_state(value: Any) -> bool:
    _, _, spatial_state = _polymer_types()
    return bool(spatial_state) and isinstance(value, spatial_state)


def _rdkit_to_structure(molecule: Any, *, conformer_id: int = -1) -> StructureData:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise TypeError("RDKit molecule input requires RDKit") from exc
    if not isinstance(molecule, Chem.Mol):
        raise TypeError("not an RDKit Mol")
    if molecule.GetNumConformers() == 0:
        raise ValueError("RDKit Mol has no conformer; generate 3D coordinates first")
    conformer = molecule.GetConformer(conformer_id)
    positions = np.asarray(
        [list(conformer.GetAtomPosition(index)) for index in range(molecule.GetNumAtoms())],
        dtype=float,
    )
    bonds = np.asarray(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()],
        dtype=np.int64,
    )
    orders = np.asarray([bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()], dtype=float)
    return StructureData(
        atomic_numbers=np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()]),
        positions=positions,
        bonds=None if not len(bonds) else bonds,
        bond_orders=None if not len(bonds) else orders,
        info={"source_type": "rdkit"},
    )


def coerce_structure(
    source: Any,
    *,
    kind: str | None = None,
    state_id: str | None = None,
    frame_index: int = 0,
    format: str | None = None,
) -> StructureData:
    """Convert supported ZynNova, ASE, RDKit, mapping, or file inputs."""
    polymer_record, molecular_graph, _ = _polymer_types()
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".zpoly"):
        from zynnova.structure.polymer import load_zpoly, record2stru

        record = load_zpoly(source)
        return record2stru(record, state_id=state_id, frame_index=frame_index)
    if polymer_record and isinstance(source, polymer_record):
        from zynnova.structure.polymer import record2stru

        return record2stru(source, state_id=state_id, frame_index=frame_index)
    if molecular_graph and isinstance(source, molecular_graph):
        return source.to_structure()
    try:
        return _rdkit_to_structure(source)
    except TypeError:
        pass
    normalized_kind = None if kind in {None, "auto", "polymer"} else kind
    return load_structure(source, format=format, kind=normalized_kind)


def infer_structure_kind(source: Any, *, format: str | None = None) -> str:
    if is_polymer_record(source) or is_polymer_spatial_state(source):
        return "polymer"
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".zpoly"):
        return "polymer"
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        if not source:
            raise ValueError("cannot infer structure kind from an empty sequence")
        return infer_structure_kind(source[0], format=format)
    structure = coerce_structure(source, format=format)
    return "crystal" if np.any(structure.pbc) else "molecule"


def coerce_trajectory(
    source: Any,
    *,
    state_id: str | None = None,
    format: str | None = None,
) -> list[StructureData]:
    """Normalize a sequence or a PolymerRecord trajectory to StructureData frames."""
    polymer_record, _, _ = _polymer_types()
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".zpoly"):
        from zynnova.structure.polymer import load_zpoly

        source = load_zpoly(source)
    elif isinstance(source, (str, Path)):
        try:
            from ase.io import iread
        except ImportError as exc:
            raise ImportError(
                "ASE is required to read trajectory files; install zynnova[io]"
            ) from exc
        return [
            StructureData.from_ase(atoms, source=str(source))
            for atoms in iread(str(source), index=":", format=format)
        ]
    if polymer_record and isinstance(source, polymer_record):
        from zynnova.structure.polymer import Resolution

        states = source.spatial_states
        if state_id is not None:
            states = [source.get_state(state_id)]
        output: list[StructureData] = []
        for state in states:
            for index, frame in enumerate(state.frames):
                if frame.resolution is not Resolution.ATOMISTIC:
                    continue
                try:
                    output.append(coerce_structure(source, state_id=state.id, frame_index=index))
                except (KeyError, ValueError):
                    continue
        if not output:
            output.append(coerce_structure(source, state_id=state_id))
        return output
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return [coerce_structure(item, format=format) for item in source]
    return [coerce_structure(source, format=format)]


def polymer_atom_annotations(
    record: Any,
    structure: StructureData,
    *,
    state_id: str | None = None,
    frame_index: int = 0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return PDB residue numbers, names, and chain IDs for a PolymerRecord."""
    atom_to_occurrence: list[str] | None = None
    states = record.spatial_states
    if state_id is not None:
        states = [record.get_state(state_id)]
    for state in states:
        if not state.frames:
            continue
        frame = state.frames[frame_index]
        mapping = frame.metadata.get("atom_to_occurrence")
        if mapping is not None and len(mapping) == structure.num_atoms:
            atom_to_occurrence = [str(item) for item in mapping]
            break

    node_map = {node.id: node for node in record.architecture.nodes}
    occurrence_order = {node.id: index + 1 for index, node in enumerate(record.architecture.nodes)}
    if atom_to_occurrence is None:
        return (
            np.ones(structure.num_atoms, dtype=np.int64),
            ["POL"] * structure.num_atoms,
            ["A"] * structure.num_atoms,
        )

    residues = np.asarray([occurrence_order.get(item, 1) for item in atom_to_occurrence])
    names: list[str] = []
    chains: list[str] = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chain_lookup: dict[str, str] = {}
    for occurrence in atom_to_occurrence:
        node = node_map.get(occurrence)
        unit_id = "POL" if node is None else node.unit_id
        names.append(unit_id.upper()[:3] or "POL")
        chain_key = str(node.features.get("chain_id", "0")) if node is not None else "0"
        if chain_key not in chain_lookup:
            chain_lookup[chain_key] = alphabet[len(chain_lookup) % len(alphabet)]
        chains.append(chain_lookup[chain_key])
    return residues, names, chains


def iter_atomistic_frames(record: Any, *, state_id: str | None = None) -> Iterable[tuple[str, int]]:
    from zynnova.structure.polymer import Resolution

    states = record.spatial_states if state_id is None else [record.get_state(state_id)]
    for state in states:
        for index, frame in enumerate(state.frames):
            if frame.resolution is Resolution.ATOMISTIC:
                yield state.id, index
