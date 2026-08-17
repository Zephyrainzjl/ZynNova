from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..core.chemistry import Atom, Bond
from ..core.polymer import PolymerRecord, PolymerUnit


ATOM_FEATURE_NAMES = (
    "atomic_number",
    "formal_charge",
    "aromatic",
    "mass",
    "partial_charge",
)

BOND_FEATURE_NAMES = (
    "bond_order",
    "aromatic",
    "conjugated",
)


def atom_features(atom: Atom) -> np.ndarray:
    return np.asarray(
        [
            atom.atomic_number,
            atom.formal_charge,
            float(atom.aromatic),
            atom.mass if atom.mass is not None else 0.0,
            atom.partial_charge if atom.partial_charge is not None else 0.0,
        ],
        dtype=np.float32,
    )


def bond_features(bond: Bond) -> np.ndarray:
    return np.asarray(
        [bond.order, float(bond.aromatic), float(bond.conjugated)],
        dtype=np.float32,
    )


def unit_scalar_features(unit: PolymerUnit, composition: float = 0.0) -> np.ndarray:
    graph = unit.graph
    molecular_weight = 0.0
    aromatic_count = 0
    hetero_count = 0
    for atom in graph.atoms:
        molecular_weight += atom.mass or 0.0
        aromatic_count += int(atom.aromatic)
        hetero_count += int(atom.atomic_number not in {1, 6})
    return np.asarray(
        [
            graph.num_atoms,
            len(graph.bonds),
            len(graph.ports),
            molecular_weight,
            aromatic_count,
            hetero_count,
            composition,
        ],
        dtype=np.float32,
    )


def graph_level_features(record: PolymerRecord) -> tuple[np.ndarray, list[str]]:
    names = [
        "num_unit_types",
        "num_architecture_nodes",
        "num_architecture_edges",
        "crosslink_density",
        "number_of_chains",
        "dp_representative",
        "mw_representative",
    ]
    ensemble = record.ensemble
    values = [
        len(record.units),
        len(record.architecture.nodes),
        len(record.architecture.edges),
        ensemble.crosslink_density or 0.0,
        float(ensemble.number_of_chains or 0),
        (
            ensemble.degree_of_polymerization.representative_value()
            if ensemble.degree_of_polymerization is not None
            else 0.0
        )
        or 0.0,
        (
            ensemble.molecular_weight.representative_value()
            if ensemble.molecular_weight is not None
            else 0.0
        )
        or 0.0,
    ]
    return np.asarray(values, dtype=np.float32), names


def stable_vocabulary(values: Mapping[str, object] | list[str] | set[str]) -> dict[str, int]:
    iterable = values.keys() if isinstance(values, Mapping) else values
    return {value: index for index, value in enumerate(sorted(iterable))}
