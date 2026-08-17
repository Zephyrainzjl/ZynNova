from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data import MaterialSample
from ..structure.polymer import PolymerRecord

_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|@@?|%\d{2}|"
    r"\(|\)|=|#|-|\+|\\|/|\.|:|~|@|\?|\*|\$|\d|[A-Za-z])"
)
_ATOMIC_MASSES = {
    1: 1.008,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    14: 28.085,
    15: 30.974,
    16: 32.06,
    17: 35.45,
    35: 79.904,
    53: 126.904,
}
_ELEMENTS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


@dataclass(frozen=True, slots=True)
class PolymerGraphArrays:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_weights: np.ndarray


class PSMILESTokenizer:
    """Reversible PSMILES tokenizer with a checkpointable, dataset-fitted vocabulary."""

    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    MASK = "[MASK]"
    UNK = "[UNK]"
    SPECIAL_TOKENS = (PAD, BOS, EOS, MASK, UNK)

    def __init__(self, tokens: Sequence[str] | None = None) -> None:
        vocabulary = list(self.SPECIAL_TOKENS)
        for token in tokens or ():
            if token not in vocabulary:
                vocabulary.append(str(token))
        self.tokens = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.MASK]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @classmethod
    def fit(
        cls,
        psmiles_values: Iterable[str],
        *,
        min_frequency: int = 1,
    ) -> PSMILESTokenizer:
        counts: Counter[str] = Counter()
        for value in psmiles_values:
            counts.update(tokenize_psmiles(value))
        tokens = [
            token
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_frequency
        ]
        return cls(tokens)

    def encode(
        self,
        psmiles: str,
        *,
        max_length: int,
        add_special_tokens: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = tokenize_psmiles(psmiles)
        if add_special_tokens:
            tokens = [self.BOS, *tokens, self.EOS]
        if len(tokens) > max_length:
            if add_special_tokens:
                tokens = tokens[: max_length - 1] + [self.EOS]
            else:
                tokens = tokens[:max_length]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        attention = [True] * len(ids)
        ids.extend([self.pad_id] * (max_length - len(ids)))
        attention.extend([False] * (max_length - len(attention)))
        return np.asarray(ids, dtype=np.int64), np.asarray(attention, dtype=bool)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        remove_special_tokens: bool = True,
    ) -> str:
        decoded: list[str] = []
        for index in token_ids:
            token = self.tokens[int(index)] if 0 <= int(index) < len(self.tokens) else self.UNK
            if token == self.EOS:
                break
            if remove_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            decoded.append(token)
        return "".join(decoded)

    def state_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> PSMILESTokenizer:
        tokens = [token for token in state["tokens"] if token not in cls.SPECIAL_TOKENS]
        return cls(tokens)


def tokenize_psmiles(psmiles: str) -> list[str]:
    value = "".join(str(psmiles).split())
    tokens = _TOKEN_PATTERN.findall(value)
    if "".join(tokens) != value:
        raise ValueError(f"PSMILES contains unsupported token(s): {psmiles!r}")
    return tokens


def extract_psmiles(value: MaterialSample | PolymerRecord | str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, MaterialSample):
        direct = value.metadata.get("psmiles")
        if direct:
            return str(direct).strip()
        if value.structure is None:
            raise ValueError(f"sample {value.id!r} has no polymer structure")
        return extract_psmiles(value.structure)
    direct = value.metadata.get("psmiles")
    if direct:
        return str(direct).strip()
    unit_values = [
        str(unit.metadata["psmiles"]).strip()
        for unit in value.units.values()
        if unit.metadata.get("psmiles")
    ]
    if len(unit_values) == 1:
        return unit_values[0]
    raise ValueError(f"polymer record {value.id!r} does not carry one unambiguous PSMILES string")


def polymer_record(value: MaterialSample | PolymerRecord | str) -> PolymerRecord:
    if isinstance(value, PolymerRecord):
        return value
    if isinstance(value, MaterialSample) and isinstance(value.structure, PolymerRecord):
        return value.structure
    from ..data.polymer.base import polymer_record_from_psmiles

    identifier = value.id if isinstance(value, MaterialSample) else "polymer"
    return polymer_record_from_psmiles(extract_psmiles(value), record_id=identifier)


def polymer_graph_arrays(value: MaterialSample | PolymerRecord | str) -> PolymerGraphArrays:
    record = polymer_record(value)
    node_parts: list[list[float]] = []
    weight_parts: list[float] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_parts: list[list[float]] = []
    offset = 0
    composition = record.ensemble.composition
    unit_count = max(len(record.units), 1)
    for unit_id in sorted(record.units):
        unit = record.units[unit_id]
        graph = unit.graph
        unit_weight = float(composition.get(unit_id, 1.0 / unit_count))
        atom_count = max(len(graph.atoms), 1)
        port_atoms = {port.atom_index for port in graph.ports}
        for atom_index, atom in enumerate(graph.atoms):
            mass = atom.mass
            if mass is None:
                mass = _ATOMIC_MASSES.get(atom.atomic_number, float(atom.atomic_number) * 2.0)
            node_parts.append(
                [
                    float(atom.atomic_number) / 100.0,
                    float(atom.formal_charge) / 4.0,
                    float(atom.aromatic),
                    float(mass) / 200.0,
                    float(atom.partial_charge or 0.0),
                    float(atom_index in port_atoms),
                    unit_weight,
                ]
            )
            weight_parts.append(unit_weight / atom_count)
        for bond in graph.bonds:
            attributes = [
                float(bond.order) / 3.0,
                float(bond.aromatic),
                float(bond.conjugated),
                float(bond.kind != "covalent"),
            ]
            source = offset + bond.source
            target = offset + bond.target
            edge_pairs.extend(((source, target), (target, source)))
            edge_parts.extend((attributes, attributes))
        offset += len(graph.atoms)
    if not node_parts:
        raise ValueError(f"polymer record {record.id!r} contains no atoms")
    edge_index = (
        np.asarray(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64)
    )
    edge_features = (
        np.asarray(edge_parts, dtype=np.float32)
        if edge_parts
        else np.zeros((0, 4), dtype=np.float32)
    )
    return PolymerGraphArrays(
        node_features=np.asarray(node_parts, dtype=np.float32),
        edge_index=edge_index,
        edge_features=edge_features,
        node_weights=np.asarray(weight_parts, dtype=np.float32),
    )


def configurational_entropy_R(
    bond_counts: Mapping[str, float],
    *,
    ignore_carbon_single: bool = True,
) -> float:
    """Return -sum(c_i log(c_i)); the paper's configurational entropy divided by R."""

    counts = {
        str(name): float(count)
        for name, count in bond_counts.items()
        if float(count) > 0
        and not (
            ignore_carbon_single and str(name).replace(" ", "") in {"C-C", "C-C:1", "C-C(single)"}
        )
    }
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def polymer_physics_descriptors(
    value: MaterialSample | PolymerRecord | str,
) -> dict[str, float]:
    """Calculate exact composition descriptors used by the physics-aware losses.

    Implicit hydrogen bonds are counted through RDKit when it is installed. The
    fallback uses the semantic graph and therefore remains usable for already
    materialized ``PolymerRecord`` objects without importing RDKit.
    """

    record = polymer_record(value)
    bond_counts: Counter[str] = Counter()
    atom_counts: Counter[int] = Counter()
    aromatic_atoms = 0.0
    total_weight = 0.0
    used_rdkit = False
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(extract_psmiles(value))
        if molecule is not None:
            molecule = Chem.AddHs(molecule)
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    continue
                atom_counts[atom.GetAtomicNum()] += 1.0
                aromatic_atoms += float(atom.GetIsAromatic())
            for bond in molecule.GetBonds():
                source = bond.GetBeginAtom().GetAtomicNum()
                target = bond.GetEndAtom().GetAtomicNum()
                if source == 0 or target == 0:
                    continue
                names = sorted(
                    (
                        _ELEMENTS.get(source, str(source)),
                        _ELEMENTS.get(target, str(target)),
                    )
                )
                order = float(bond.GetBondTypeAsDouble())
                suffix = "" if math.isclose(order, 1.0) else f":{order:g}"
                bond_counts[f"{names[0]}-{names[1]}{suffix}"] += 1.0
            used_rdkit = True
    except ImportError:
        pass

    if not used_rdkit:
        composition = record.ensemble.composition
        default_weight = 1.0 / max(len(record.units), 1)
        for unit_id, unit in record.units.items():
            weight = float(composition.get(unit_id, default_weight))
            for atom in unit.graph.atoms:
                atom_counts[atom.atomic_number] += weight
                aromatic_atoms += weight * float(atom.aromatic)
                mass = atom.mass or _ATOMIC_MASSES.get(
                    atom.atomic_number,
                    float(atom.atomic_number) * 2.0,
                )
                total_weight += weight * float(mass)
            for bond in unit.graph.bonds:
                source = unit.graph.atoms[bond.source].atomic_number
                target = unit.graph.atoms[bond.target].atomic_number
                names = sorted(
                    (
                        _ELEMENTS.get(source, str(source)),
                        _ELEMENTS.get(target, str(target)),
                    )
                )
                suffix = "" if math.isclose(bond.order, 1.0) else f":{bond.order:g}"
                bond_counts[f"{names[0]}-{names[1]}{suffix}"] += weight

    total_atoms = max(sum(atom_counts.values()), 1.0)
    if used_rdkit:
        total_weight = sum(
            count * _ATOMIC_MASSES.get(number, float(number) * 2.0)
            for number, count in atom_counts.items()
        )
    effective_bond_counts = {
        name: count
        for name, count in bond_counts.items()
        if name.replace(" ", "") not in {"C-C", "C-C:1", "C-C(single)"}
    }
    entropy = configurational_entropy_R(
        effective_bond_counts,
        ignore_carbon_single=False,
    )
    polar_bonds = sum(
        count
        for name, count in bond_counts.items()
        if any(element in name for element in ("F", "O", "N", "Cl"))
    )
    total_bonds = max(sum(bond_counts.values()), 1.0)
    non_carbon = sum(count for number, count in atom_counts.items() if number not in {1, 6})
    descriptors = {
        "configurational_entropy_R": float(entropy),
        "bond_type_count": float(sum(count > 0 for count in effective_bond_counts.values())),
        "fluorine_atomic_fraction": float(atom_counts[9] / total_atoms),
        "oxygen_atomic_fraction": float(atom_counts[8] / total_atoms),
        "heteroatom_fraction": float(non_carbon / total_atoms),
        "aromatic_atom_fraction": float(aromatic_atoms / total_atoms),
        "polar_bond_fraction": float(polar_bonds / total_bonds),
        "heavy_atom_count": float(total_atoms - atom_counts[1]),
        "repeat_unit_molar_mass_g_mol": float(total_weight),
        "high_entropy_margin_R": float(entropy - 1.5),
    }
    return descriptors


def family_key(value: MaterialSample | PolymerRecord | str) -> str:
    """Stable leakage-resistant family key used for grouped train/valid/test splits."""

    psmiles = extract_psmiles(value)
    normalized = re.sub(r"\[\*:?[\d]*\]|\*", "[*]", psmiles)
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolHash

        molecule = Chem.MolFromSmiles(normalized)
        if molecule is not None:
            normalized = rdMolHash.MolHash(
                molecule,
                rdMolHash.HashFunction.AnonymousGraph,
            )
    except ImportError:
        normalized = re.sub(r"\d", "", normalized)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


def split_by_family(
    samples: Sequence[MaterialSample],
    *,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[MaterialSample], list[MaterialSample], list[MaterialSample]]:
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, abs_tol=1.0e-6):
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1")
    predefined = {"train": [], "valid": [], "test": []}
    has_predefined = all(
        sample.split in {"train", "valid", "validation", "val", "test"} for sample in samples
    )
    if has_predefined:
        for sample in samples:
            key = "valid" if sample.split in {"validation", "val"} else str(sample.split)
            predefined[key].append(sample)
        return predefined["train"], predefined["valid"], predefined["test"]

    groups: dict[str, list[MaterialSample]] = {}
    for sample in samples:
        groups.setdefault(family_key(sample), []).append(sample)
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha1(f"{seed}:{item[0]}".encode()).hexdigest(),
    )
    targets = (
        len(samples) * train_ratio,
        len(samples) * valid_ratio,
        len(samples) * test_ratio,
    )
    splits: tuple[list[MaterialSample], list[MaterialSample], list[MaterialSample]] = (
        [],
        [],
        [],
    )
    for _, members in ordered:
        deficits = [
            (targets[index] - len(splits[index])) / max(targets[index], 1.0) for index in range(3)
        ]
        index = max(range(3), key=deficits.__getitem__)
        splits[index].extend(members)
    return splits


def numeric_value(mapping: Mapping[str, Any], name: str) -> float | None:
    value = mapping.get(name)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "PSMILESTokenizer",
    "PolymerGraphArrays",
    "configurational_entropy_R",
    "extract_psmiles",
    "family_key",
    "numeric_value",
    "polymer_graph_arrays",
    "polymer_physics_descriptors",
    "polymer_record",
    "split_by_family",
    "tokenize_psmiles",
]
