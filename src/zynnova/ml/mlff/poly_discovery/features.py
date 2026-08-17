from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


_SMARTS: dict[str, str] = {
    "carbonyl": "[CX3]=[OX1]",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "imide": "[NX3]([CX3](=[OX1]))[CX3](=[OX1])",
    "ether": "[OD2]([#6])[#6]",
    "hydroxyl": "[OX2H]",
    "nitrile": "[CX2]#N",
    "sulfone": "[SX4](=[OX1])(=[OX1])",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "carbonate": "[OX2][CX3](=[OX1])[OX2]",
    "conjugated_alkene": "[C,c]=[C,c]",
    "imine": "[CX3]=[NX2]",
    "fluorinated_carbon": "[#6]-[F]",
    "chlorinated_carbon": "[#6]-[Cl]",
    "aromatic_ring_atom": "[a]",
    "amine": "[NX3;H2,H1,H0;!$(N-C=O)]",
}

_FALLBACK_PATTERNS: dict[str, str] = {
    "carbonyl": r"C\(=O\)|C=O",
    "ester": r"C\(=O\)O|OC\(=O\)",
    "amide": r"NC\(=O\)|C\(=O\)N",
    "imide": r"C\(=O\)N.*C\(=O\)",
    "ether": r"COC|cOc",
    "hydroxyl": r"(?<![A-Za-z])O(?![A-Za-z])|\[OH",
    "nitrile": r"C#N",
    "sulfone": r"S\(=O\)\(=O\)",
    "sulfonamide": r"S\(=O\)\(=O\)N",
    "carbonate": r"OC\(=O\)O",
    "conjugated_alkene": r"C=C|c=c",
    "imine": r"C=N",
    "fluorinated_carbon": r"F",
    "chlorinated_carbon": r"Cl",
    "aromatic_ring_atom": r"[cnosp]",
    "amine": r"N",
}


@dataclass(slots=True)
class PolymerFeatureVector:
    psmiles: str
    values: dict[str, float]
    exact_chemistry: bool
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def shannon_entropy(fractions: Sequence[float], *, normalize: bool = False) -> float:
    """Return ``-sum(x log x)`` after normalizing non-negative fractions."""

    values = np.asarray(fractions, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("fractions must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if normalize and probabilities.size > 1:
        entropy /= math.log(probabilities.size)
    return entropy


def effective_component_count(fractions: Sequence[float]) -> float:
    """Hill number of order one, ``exp(S/R)``."""

    return float(math.exp(shannon_entropy(fractions)))


class FunctionalGroupFeaturizer:
    """Chemical and entropy descriptors for polymer repeat-unit SMILES.

    RDKit SMARTS matching is used when available. The dependency-free fallback
    is deliberately marked approximate and should only support smoke tests or
    early triage, never publication-grade functional-group attribution.
    """

    def __init__(
        self,
        *,
        normalize_group_counts: bool = True,
        exclude_cc_single_from_bond_entropy: bool = True,
    ) -> None:
        self.normalize_group_counts = bool(normalize_group_counts)
        self.exclude_cc_single_from_bond_entropy = bool(
            exclude_cc_single_from_bond_entropy
        )

    def transform(
        self,
        psmiles: str,
        *,
        monomer_fractions: Mapping[str, float] | None = None,
        extra_features: Mapping[str, float] | None = None,
    ) -> PolymerFeatureVector:
        psmiles = str(psmiles).strip()
        if not psmiles:
            raise ValueError("psmiles cannot be empty")
        try:
            values, metadata = self._rdkit_features(psmiles)
            exact = True
            warnings: tuple[str, ...] = ()
        except ImportError:
            values, metadata = self._fallback_features(psmiles)
            exact = False
            warnings = (
                "RDKit is unavailable; functional-group counts are approximate.",
            )
        except ValueError as exc:
            values, metadata = self._fallback_features(psmiles)
            exact = False
            warnings = (f"RDKit could not parse pSMILES ({exc}); fallback counts used.",)

        group_names = tuple(_SMARTS)
        group_counts = [values.get(f"group_{name}_count", 0.0) for name in group_names]
        values["functional_group_entropy_R"] = shannon_entropy(group_counts)
        values["functional_group_effective_count"] = effective_component_count(
            group_counts
        )
        if monomer_fractions:
            fractions = [float(value) for value in monomer_fractions.values()]
            values["monomer_configurational_entropy_R"] = shannon_entropy(fractions)
            values["monomer_effective_component_count"] = effective_component_count(
                fractions
            )
        if extra_features:
            values.update({str(key): float(value) for key, value in extra_features.items()})
        return PolymerFeatureVector(
            psmiles=psmiles,
            values=values,
            exact_chemistry=exact,
            warnings=warnings,
            metadata=metadata,
        )

    def transform_many(
        self,
        psmiles: Sequence[str],
    ) -> list[PolymerFeatureVector]:
        return [self.transform(value) for value in psmiles]

    def _rdkit_features(self, psmiles: str) -> tuple[dict[str, float], dict[str, Any]]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except ImportError as exc:
            raise ImportError from exc
        molecule = Chem.MolFromSmiles(psmiles)
        if molecule is None:
            raise ValueError("invalid repeat-unit SMILES")
        atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0]
        heavy_count = max(len(atoms), 1)
        atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=int)
        values: dict[str, float] = {
            "heavy_atom_count": float(len(atoms)),
            "molecular_weight_repeat_g_mol": float(Descriptors.MolWt(molecule)),
            "heteroatom_fraction": float(np.mean(atomic_numbers != 6)),
            "fluorine_fraction": float(np.mean(atomic_numbers == 9)),
            "chlorine_fraction": float(np.mean(atomic_numbers == 17)),
            "nitrogen_fraction": float(np.mean(atomic_numbers == 7)),
            "oxygen_fraction": float(np.mean(atomic_numbers == 8)),
            "aromatic_atom_fraction": float(
                sum(atom.GetIsAromatic() for atom in atoms) / heavy_count
            ),
            "formal_charge_abs": float(
                sum(abs(atom.GetFormalCharge()) for atom in atoms)
            ),
        }
        try:
            from rdkit.Chem import Lipinski

            values["rotatable_bond_fraction"] = float(
                Lipinski.NumRotatableBonds(molecule) / heavy_count
            )
        except (ImportError, AttributeError):
            values["rotatable_bond_fraction"] = 0.0

        raw_group_counts: list[float] = []
        for name, smarts in _SMARTS.items():
            query = Chem.MolFromSmarts(smarts)
            count = float(len(molecule.GetSubstructMatches(query))) if query else 0.0
            raw_group_counts.append(count)
            values[f"group_{name}_count"] = count
            values[f"group_{name}_fraction"] = count / heavy_count
            if self.normalize_group_counts:
                values[f"group_{name}"] = count / heavy_count
            else:
                values[f"group_{name}"] = count

        bond_counts: dict[str, int] = {}
        for bond in molecule.GetBonds():
            left = bond.GetBeginAtom()
            right = bond.GetEndAtom()
            if left.GetAtomicNum() == 0 or right.GetAtomicNum() == 0:
                continue
            elements = tuple(sorted((left.GetSymbol(), right.GetSymbol())))
            order = float(bond.GetBondTypeAsDouble())
            if (
                self.exclude_cc_single_from_bond_entropy
                and elements == ("C", "C")
                and abs(order - 1.0) < 1.0e-8
            ):
                continue
            key = f"{elements[0]}-{elements[1]}:{order:g}"
            bond_counts[key] = bond_counts.get(key, 0) + 1
        values["bond_type_count"] = float(len(bond_counts))
        values["bond_configurational_entropy_R"] = shannon_entropy(
            list(bond_counts.values())
        )
        values["bond_effective_type_count"] = effective_component_count(
            list(bond_counts.values())
        )
        return values, {
            "bond_type_counts": bond_counts,
            "group_names": tuple(_SMARTS),
            "group_raw_counts": tuple(raw_group_counts),
        }

    def _fallback_features(self, psmiles: str) -> tuple[dict[str, float], dict[str, Any]]:
        token_pattern = re.compile(r"Cl|Br|[BCNOFPSIbcno*]")
        tokens = token_pattern.findall(psmiles)
        atoms = [token for token in tokens if token != "*"]
        heavy_count = max(len(atoms), 1)
        carbon_count = sum(token in {"C", "c"} for token in atoms)
        values: dict[str, float] = {
            "heavy_atom_count": float(len(atoms)),
            "molecular_weight_repeat_g_mol": float("nan"),
            "heteroatom_fraction": float(1.0 - carbon_count / heavy_count),
            "fluorine_fraction": float(atoms.count("F") / heavy_count),
            "chlorine_fraction": float(atoms.count("Cl") / heavy_count),
            "nitrogen_fraction": float(
                sum(token in {"N", "n"} for token in atoms) / heavy_count
            ),
            "oxygen_fraction": float(
                sum(token in {"O", "o"} for token in atoms) / heavy_count
            ),
            "aromatic_atom_fraction": float(
                sum(token.islower() for token in atoms) / heavy_count
            ),
            "formal_charge_abs": float(
                len(re.findall(r"\[[^\]]*[+-]\d*[^\]]*\]", psmiles))
            ),
            "rotatable_bond_fraction": float(psmiles.count("-") / heavy_count),
        }
        counts: dict[str, float] = {}
        for name, pattern in _FALLBACK_PATTERNS.items():
            count = float(len(re.findall(pattern, psmiles)))
            counts[name] = count
            values[f"group_{name}_count"] = count
            values[f"group_{name}_fraction"] = count / heavy_count
            values[f"group_{name}"] = (
                count / heavy_count if self.normalize_group_counts else count
            )
        crude_bonds = {
            "single": psmiles.count("-"),
            "double": psmiles.count("="),
            "triple": psmiles.count("#"),
            "aromatic": sum(character.islower() for character in psmiles),
        }
        positive = [value for value in crude_bonds.values() if value > 0]
        values["bond_type_count"] = float(len(positive))
        values["bond_configurational_entropy_R"] = shannon_entropy(positive)
        values["bond_effective_type_count"] = effective_component_count(positive)
        return values, {"fallback_group_counts": counts, "fallback_bonds": crude_bonds}


def feature_matrix(
    vectors: Sequence[PolymerFeatureVector],
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not vectors:
        raise ValueError("at least one feature vector is required")
    if names is None:
        common = set(vectors[0].values)
        for vector in vectors[1:]:
            common.intersection_update(vector.values)
        names = tuple(sorted(common))
    resolved = tuple(str(name) for name in names)
    matrix = np.asarray(
        [[vector.values.get(name, float("nan")) for name in resolved] for vector in vectors],
        dtype=float,
    )
    return matrix, resolved


__all__ = [
    "FunctionalGroupFeaturizer",
    "PolymerFeatureVector",
    "effective_component_count",
    "feature_matrix",
    "shannon_entropy",
]
