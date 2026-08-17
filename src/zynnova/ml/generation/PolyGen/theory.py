from __future__ import annotations

from dataclasses import dataclass

from ...polymer_utils import polymer_physics_descriptors
from ...prediction.PolyPrediction.physics import high_entropy_report


@dataclass(frozen=True, slots=True)
class IrradiationMechanism:
    name: str
    introduced_bonds: tuple[str, ...]
    physical_role: str


PAPER_IRRADIATION_MECHANISMS = (
    IrradiationMechanism(
        name="dehydrofluorination_crosslinking",
        introduced_bonds=(),
        physical_role=(
            "Connects defective chains but does not by itself add a new bond type "
            "to the paper's configurational-entropy count."
        ),
    ),
    IrradiationMechanism(
        name="unsaturation",
        introduced_bonds=("C=C",),
        physical_role=(
            "Adds unsaturation and perturbs the energy contrast between 3/1-helix "
            "and all-trans conformations."
        ),
    ),
    IrradiationMechanism(
        name="oxidation_hydroxylation",
        introduced_bonds=("C=O", "C-O", "O-H"),
        physical_role=(
            "Adds several polar bond types, increasing bond configurational entropy "
            "when their molar fractions are sufficiently balanced."
        ),
    ),
)


def assess_paper_mechanisms(psmiles: str) -> dict[str, float | bool]:
    """Report structural evidence for the irradiation mechanisms in the paper."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(psmiles)
    if molecule is None:
        raise ValueError("cannot assess an invalid PSMILES string")
    patterns = {
        "has_C_F_polar_bond": "[#6]-[#9]",
        "has_C_C_double_bond": "[#6]=[#6]",
        "has_carbonyl_bond": "[#6]=[#8]",
        "has_C_O_single_bond": "[#6]-[#8]",
        "has_O_H_bond": "[#8;H1]",
        "has_branch_or_crosslink_center": "[#6;D3,D4]",
    }
    report: dict[str, float | bool] = {}
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        report[name] = bool(pattern is not None and molecule.HasSubstructMatch(pattern))
    descriptors = polymer_physics_descriptors(psmiles)
    report.update(high_entropy_report(descriptors))
    report["supports_unsaturation_mechanism"] = report["has_C_C_double_bond"]
    report["supports_oxidation_hydroxylation_mechanism"] = bool(
        report["has_carbonyl_bond"] and (report["has_C_O_single_bond"] or report["has_O_H_bond"])
    )
    report["contains_paper_guided_defect_motif"] = bool(
        report["supports_unsaturation_mechanism"]
        or report["supports_oxidation_hydroxylation_mechanism"]
    )
    return report


__all__ = [
    "IrradiationMechanism",
    "PAPER_IRRADIATION_MECHANISMS",
    "assess_paper_mechanisms",
]
