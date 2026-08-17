"""Built-in phase-field model catalog."""

from .base import CustomPhaseFieldModel, PhaseFieldModel
from .canonical import (
    AdvectiveCahnHilliardModel,
    AllenCahnModel,
    CahnHilliardModel,
    CoupledAllenCahnCahnHilliardModel,
    ReactionCahnHilliardModel,
)
from .electrochemical import (
    ElectrochemicalReactionPhaseFieldModel,
    ElectrodepositionModel,
    IntercalationPhaseFieldModel,
)
from .fracture import (
    ChemoMechanicalFractureModel,
    FatiguePhaseFieldFractureModel,
    FractureRegularization,
    PhaseFieldFractureModel,
)
from .multiphase import (
    GrandPotentialModel,
    GrainGrowthModel,
    KKSModel,
    MultiphaseFieldModel,
    OrientationFieldModel,
    SinteringModel,
)
from .pfc import (
    MolecularBeamEpitaxyModel,
    OhtaKawasakiModel,
    PhaseFieldCrystalModel,
    SwiftHohenbergModel,
)
from .solidification import BinaryAlloySolidificationModel, DendriticSolidificationModel

__all__ = [
    "AdvectiveCahnHilliardModel",
    "AllenCahnModel",
    "BinaryAlloySolidificationModel",
    "CahnHilliardModel",
    "ChemoMechanicalFractureModel",
    "CoupledAllenCahnCahnHilliardModel",
    "CustomPhaseFieldModel",
    "DendriticSolidificationModel",
    "ElectrochemicalReactionPhaseFieldModel",
    "ElectrodepositionModel",
    "FatiguePhaseFieldFractureModel",
    "FractureRegularization",
    "GrandPotentialModel",
    "GrainGrowthModel",
    "IntercalationPhaseFieldModel",
    "KKSModel",
    "MolecularBeamEpitaxyModel",
    "MultiphaseFieldModel",
    "OhtaKawasakiModel",
    "OrientationFieldModel",
    "PhaseFieldCrystalModel",
    "PhaseFieldFractureModel",
    "PhaseFieldModel",
    "ReactionCahnHilliardModel",
    "SinteringModel",
    "SwiftHohenbergModel",
]
