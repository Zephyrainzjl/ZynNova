"""Model registry and aliases for the complete built-in phase-field catalog."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import (
    AdvectiveCahnHilliardModel,
    AllenCahnModel,
    BinaryAlloySolidificationModel,
    CahnHilliardModel,
    ChemoMechanicalFractureModel,
    CoupledAllenCahnCahnHilliardModel,
    DendriticSolidificationModel,
    ElectrochemicalReactionPhaseFieldModel,
    ElectrodepositionModel,
    FatiguePhaseFieldFractureModel,
    GrandPotentialModel,
    GrainGrowthModel,
    IntercalationPhaseFieldModel,
    KKSModel,
    MolecularBeamEpitaxyModel,
    MultiphaseFieldModel,
    OhtaKawasakiModel,
    OrientationFieldModel,
    PhaseFieldCrystalModel,
    PhaseFieldFractureModel,
    PhaseFieldModel,
    ReactionCahnHilliardModel,
    SinteringModel,
    SwiftHohenbergModel,
)

ModelFactory = Callable[..., PhaseFieldModel]


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    name: str
    factory: ModelFactory
    aliases: tuple[str, ...]
    category: str
    description: str


class PhaseFieldModelRegistry:
    """Extensible registry covering canonical and application-specific models."""

    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._alias_to_name: dict[str, str] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("_", "-").replace(" ", "-")

    def register(
        self,
        name: str,
        factory: ModelFactory,
        *,
        aliases: tuple[str, ...] = (),
        category: str = "custom",
        description: str = "",
        replace: bool = False,
    ) -> None:
        canonical = self._normalize(name)
        if canonical in self._descriptors and not replace:
            raise KeyError(f"phase-field model {canonical!r} is already registered")
        descriptor = ModelDescriptor(
            canonical,
            factory,
            tuple(self._normalize(alias) for alias in aliases),
            category,
            description,
        )
        self._descriptors[canonical] = descriptor
        self._alias_to_name[canonical] = canonical
        for alias in descriptor.aliases:
            if alias in self._alias_to_name and not replace:
                raise KeyError(f"phase-field alias {alias!r} is already registered")
            self._alias_to_name[alias] = canonical

    def create(self, name: str, /, **parameters: Any) -> PhaseFieldModel:
        normalized = self._normalize(name)
        try:
            canonical = self._alias_to_name[normalized]
            descriptor = self._descriptors[canonical]
        except KeyError as exc:
            choices = ", ".join(sorted(self._descriptors))
            raise KeyError(f"unknown phase-field model {name!r}; available: {choices}") from exc
        return descriptor.factory(**parameters)

    def descriptor(self, name: str) -> ModelDescriptor:
        normalized = self._normalize(name)
        canonical = self._alias_to_name[normalized]
        return self._descriptors[canonical]

    def names(self, *, category: str | None = None) -> tuple[str, ...]:
        descriptors = self._descriptors.values()
        if category is not None:
            descriptors = (item for item in descriptors if item.category == category)
        return tuple(sorted(item.name for item in descriptors))

    def catalog(self) -> Mapping[str, ModelDescriptor]:
        return dict(self._descriptors)


MODEL_REGISTRY = PhaseFieldModelRegistry()


def _register_builtins() -> None:
    entries = [
        ("allen-cahn", AllenCahnModel, ("model-a", "ac"), "canonical", "Non-conserved order-parameter dynamics."),
        ("cahn-hilliard", CahnHilliardModel, ("model-b", "ch"), "canonical", "Conserved composition dynamics."),
        ("model-c", CoupledAllenCahnCahnHilliardModel, ("ac-ch", "coupled-ac-ch"), "canonical", "Coupled conserved/non-conserved dynamics."),
        ("advective-cahn-hilliard", AdvectiveCahnHilliardModel, ("model-h", "ch-navier-stokes"), "flow", "Cahn-Hilliard transport with external velocity coupling."),
        ("reaction-cahn-hilliard", ReactionCahnHilliardModel, ("rch",), "reaction", "Conserved phase separation with reactions."),
        ("grain-growth", GrainGrowthModel, ("polycrystal", "multi-order-parameter"), "microstructure", "Polycrystalline grain coarsening."),
        ("multiphase-field", MultiphaseFieldModel, ("steinbach", "multiphase"), "microstructure", "Simplex-constrained multiphase evolution."),
        ("kks", KKSModel, ("kim-kim-suzuki",), "thermodynamics", "Two-phase KKS partitioning model."),
        ("grand-potential", GrandPotentialModel, ("grand-potential-multiphase",), "thermodynamics", "Chemical-potential-based multiphase formulation."),
        ("sintering", SinteringModel, ("powder-sintering",), "microstructure", "Density and grain evolution during sintering."),
        ("orientation-field", OrientationFieldModel, ("kwc", "kobayashi-warren-carter"), "microstructure", "Phase and crystallographic orientation evolution."),
        ("phase-field-crystal", PhaseFieldCrystalModel, ("pfc",), "pattern", "Atomically resolved periodic density on diffusive time scales."),
        ("swift-hohenberg", SwiftHohenbergModel, ("sh",), "pattern", "Pattern-forming Swift-Hohenberg dynamics."),
        ("ohta-kawasaki", OhtaKawasakiModel, ("block-copolymer", "diblock"), "polymer", "Microphase separation with long-range interaction."),
        ("molecular-beam-epitaxy", MolecularBeamEpitaxyModel, ("mbe", "epitaxial-growth"), "surface", "Fourth-order epitaxial surface growth."),
        ("dendritic-solidification", DendriticSolidificationModel, ("kobayashi-dendrite", "dendrite"), "solidification", "Anisotropic thermal dendrite growth."),
        ("binary-alloy-solidification", BinaryAlloySolidificationModel, ("alloy-solidification",), "solidification", "Coupled phase, solute, and thermal fields."),
        ("intercalation-phase-field", IntercalationPhaseFieldModel, ("battery-particle", "lithiation"), "electrochemical", "Phase-separating intercalation electrode."),
        ("electrodeposition", ElectrodepositionModel, ("plating", "dendrite-electrodeposition"), "electrochemical", "Metal deposition with ions and electric potential."),
        ("electrochemical-reaction", ElectrochemicalReactionPhaseFieldModel, ("butler-volmer-phase-field",), "electrochemical", "Generic reactive electrochemical phase field."),
        ("phase-field-fracture", PhaseFieldFractureModel, ("at1", "at2", "cohesive-fracture"), "fracture", "Brittle/cohesive diffuse fracture."),
        ("fatigue-phase-field-fracture", FatiguePhaseFieldFractureModel, ("fatigue-fracture",), "fracture", "Fatigue-degraded fracture toughness."),
        ("chemo-mechanical-fracture", ChemoMechanicalFractureModel, ("battery-fracture", "diffusion-fracture"), "fracture", "Composition-damage coupling."),
    ]
    for name, factory, aliases, category, description in entries:
        MODEL_REGISTRY.register(
            name,
            factory,
            aliases=aliases,
            category=category,
            description=description,
        )


_register_builtins()


def register_model(*args: Any, **kwargs: Any) -> None:
    MODEL_REGISTRY.register(*args, **kwargs)


def create_model(name: str, /, **parameters: Any) -> PhaseFieldModel:
    return MODEL_REGISTRY.create(name, **parameters)


def available_models(*, category: str | None = None) -> tuple[str, ...]:
    return MODEL_REGISTRY.names(category=category)


__all__ = [
    "MODEL_REGISTRY",
    "ModelDescriptor",
    "PhaseFieldModelRegistry",
    "available_models",
    "create_model",
    "register_model",
]
