"""Closed-loop physical-property discovery and design for polymers.

The package extends ZynNova without changing the existing JouleWeave,
PolyPrism, or PolyLoom APIs. Heavy optional dependencies are imported only when
their corresponding potential, simulation, or generation operation is called.
"""

from __future__ import annotations

from .active import (
    AcquisitionBreakdown,
    ActiveDiscoveryLoop,
    propose_counterfactual_pairs,
)
from .campaign import PolymerDiscoveryCampaign
from .config import (
    ActiveLearningConfig,
    MechanismConstraint,
    MechanismDiscoveryConfig,
    MechanismGenerationConfig,
    PolymerPotentialConfig,
    PotentialPreset,
)
from .datasets import (
    PUBLIC_POLYMER_DATASETS,
    DatasetRole,
    PublicPolymerDataset,
    list_public_polymer_datasets,
    load_zynnova_polymer_dataset,
    observations_from_material_samples,
    public_dataset_plan,
)
from .discovery import MechanismDiscoveryEngine
from .features import (
    FunctionalGroupFeaturizer,
    PolymerFeatureVector,
    effective_component_count,
    feature_matrix,
    shannon_entropy,
)
from .generation import (
    MechanismGuidedCandidate,
    MechanismGuidedGenerationResult,
    generate_mechanism_guided_polymers,
    rank_mechanism_candidates,
)
from .observables import (
    BarrierStatistics,
    DielectricEstimate,
    DiffusionEstimate,
    charge_discharge_efficiency,
    cohesive_energy_density,
    dielectric_from_dipole_fluctuations,
    diffusion_from_msd,
    linear_dielectric_energy_density,
    mean_squared_displacement,
    recoverable_energy_density,
    summarize_barriers,
)
from .potential import (
    CommitteePrediction,
    LoadedPolymerPotential,
    PotentialCommittee,
    SelectedFrame,
    build_polymer_jouleweave_config,
    load_polymer_potential,
    potential_config_snapshot,
    train_polymer_potential,
)
from .physics_learning import (
    BACKENDS,
    BackendStatus,
    CONDUCTIVITY,
    CURRENT,
    DIFFUSIVITY,
    DIMENSIONLESS,
    DIPOLE,
    DimensionlessGroup,
    DynamicsDiscoveryReport,
    DynamicsEquation,
    ELECTRIC_FIELD,
    ENERGY,
    FREQUENCY,
    HessianInteractionDecomposer,
    InteractionDecomposition,
    InteractionEdge,
    KANOracleDiagnostics,
    LENGTH,
    MASS,
    NativeSymbolicBackend,
    PSEBackend,
    PhyE2EBackend,
    PhySOBackend,
    PIEZOELECTRIC_D,
    POLARIZATION,
    PRESSURE,
    PhysicalDimension,
    PhysicsEquation,
    PhysicsKANOracle,
    PhysicsLearningConfig,
    PhysicsLearningEngine,
    PhysicsLearningReport,
    PySRBackend,
    QuadraticInteractionOracle,
    SparseDynamicsDiscoverer,
    TEMPERATURE,
    TIME,
    VariableSpec,
    buckingham_pi_groups,
    create_backend,
    discover_physical_laws,
    discover_sparse_dynamics,
    dynamics_discovery_report_from_dict,
    infer_variable_spec,
    physics_learning_report_from_dict,
    resolve_variable_specs,
)
from .priors import MECHANISM_PRIORS, MechanismPrior, priors_for_target
from .reporting import (
    discovery_report_from_dict,
    render_discovery_markdown,
    save_discovery_report,
)
from .schema import (
    ActiveCandidate,
    DiscoveredLaw,
    DiscoveryReport,
    EvidenceLevel,
    FeatureEffect,
    MatchedPairEffect,
    MechanismHypothesis,
    MediationResult,
    Observation,
    SimulationRequest,
)
from .simulation import DipoleProbeResult, EnergyProfile, PolymerMechanismSimulator
from .symbolic import SymbolicLawMiner


def create_polymer_discovery_campaign(**kwargs):
    return PolymerDiscoveryCampaign(**kwargs)


def _register_model() -> None:
    from ...registry import MODELS

    try:
        MODELS.get("discovery", "polymer_physics")
    except KeyError:
        MODELS.register(
            "discovery",
            "polymer_physics",
            description=(
                "JouleWeave-driven polymer mechanism discovery, active validation, "
                "and mechanism-guided PolyLoom generation"
            ),
        )(create_polymer_discovery_campaign)


_register_model()

barrier_statistics = summarize_barriers
storage_efficiency = charge_discharge_efficiency


__all__ = [
    "AcquisitionBreakdown",
    "ActiveCandidate",
    "ActiveDiscoveryLoop",
    "ActiveLearningConfig",
    "BarrierStatistics",
    "BACKENDS",
    "BackendStatus",
    "CommitteePrediction",
    "CONDUCTIVITY",
    "CURRENT",
    "DatasetRole",
    "DIFFUSIVITY",
    "DIMENSIONLESS",
    "DIPOLE",
    "DielectricEstimate",
    "DiffusionEstimate",
    "DimensionlessGroup",
    "DipoleProbeResult",
    "DiscoveredLaw",
    "DiscoveryReport",
    "DynamicsDiscoveryReport",
    "DynamicsEquation",
    "ELECTRIC_FIELD",
    "ENERGY",
    "EnergyProfile",
    "EvidenceLevel",
    "FeatureEffect",
    "FREQUENCY",
    "FunctionalGroupFeaturizer",
    "HessianInteractionDecomposer",
    "InteractionDecomposition",
    "InteractionEdge",
    "KANOracleDiagnostics",
    "LENGTH",
    "LoadedPolymerPotential",
    "MECHANISM_PRIORS",
    "MASS",
    "MatchedPairEffect",
    "MechanismConstraint",
    "MechanismDiscoveryConfig",
    "MechanismDiscoveryEngine",
    "MechanismGenerationConfig",
    "MechanismGuidedCandidate",
    "MechanismGuidedGenerationResult",
    "MechanismHypothesis",
    "MechanismPrior",
    "MediationResult",
    "NativeSymbolicBackend",
    "Observation",
    "PUBLIC_POLYMER_DATASETS",
    "PolymerDiscoveryCampaign",
    "PolymerFeatureVector",
    "PolymerMechanismSimulator",
    "PolymerPotentialConfig",
    "PotentialCommittee",
    "PotentialPreset",
    "PSEBackend",
    "PhyE2EBackend",
    "PhySOBackend",
    "PIEZOELECTRIC_D",
    "POLARIZATION",
    "PRESSURE",
    "PhysicalDimension",
    "PhysicsEquation",
    "PhysicsKANOracle",
    "PhysicsLearningConfig",
    "PhysicsLearningEngine",
    "PhysicsLearningReport",
    "PySRBackend",
    "QuadraticInteractionOracle",
    "PublicPolymerDataset",
    "SelectedFrame",
    "SimulationRequest",
    "SparseDynamicsDiscoverer",
    "SymbolicLawMiner",
    "TEMPERATURE",
    "TIME",
    "VariableSpec",
    "barrier_statistics",
    "build_polymer_jouleweave_config",
    "buckingham_pi_groups",
    "cohesive_energy_density",
    "charge_discharge_efficiency",
    "create_polymer_discovery_campaign",
    "dielectric_from_dipole_fluctuations",
    "diffusion_from_msd",
    "discover_physical_laws",
    "discover_sparse_dynamics",
    "dynamics_discovery_report_from_dict",
    "discovery_report_from_dict",
    "effective_component_count",
    "feature_matrix",
    "generate_mechanism_guided_polymers",
    "infer_variable_spec",
    "linear_dielectric_energy_density",
    "list_public_polymer_datasets",
    "load_polymer_potential",
    "load_zynnova_polymer_dataset",
    "mean_squared_displacement",
    "observations_from_material_samples",
    "potential_config_snapshot",
    "physics_learning_report_from_dict",
    "priors_for_target",
    "propose_counterfactual_pairs",
    "public_dataset_plan",
    "rank_mechanism_candidates",
    "recoverable_energy_density",
    "render_discovery_markdown",
    "resolve_variable_specs",
    "save_discovery_report",
    "shannon_entropy",
    "storage_efficiency",
    "summarize_barriers",
    "train_polymer_potential",
]
