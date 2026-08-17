"""Electrochemical-thermal-mechanical-damage-aging coupling."""

from .constitutive import CoupledConstitutiveModel, CoupledMaterialConfig
from .damage import DamageConfig, DamageModel
from .gpu import GPUBackendConfig, GPUFieldBackend
from .industrial import (
    CohesiveZoneParameters,
    CohesiveZoneState,
    GeneralizedMaxwellModel,
    GeneralizedMaxwellState,
    J2PlasticityModel,
    J2PlasticityParameters,
    J2PlasticityState,
    MaxwellBranch,
    MixedModeCohesiveZone,
    ThermalReaction,
    ThermalRunawayKinetics,
    ThermalRunawayState,
)
from .monolithic import (
    FieldLayout,
    FieldSlice,
    MonolithicIteration,
    MonolithicNewtonKrylovConfig,
    MonolithicNewtonKrylovSolver,
    MonolithicSolution,
)
from .solver import CoupledSolverConfig, ElectrochemicalModel, FullyCoupledBatterySolver
from .state import (
    CoupledBatteryState,
    CoupledTrajectory,
    CouplingDiagnostics,
    infer_state_observables,
)
from .switching import (
    AdaptiveFidelityModel,
    AdaptiveFidelityPolicy,
    FidelityLevel,
    FidelityMetrics,
)

__all__ = [
    "AdaptiveFidelityModel",
    "MonolithicSolution",
    "MonolithicNewtonKrylovSolver",
    "MonolithicNewtonKrylovConfig",
    "MonolithicIteration",
    "FieldSlice",
    "FieldLayout",
    "ThermalRunawayState",
    "ThermalRunawayKinetics",
    "ThermalReaction",
    "MixedModeCohesiveZone",
    "MaxwellBranch",
    "J2PlasticityState",
    "J2PlasticityParameters",
    "J2PlasticityModel",
    "GeneralizedMaxwellState",
    "GeneralizedMaxwellModel",
    "CohesiveZoneState",
    "CohesiveZoneParameters",
    "AdaptiveFidelityPolicy",
    "CoupledBatteryState",
    "CoupledConstitutiveModel",
    "CoupledMaterialConfig",
    "CoupledSolverConfig",
    "CoupledTrajectory",
    "CouplingDiagnostics",
    "DamageConfig",
    "DamageModel",
    "ElectrochemicalModel",
    "FidelityLevel",
    "FidelityMetrics",
    "FullyCoupledBatterySolver",
    "GPUBackendConfig",
    "GPUFieldBackend",
    "infer_state_observables",
]
