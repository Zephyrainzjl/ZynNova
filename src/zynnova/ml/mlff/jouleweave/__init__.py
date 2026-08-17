"""JouleWeave: smooth Cartesian-field neural interatomic potentials."""

from ...registry import MODELS
from .active_learning import (
    ActiveLearningCampaign,
    ActiveLearningConfig,
    ActiveLearningCycleRecord,
    ActiveLearningResult,
    QueryCandidate,
    StructureSampler,
)
from .closed_loop import (
    LowScaleCampaignPoint,
    LowScaleCampaignResult,
    LowScaleClosedLoop,
)
from .dft_oracle import (
    CallableDFTOracle,
    DFTOracle,
    DFTReferenceResult,
    DFTUncertaintyAssessment,
    RedundantDFTOracle,
)
from .electrochemical_calculator import (
    constant_potential_input_adapter,
    constant_potential_jouleweave_calculator,
    load_constant_potential_checkpoint,
)
from .electrochemical_training import (
    ConstantPotentialTrainConfig,
    constant_potential_loss,
    train_constant_potential,
)
from .grand_canonical import ConstantPotentialJouleWeave, GrandCanonicalConfig
from .barrier import (
    MigrationBarrierConfig,
    MigrationBarrierModel,
    MigrationBarrierPrediction,
    migration_barrier_loss,
)
from .calculator import (
    jouleweave_calculator,
    load_jouleweave,
    load_jouleweave_calculator,
)
from .cathode import (
    CathodeCyclingConfig,
    CathodeCyclingResult,
    CathodeCyclingWorkflow,
    CathodePhaseRecord,
    VoltageStep,
)
from .config import (
    ChargeLabelScheme,
    JouleWeaveConfig,
    JouleWeaveDataConfig,
    JouleWeaveModelConfig,
    JouleWeaveTrainConfig,
)
from .data import (
    fit_energy_statistics,
    jouleweave_collate,
    jouleweave_task,
    prepare_jouleweave_datamodule,
)
from .electronic import (
    BaderRunner,
    ChargeOxidationCalibrator,
    ChargePartitionLabels,
    OxidationStateAssignment,
    OxidationStateResolver,
    attach_partition_labels,
    read_bader_acf,
    read_ddec_charges,
)
from .graph import build_periodic_radius_graph
from .lammps_mliap import (
    JouleWeaveMLIAP,
    export_jouleweave_checkpoint_mliap,
    export_jouleweave_mliap,
    jouleweave_mliap_commands,
    load_jouleweave_mliap,
)
from .materials import (
    ElasticityResult,
    EquationOfStateResult,
    JouleWeaveMaterials,
    PhononResult,
    calculate_jouleweave_elasticity,
    calculate_jouleweave_phonons,
    fit_jouleweave_eos,
    optimize_jouleweave_structure,
)
from .model import JouleWeave
from .ncm import (
    NCMCompositionEnumerator,
    NCMConfiguration,
    NCMEnumerationConfig,
    ncm_mixing_statistics,
)
from .parameter_extraction import (
    AtomisticExtractionInput,
    AtomisticParameterSurface,
    AutomaticParameterExtractor,
    ParameterEstimate,
    arrhenius_fit,
    cation_transference_number,
    diffusion_from_msd,
    exchange_current_density,
    nernst_einstein_conductivity,
    reaction_rate_from_barrier,
)
from .rare_events import (
    DimerConfig,
    DimerResult,
    JouleWeaveDimer,
    JouleWeaveMetadynamics,
    JouleWeaveNEB,
    MetadynamicsConfig,
    MetadynamicsResult,
    NEBConfig,
    NEBResult,
    well_tempered_metadynamics_input,
)
from .trainer import train_jouleweave
from .uncertainty import (
    CommitteePrediction,
    ConformalUncertaintyCalibrator,
    JouleWeaveCommittee,
)


@MODELS.register(
    "mlff",
    "jouleweave",
    description=(
        "Smooth O(3)-equivariant Cartesian-field potential with physical priors "
        "and a distributed LAMMPS ML-IAP adapter"
    ),
)
def create_jouleweave(
    config: JouleWeaveModelConfig | None = None,
) -> JouleWeave:
    return JouleWeave(config)


__all__ = [
    "ActiveLearningCampaign",
    "ActiveLearningConfig",
    "ActiveLearningCycleRecord",
    "ActiveLearningResult",
    "AtomisticExtractionInput",
    "AtomisticParameterSurface",
    "AutomaticParameterExtractor",
    "CallableDFTOracle",
    "CommitteePrediction",
    "ConformalUncertaintyCalibrator",
    "ConstantPotentialJouleWeave",
    "ConstantPotentialTrainConfig",
    "DFTOracle",
    "DFTReferenceResult",
    "DFTUncertaintyAssessment",
    "GrandCanonicalConfig",
    "JouleWeaveCommittee",
    "LowScaleCampaignPoint",
    "LowScaleCampaignResult",
    "LowScaleClosedLoop",
    "ParameterEstimate",
    "QueryCandidate",
    "RedundantDFTOracle",
    "StructureSampler",
    "JouleWeave",
    "BaderRunner",
    "CathodeCyclingConfig",
    "CathodeCyclingResult",
    "CathodeCyclingWorkflow",
    "CathodePhaseRecord",
    "ChargeLabelScheme",
    "ChargeOxidationCalibrator",
    "ChargePartitionLabels",
    "DimerConfig",
    "DimerResult",
    "JouleWeaveConfig",
    "JouleWeaveDataConfig",
    "JouleWeaveMaterials",
    "JouleWeaveMLIAP",
    "JouleWeaveDimer",
    "JouleWeaveMetadynamics",
    "JouleWeaveNEB",
    "JouleWeaveModelConfig",
    "JouleWeaveTrainConfig",
    "MetadynamicsConfig",
    "MetadynamicsResult",
    "MigrationBarrierConfig",
    "MigrationBarrierModel",
    "MigrationBarrierPrediction",
    "NCMCompositionEnumerator",
    "NCMConfiguration",
    "NCMEnumerationConfig",
    "NEBConfig",
    "NEBResult",
    "OxidationStateAssignment",
    "OxidationStateResolver",
    "ElasticityResult",
    "EquationOfStateResult",
    "PhononResult",
    "build_periodic_radius_graph",
    "calculate_jouleweave_elasticity",
    "calculate_jouleweave_phonons",
    "attach_partition_labels",
    "create_jouleweave",
    "export_jouleweave_checkpoint_mliap",
    "export_jouleweave_mliap",
    "fit_energy_statistics",
    "fit_jouleweave_eos",
    "jouleweave_calculator",
    "jouleweave_collate",
    "jouleweave_mliap_commands",
    "jouleweave_task",
    "load_constant_potential_checkpoint",
    "load_jouleweave",
    "load_jouleweave_calculator",
    "load_jouleweave_mliap",
    "migration_barrier_loss",
    "ncm_mixing_statistics",
    "optimize_jouleweave_structure",
    "prepare_jouleweave_datamodule",
    "read_bader_acf",
    "read_ddec_charges",
    "train_jouleweave",
    "arrhenius_fit",
    "cation_transference_number",
    "constant_potential_input_adapter",
    "constant_potential_jouleweave_calculator",
    "constant_potential_loss",
    "diffusion_from_msd",
    "exchange_current_density",
    "nernst_einstein_conductivity",
    "reaction_rate_from_barrier",
    "train_constant_potential",
    "VoltageStep",
    "well_tempered_metadynamics_input",
]
