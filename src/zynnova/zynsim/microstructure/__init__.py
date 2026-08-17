"""Characterization-driven generation, evolution, and conformal meshing."""

from .evolution import MicrostructureEvolutionConfig, MicrostructureEvolutionModel
from .generation import (
    CharacterizationDrivenGenerator,
    GeneratedMicrostructure,
    GenerationModel,
    apply_manufacturing_controls,
)
from .imaging import (
    ImageSegmentationConfig,
    PhaseImageReport,
    PhaseThreshold,
    multi_otsu_thresholds,
    read_image,
    segment_electrode_image,
)
from .morphology import (
    PhaseMorphology,
    ZynMorphDescriptor,
    characterize_morphology,
    retarget_phase_fractions,
)
from .pipeline import ImageElectrodeFEMConfig, ImageElectrodeFEMResult, image_to_electrode_fem
from .reconstruction import (
    ImageToVoxelConfig,
    ImageToVoxelResult,
    OrthogonalImages,
    reconstruct_electrode_volume,
)
from .meshing import (
    ConformalMicrostructureMesher,
    InterfaceConformityReport,
    SimulationReadyMicrostructure,
    audit_interface_conformity,
)
from .schema import (
    BatteryPhase,
    DEFAULT_PHASE_NAMES,
    ManufacturingProcessControl,
    validate_phase_labels,
)

__all__ = [
    "BatteryPhase",
    "OrthogonalImages",
    "ImageToVoxelResult",
    "ImageToVoxelConfig",
    "ZynMorphDescriptor",
    "PhaseMorphology",
    "PhaseThreshold",
    "PhaseImageReport",
    "ImageSegmentationConfig",
    "ImageElectrodeFEMConfig",
    "ImageElectrodeFEMResult",
    "CharacterizationDrivenGenerator",
    "ConformalMicrostructureMesher",
    "DEFAULT_PHASE_NAMES",
    "GeneratedMicrostructure",
    "GenerationModel",
    "InterfaceConformityReport",
    "ManufacturingProcessControl",
    "MicrostructureEvolutionConfig",
    "MicrostructureEvolutionModel",
    "SimulationReadyMicrostructure",
    "apply_manufacturing_controls",
    "segment_electrode_image",
    "retarget_phase_fractions",
    "reconstruct_electrode_volume",
    "read_image",
    "multi_otsu_thresholds",
    "characterize_morphology",
    "image_to_electrode_fem",
    "audit_interface_conformity",
    "validate_phase_labels",
]
