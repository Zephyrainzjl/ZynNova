"""ZynMorph: conditional battery microstructures and FEM-ready meshes."""

from .comsol import (
    COMSOLMPHTXTInfo,
    COMSOLMeshExportReport,
    GeneralCOMSOLExportReport,
    HexTopologyAudit,
    LargeVoxelMeshPlan,
    audit_comsol_hex8_connectivity,
    audit_comsol_hex8_topology,
    default_battery_domain_selections,
    default_coordinate_boundary_unions,
    export_comsol_mphtxt,
    export_voxel_comsol_mphtxt,
    inspect_comsol_mphtxt,
    plan_voxel_comsol_mphtxt,
    to_comsol_tet_mesh,
)
from .generation import GenerationResult, SpectralConditionalGenerator
from .meshing import FEMMeshResult, export_fem_mesh, mesh_microstructure
from .metrics import MicrostructureMetrics, PhaseMetrics, analyze_microstructure
from .pipeline import ZynMorphRun, run_zynmorph
from .reconstruction import SliceObservation, reconstruct_from_slices
from .registry import BACKENDS, PUBLIC_REFERENCE_BACKENDS
from .schema import BatteryPhase, GenerationConfig, MicrostructureCondition
from .surface import (
    JunctionRegularizationReport,
    MultiphasePLC,
    SurfacePLCAudit,
    audit_multiphase_plc,
    count_nonmanifold_voxel_edges,
    extract_multiphase_plc,
    regularize_nonmanifold_junctions,
    smooth_multiphase_plc,
)
from .tetgen import (
    LocalRefinementZone,
    RegionSeed,
    TetGenMeshResult,
    TetGenMeshingConfig,
    TetGenNativeStatus,
    build_region_seeds,
    mesh_microstructure_tetgen,
    tetgen_native_status,
)
from .volume import MicrostructureVolume

__all__ = [
    "BACKENDS",
    "PUBLIC_REFERENCE_BACKENDS",
    "BatteryPhase",
    "COMSOLMPHTXTInfo",
    "COMSOLMeshExportReport",
    "FEMMeshResult",
    "GeneralCOMSOLExportReport",
    "HexTopologyAudit",
    "LargeVoxelMeshPlan",
    "GenerationConfig",
    "GenerationResult",
    "MicrostructureCondition",
    "MicrostructureMetrics",
    "MicrostructureVolume",
    "PhaseMetrics",
    "SliceObservation",
    "SpectralConditionalGenerator",
    "ZynMorphRun",
    "analyze_microstructure",
    "audit_comsol_hex8_connectivity",
    "audit_comsol_hex8_topology",
    "default_battery_domain_selections",
    "default_coordinate_boundary_unions",
    "export_comsol_mphtxt",
    "export_fem_mesh",
    "export_voxel_comsol_mphtxt",
    "inspect_comsol_mphtxt",
    "mesh_microstructure",
    "plan_voxel_comsol_mphtxt",
    "reconstruct_from_slices",
    "run_zynmorph",
    "to_comsol_tet_mesh",
    "JunctionRegularizationReport",
    "LocalRefinementZone",
    "MultiphasePLC",
    "RegionSeed",
    "SurfacePLCAudit",
    "TetGenMeshResult",
    "TetGenMeshingConfig",
    "TetGenNativeStatus",
    "audit_multiphase_plc",
    "build_region_seeds",
    "count_nonmanifold_voxel_edges",
    "extract_multiphase_plc",
    "mesh_microstructure_tetgen",
    "regularize_nonmanifold_junctions",
    "smooth_multiphase_plc",
    "tetgen_native_status",
]
