"""Geometry, boundary-condition, linear-algebra, and result primitives."""

from .boundary import DirichletBC, SurfaceLoad
from .general_mesh import ElementBlock, GeneralMesh, MeshQualityReport, voxel_to_general_mesh
from .linalg import LinearSolveOptions, apply_dirichlet, solve_linear
from .mesh import Mesh, box_tetrahedral_mesh
from .results import FieldResult, SolverDiagnostics, TimeSeriesResult
from .voxel import VoxelMeshResult, voxel_interface_areas, voxel_to_tetrahedral_mesh
from .voxel_reconstruction import (
    TetMeshQualitySummary,
    VoxelFEMReconstructionConfig,
    VoxelFEMReconstructionReport,
    VoxelFEMReconstructionResult,
    reconstruct_voxel_fem_mesh,
    tetrahedron_mean_ratio,
    tetrahedron_signed_six_volumes,
    voxel_interface_faces,
    voxel_to_fem_mesh,
)

__all__ = [
    "DirichletBC",
    "ElementBlock",
    "FieldResult",
    "GeneralMesh",
    "MeshQualityReport",
    "LinearSolveOptions",
    "Mesh",
    "SolverDiagnostics",
    "SurfaceLoad",
    "TimeSeriesResult",
    "TetMeshQualitySummary",
    "VoxelFEMReconstructionConfig",
    "VoxelFEMReconstructionReport",
    "VoxelFEMReconstructionResult",
    "VoxelMeshResult",
    "apply_dirichlet",
    "box_tetrahedral_mesh",
    "reconstruct_voxel_fem_mesh",
    "solve_linear",
    "tetrahedron_mean_ratio",
    "tetrahedron_signed_six_volumes",
    "voxel_interface_areas",
    "voxel_interface_faces",
    "voxel_to_fem_mesh",
    "voxel_to_general_mesh",
    "voxel_to_tetrahedral_mesh",
]
