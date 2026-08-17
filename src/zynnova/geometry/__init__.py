"""Shared geometry data, repair, quality, meshing, and export."""

from .importing import load_triangle_mesh
from .export import export_point_cloud, export_triangle_mesh, export_volume_mesh
from .quality import (
    TetraQuality,
    TriangleQuality,
    edge_incidence,
    tetra_quality,
    tetrahedron_mean_ratio,
    tetrahedron_signed_volumes,
    triangle_areas,
    triangle_quality,
)
from .repair import RepairReport, clean_triangle_mesh, normalize_mesh
from .types import (
    Camera,
    GaussianSplat,
    PointCloud,
    SceneAsset,
    SceneBundle,
    TriangleMesh,
    VolumeMesh,
)
from .voxel import (
    VoxelMeshResult,
    extract_material_surfaces,
    select_volume_regions,
    voxel_to_tetrahedra,
)

__all__ = [
    "Camera",
    "GaussianSplat",
    "PointCloud",
    "RepairReport",
    "SceneAsset",
    "SceneBundle",
    "TetraQuality",
    "TriangleMesh",
    "TriangleQuality",
    "VolumeMesh",
    "VoxelMeshResult",
    "clean_triangle_mesh",
    "edge_incidence",
    "tetrahedron_mean_ratio",
    "tetrahedron_signed_volumes",
    "triangle_areas",
    "export_point_cloud",
    "export_triangle_mesh",
    "export_volume_mesh",
    "extract_material_surfaces",
    "normalize_mesh",
    "select_volume_regions",
    "tetra_quality",
    "triangle_quality",
    "voxel_to_tetrahedra",
    "load_triangle_mesh",
]
