"""FEM-ready multi-material Tet4 meshing and quality evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..core.exceptions import GeometryError
from ..geometry import (
    TetraQuality,
    TriangleMesh,
    VolumeMesh,
    VoxelMeshResult,
    export_triangle_mesh,
    export_volume_mesh,
    tetra_quality,
    voxel_to_tetrahedra,
)
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class FEMMeshResult:
    mesh: VolumeMesh
    boundary: TriangleMesh
    interface_faces: Mapping[tuple[int, int], object]
    quality: TetraQuality
    exports: Mapping[str, Path]


def mesh_microstructure(
    volume: MicrostructureVolume,
    *,
    maximum_tetrahedra: int = 12_000_000,
) -> FEMMeshResult:
    tetrahedra = int(volume.labels.size) * 6
    if tetrahedra > maximum_tetrahedra:
        raise GeometryError(
            f"structured mesh would contain {tetrahedra:,} tetrahedra, exceeding "
            f"maximum_tetrahedra={maximum_tetrahedra:,}"
        )
    result: VoxelMeshResult = voxel_to_tetrahedra(
        volume.labels,
        spacing=volume.voxel_size_m,
        origin=volume.origin_m,
        region_names=volume.phase_names,
    )
    quality = tetra_quality(result.volume_mesh)
    if not quality.fem_ready:
        raise GeometryError(
            "generated structured mesh is not FEM-ready: "
            f"inverted={quality.inverted_cells}, degenerate={quality.degenerate_cells}"
        )
    return FEMMeshResult(
        mesh=result.volume_mesh,
        boundary=result.boundary_mesh,
        interface_faces=result.interface_faces,
        quality=quality,
        exports={},
    )


def export_fem_mesh(
    result: FEMMeshResult,
    directory: str | Path,
    *,
    formats: Sequence[str] = ("vtk", "msh", "inp"),
    export_boundary: bool = True,
) -> FEMMeshResult:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for format_name in formats:
        suffix = format_name.lstrip(".")
        path = export_volume_mesh(root / f"microstructure.{suffix}", result.mesh)
        outputs[suffix] = path
    if export_boundary:
        outputs["boundary-ply"] = export_triangle_mesh(root / "boundary.ply", result.boundary)
        outputs["boundary-stl"] = export_triangle_mesh(root / "boundary.stl", result.boundary)
    return FEMMeshResult(
        mesh=result.mesh,
        boundary=result.boundary,
        interface_faces=result.interface_faces,
        quality=result.quality,
        exports=outputs,
    )


__all__ = ["FEMMeshResult", "export_fem_mesh", "mesh_microstructure"]
