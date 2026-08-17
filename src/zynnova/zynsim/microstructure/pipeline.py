"""One-call image-to-voxel-to-FEM-to-COMSOL electrode pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core import GeneralMesh, voxel_to_general_mesh
from ..io.comsol_general import (
    GeneralCOMSOLExportReport,
    LargeVoxelMeshPlan,
    plan_large_voxel_mesh,
    write_general_comsol_mphtxt,
    write_large_voxel_comsol_mphtxt,
)
from .imaging import ImageSegmentationConfig
from .reconstruction import (
    ImageToVoxelConfig,
    ImageToVoxelResult,
    OrthogonalImages,
    reconstruct_electrode_volume,
)


@dataclass(frozen=True, slots=True)
class ImageElectrodeFEMConfig:
    reconstruction: ImageToVoxelConfig = field(default_factory=ImageToVoxelConfig)
    voxel_size_m: float | tuple[float, float, float] = 1.0e-7
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    element_type: str = "hex8"
    phase_names: Mapping[int, str] | None = None
    include_exterior_boundaries: bool = True
    include_material_interfaces: bool = True
    maximum_in_memory_elements: int = 20_000_000
    force_out_of_core: bool = False
    comsol_path: str | Path | None = None
    chunk_size: int = 262_144

    def __post_init__(self) -> None:
        if self.element_type.lower() not in {"hex8", "tet4"}:
            raise ValueError("image voxel meshing supports hex8 or tet4")
        if self.maximum_in_memory_elements < 1 or self.chunk_size < 1:
            raise ValueError("mesh/chunk limits must be positive")


@dataclass(frozen=True, slots=True)
class ImageElectrodeFEMResult:
    reconstruction: ImageToVoxelResult
    mesh_plan: LargeVoxelMeshPlan
    mesh: GeneralMesh | None
    comsol_export: GeneralCOMSOLExportReport | None
    out_of_core: bool


def image_to_electrode_fem(
    images: OrthogonalImages | np.ndarray | str | Path,
    *,
    segmentation: ImageSegmentationConfig | Mapping[str, ImageSegmentationConfig],
    config: ImageElectrodeFEMConfig | None = None,
) -> ImageElectrodeFEMResult:
    """Segment, reconstruct, characterize, mesh, and optionally export COMSOL.

    Large jobs automatically switch to the streaming writer.  In this mode no
    global node or connectivity array is allocated; ``mesh`` is intentionally
    ``None`` and the COMSOL file is the authoritative out-of-core artifact.
    """

    resolved = config or ImageElectrodeFEMConfig()
    reconstruction = reconstruct_electrode_volume(
        images,
        segmentation=segmentation,
        config=resolved.reconstruction,
    )
    plan = plan_large_voxel_mesh(
        reconstruction.phase_labels,
        element_type=resolved.element_type,
        include_exterior_boundaries=resolved.include_exterior_boundaries,
        include_material_interfaces=resolved.include_material_interfaces,
    )
    out_of_core = bool(
        resolved.force_out_of_core
        or plan.volume_element_count > resolved.maximum_in_memory_elements
    )
    mesh: GeneralMesh | None = None
    export: GeneralCOMSOLExportReport | None = None
    if out_of_core:
        if resolved.comsol_path is None:
            raise ValueError("out-of-core image meshing requires comsol_path")
        export = write_large_voxel_comsol_mphtxt(
            resolved.comsol_path,
            reconstruction.phase_labels,
            voxel_size_m=resolved.voxel_size_m,
            origin_m=resolved.origin_m,
            element_type=resolved.element_type,
            phase_names=resolved.phase_names,
            include_exterior_boundaries=resolved.include_exterior_boundaries,
            include_material_interfaces=resolved.include_material_interfaces,
            chunk_size=resolved.chunk_size,
        )
    else:
        mesh = voxel_to_general_mesh(
            reconstruction.phase_labels,
            voxel_size_m=resolved.voxel_size_m,
            origin_m=resolved.origin_m,
            element_type=resolved.element_type,
            include_exterior_boundaries=resolved.include_exterior_boundaries,
            include_material_interfaces=resolved.include_material_interfaces,
        )
        if resolved.comsol_path is not None:
            export = write_general_comsol_mphtxt(resolved.comsol_path, mesh)
    return ImageElectrodeFEMResult(reconstruction, plan, mesh, export, out_of_core)


__all__ = [
    "ImageElectrodeFEMConfig",
    "ImageElectrodeFEMResult",
    "image_to_electrode_fem",
]
