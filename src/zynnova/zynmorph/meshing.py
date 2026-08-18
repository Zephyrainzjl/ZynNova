"""FEM-ready multi-material meshing and quality evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from .comsol import export_comsol_mphtxt
from .tetgen import TetGenMeshingConfig, mesh_microstructure_tetgen
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class FEMMeshResult:
    mesh: VolumeMesh
    boundary: TriangleMesh
    interface_faces: Mapping[tuple[int, int], object]
    quality: TetraQuality
    exports: Mapping[str, Path]
    backend: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def mesh_microstructure(
    volume: MicrostructureVolume,
    *,
    method: str = "tetgen",
    tetgen_config: TetGenMeshingConfig | Mapping[str, object] | None = None,
    maximum_tetrahedra: int = 12_000_000,
) -> FEMMeshResult:
    """Mesh a labeled volume with an explicit structured or TetGen backend.

    ``structured`` is the retained compatibility/debug path and splits every
    voxel into six Tet4 cells.  ``tetgen`` is the production adaptive path:
    one conforming smoothed multi-material PLC is passed to the native TetGen
    C++ kernel.  There is deliberately no silent fallback between the two.
    """

    normalized = str(method).strip().lower().replace("_", "-")
    aliases = {
        "voxel": "structured",
        "six-tet": "structured",
        "six-tets-per-voxel": "structured",
        "adaptive": "tetgen",
        "tetgen-1.6": "tetgen",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "tetgen":
        if tetgen_config is None:
            resolved_tetgen = TetGenMeshingConfig()
        elif isinstance(tetgen_config, TetGenMeshingConfig):
            resolved_tetgen = tetgen_config
        else:
            resolved_tetgen = TetGenMeshingConfig(**dict(tetgen_config))
        adaptive = mesh_microstructure_tetgen(
            volume,
            config=resolved_tetgen,
            maximum_tetrahedra=maximum_tetrahedra,
        )
        return FEMMeshResult(
            mesh=adaptive.mesh,
            boundary=adaptive.boundary,
            interface_faces=adaptive.interface_faces,
            quality=adaptive.quality,
            exports={},
            backend="tetgen-1.6.0-adaptive-plc",
            metadata={
                **adaptive.metadata,
                "switches": adaptive.switches,
                "input_plc_audit": adaptive.input_plc_audit,
                "junction_report": adaptive.junction_report,
                "region_seed_count": len(adaptive.region_seeds),
                "meshed_volume": adaptive.meshed_volume,
                "input_plc": adaptive.input_plc,
                "output_surface": adaptive.output_surface,
            },
        )
    if normalized != "structured":
        raise ValueError("method must be 'structured' or 'tetgen'")

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
        backend="structured-six-tets-per-voxel",
        metadata={"adaptive": False, "voxel_shape_zyx": volume.shape},
    )


def export_fem_mesh(
    result: FEMMeshResult,
    directory: str | Path,
    *,
    formats: Sequence[str] = ("vtk", "msh", "inp"),
    export_boundary: bool = True,
    comsol_domain_selections: Mapping[str, Iterable[int]] | None = None,
    comsol_boundary_selections: Mapping[str, Iterable[str]] | None = None,
    comsol_options: Mapping[str, object] | None = None,
) -> FEMMeshResult:
    """Export one FEM mesh, including native COMSOL MPHTXT when requested."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for format_name in formats:
        suffix = str(format_name).lower().lstrip(".")
        if suffix == "comsol":
            suffix = "mphtxt"
        if suffix == "mphtxt":
            options = dict(comsol_options or {})
            report = export_comsol_mphtxt(
                root / "microstructure.mphtxt",
                result.mesh,
                domain_selections=comsol_domain_selections,
                boundary_selections=comsol_boundary_selections,
                **options,
            )
            outputs["mphtxt"] = report.path
            continue
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
        backend=result.backend,
        metadata=result.metadata,
    )


__all__ = ["FEMMeshResult", "export_fem_mesh", "mesh_microstructure"]
