"""FEM-ready multi-material meshing and quality evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
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


_COMSOL_TET_OPTION_ALIASES = {
    "include_boundary_triangles": "include_boundaries",
    "include_interface_triangles": "include_internal_interfaces",
    "include_material_interfaces": "include_internal_interfaces",
    "include_exterior_triangles": "include_exterior",
    "include_exterior_boundaries": "include_exterior",
}


def _normalize_comsol_tet_options(
    options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize historical notebook keywords and reject accidental typos."""

    normalized = dict(options or {})
    for alias, canonical in _COMSOL_TET_OPTION_ALIASES.items():
        if alias not in normalized:
            continue
        value = normalized.pop(alias)
        if canonical in normalized and bool(normalized[canonical]) != bool(value):
            raise ValueError(
                f"conflicting COMSOL options: {alias} and {canonical}"
            )
        normalized[canonical] = value

    allowed = set(inspect.signature(export_comsol_mphtxt).parameters) - {"path", "mesh"}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise TypeError(
            "unsupported COMSOL Tet4 export option(s): "
            f"{unknown}; supported options are {sorted(allowed)}"
        )
    return normalized


def mesh_microstructure(
    volume: MicrostructureVolume,
    *,
    method: str = "tetgen",
    tetgen_config: TetGenMeshingConfig | Mapping[str, object] | None = None,
    material_region_map: Mapping[int, int] | None = None,
    material_region_names: Mapping[int, str] | None = None,
    require_complete_region_map: bool = False,
    maximum_tetrahedra: int = 12_000_000,
) -> FEMMeshResult:
    """Mesh a labeled volume with an explicit structured or TetGen backend.

    ``structured`` is the retained compatibility/debug path and splits every
    voxel into six Tet4 cells.  ``tetgen`` is the production adaptive path:
    one conforming smoothed multi-material PLC is passed to the native TetGen
    C++ kernel.  There is deliberately no silent fallback between the two.
    """

    source_volume = volume
    if material_region_map:
        volume = volume.remap_regions(
            material_region_map,
            region_names=material_region_names,
            require_complete=require_complete_region_map,
        )

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
                "region_seeds": adaptive.region_seeds,
                "meshed_volume": adaptive.meshed_volume,
                "input_plc": adaptive.input_plc,
                "output_surface": adaptive.output_surface,
                "material_region_map": None
                if not material_region_map
                else {int(k): int(v) for k, v in material_region_map.items()},
                "source_phases": source_volume.phases,
                "meshed_phases": volume.phases,
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
        metadata={
            "adaptive": False,
            "voxel_shape_zyx": volume.shape,
            "material_region_map": None
            if not material_region_map
            else {int(k): int(v) for k, v in material_region_map.items()},
            "source_phases": source_volume.phases,
            "meshed_phases": volume.phases,
        },
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
            options = _normalize_comsol_tet_options(comsol_options)
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



def mesh_freeform_geometry(
    plc_or_shells,
    regions,
    *,
    tetgen_config=None,
    holes_m_xyz=(),
    void_regions=(),
    maximum_tetrahedra: int = 20_000_000,
    weld_tolerance_m: float | None = None,
) -> FEMMeshResult:
    """Mesh arbitrary closed surfaces/internal interfaces with native TetGen.

    This path makes no rectangular/voxel assumption.  ``plc_or_shells`` may be
    a preassembled ``MultiphasePLC`` or a sequence of ``SurfaceShell`` objects.
    """

    from .freeform import mesh_freeform_tetgen

    adaptive = mesh_freeform_tetgen(
        plc_or_shells,
        regions,
        config=tetgen_config,
        holes_m_xyz=holes_m_xyz,
        void_regions=void_regions,
        maximum_tetrahedra=maximum_tetrahedra,
        weld_tolerance_m=weld_tolerance_m,
    )
    return FEMMeshResult(
        mesh=adaptive.mesh,
        boundary=adaptive.boundary,
        interface_faces=adaptive.interface_faces,
        quality=adaptive.quality,
        exports={},
        backend="tetgen-1.6.0-freeform-plc",
        metadata={
            **adaptive.metadata,
            "switches": adaptive.switches,
            "input_plc_audit": adaptive.plc_audit,
            "input_plc": adaptive.plc,
            "output_surface": adaptive.output_surface,
        },
    )



def mesh_unstructured_regions(
    geometry,
    *,
    regions=None,
    tetgen_config: TetGenMeshingConfig | Mapping[str, object] | None = None,
    material_region_map: Mapping[int, int] | None = None,
    material_region_names: Mapping[int, str] | None = None,
    require_complete_region_map: bool = False,
    holes_m_xyz=(),
    void_regions=(),
    maximum_tetrahedra: int = 20_000_000,
    weld_tolerance_m: float | None = None,
) -> FEMMeshResult:
    """Universal production entry point for nonstructured multi-region Tet4.

    ``geometry`` may be a labeled :class:`MicrostructureVolume`, an assembled
    multi-material PLC, or a sequence of closed free-form shells.  The function
    never falls back to structured voxel tets.  Disconnected components can
    share one material region ID; TetGen receives one interior seed per
    component while COMSOL receives one domain entity per material ID.
    """

    if isinstance(geometry, MicrostructureVolume):
        if regions is not None:
            raise ValueError("regions must be omitted for a MicrostructureVolume")
        return mesh_microstructure(
            geometry,
            method="tetgen",
            tetgen_config=tetgen_config,
            material_region_map=material_region_map,
            material_region_names=material_region_names,
            require_complete_region_map=require_complete_region_map,
            maximum_tetrahedra=maximum_tetrahedra,
        )

    if material_region_map or material_region_names:
        raise ValueError(
            "material_region_map/material_region_names apply only to labeled volumes; "
            "free-form PLC/shell inputs must already use final material region IDs"
        )
    if regions is None:
        raise ValueError("regions are required for free-form PLC/shell geometry")
    if tetgen_config is None or isinstance(tetgen_config, TetGenMeshingConfig):
        resolved = tetgen_config
    else:
        resolved = TetGenMeshingConfig(**dict(tetgen_config))
    return mesh_freeform_geometry(
        geometry,
        regions,
        tetgen_config=resolved,
        holes_m_xyz=holes_m_xyz,
        void_regions=void_regions,
        maximum_tetrahedra=maximum_tetrahedra,
        weld_tolerance_m=weld_tolerance_m,
    )


def mesh_freeform_like_reference(
    plc_or_shells,
    regions,
    reference_mesh,
    *,
    region_map: Mapping[int, int] | None = None,
    volume_quantile: float = 0.95,
    linear_scale: float = 1.0,
    tetgen_config: TetGenMeshingConfig | Mapping[str, object] | None = None,
    holes_m_xyz=(),
    void_regions=(),
    maximum_tetrahedra: int = 20_000_000,
    weld_tolerance_m: float | None = None,
) -> FEMMeshResult:
    """Mesh arbitrary geometry using the size statistics of a reference Tet4 mesh.

    The reference contributes *only* global/per-domain tetrahedron scale.  Its
    box, boundary topology and coordinates are never copied to the new geometry.
    This is useful for reproducing the spatial density of an existing COMSOL
    unstructured mesh on a completely different particle, pore or CAD surface.
    """

    from dataclasses import asdict

    from .reference_mesh import profile_reference_mesh, tetgen_config_from_reference

    if tetgen_config is None:
        base = TetGenMeshingConfig()
    elif isinstance(tetgen_config, TetGenMeshingConfig):
        base = tetgen_config
    else:
        base = TetGenMeshingConfig(**dict(tetgen_config))

    profile = profile_reference_mesh(reference_mesh)
    resolved = tetgen_config_from_reference(
        profile,
        region_map=region_map,
        volume_quantile=volume_quantile,
        linear_scale=linear_scale,
        base=base,
    )
    result = mesh_freeform_geometry(
        plc_or_shells,
        regions,
        tetgen_config=resolved,
        holes_m_xyz=holes_m_xyz,
        void_regions=void_regions,
        maximum_tetrahedra=maximum_tetrahedra,
        weld_tolerance_m=weld_tolerance_m,
    )
    return FEMMeshResult(
        mesh=result.mesh,
        boundary=result.boundary,
        interface_faces=result.interface_faces,
        quality=result.quality,
        exports=result.exports,
        backend=result.backend,
        metadata={
            **result.metadata,
            "reference_mesh_style": {
                "volume_quantile": float(volume_quantile),
                "linear_scale": float(linear_scale),
                "region_map": None if region_map is None else {int(k): int(v) for k, v in region_map.items()},
                "profile": asdict(profile),
            },
        },
    )


__all__ = [
    "FEMMeshResult",
    "export_fem_mesh",
    "mesh_freeform_geometry",
    "mesh_freeform_like_reference",
    "mesh_microstructure",
    "mesh_unstructured_regions",
]
