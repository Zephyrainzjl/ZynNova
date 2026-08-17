"""End-to-end image-to-object, cross-format export, and FEM workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from ..core import RunManifest, dump_json, sha256_file
from ..core.serialization import to_jsonable
from ..geometry import (
    TriangleMesh,
    clean_triangle_mesh,
    export_triangle_mesh,
    export_volume_mesh,
    load_triangle_mesh,
    normalize_mesh,
    tetra_quality,
    triangle_quality,
)
from .meshing import tetrahedralize_surface
from .registry import OBJECT_BACKENDS
from .repair import repair_surface_for_fem
from .schema import ObjectConfig, ObjectRequest
from .types import ObjectResult


def run_object(
    request: ObjectRequest,
    config: ObjectConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
) -> ObjectResult:
    """Generate, normalize, repair, export, and optionally tetrahedralize an object."""

    config = config or ObjectConfig()
    options = dict(config.backend_options)
    options.update(backend_options or {})
    if request.model_id is not None:
        options.setdefault("model_id", request.model_id)
    options.setdefault("device", request.device)
    backend = OBJECT_BACKENDS.choose(request.backend, **options)
    manifest = RunManifest(
        workflow="zynnova.zynform.image_to_object",
        backend=backend.name,
        configuration={"request": to_jsonable(request), "config": to_jsonable(config)},
        provenance={
            "input_image": {
                "path": str(request.image.resolve()),
                "sha256": sha256_file(request.image),
            },
            "foreground_mask": None
            if request.foreground_mask is None
            else {
                "path": str(request.foreground_mask.resolve()),
                "sha256": sha256_file(request.foreground_mask),
            },
        },
    )
    run_directory = Path(config.output_directory).expanduser().resolve() / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    try:
        backend_output = backend.run(request, config, run_directory / "backend")
        mesh = backend_output.mesh
        if mesh is None:
            assert backend_output.native_mesh is not None
            mesh = load_triangle_mesh(backend_output.native_mesh)
        if config.clean_mesh:
            mesh, repair_report = clean_triangle_mesh(
                mesh, weld_tolerance=config.weld_tolerance
            )
            manifest.event(
                "mesh_cleaned",
                output_vertices=repair_report.output_vertices,
                output_faces=repair_report.output_faces,
                watertight=repair_report.watertight,
            )
        target_extent = request.physical_extent_m or config.normalize_extent
        if target_extent is not None:
            mesh = normalize_mesh(mesh, target_extent=target_extent)
        surface_quality = triangle_quality(mesh)
        if surface_quality.degenerate_faces or surface_quality.nonmanifold_edges:
            raise ValueError(
                "processed surface failed geometry quality gate: "
                f"degenerate_faces={surface_quality.degenerate_faces}, "
                f"nonmanifold_edges={surface_quality.nonmanifold_edges}"
            )
        surface_quality_path = dump_json(
            run_directory / "quality" / "surface.json", surface_quality
        )
        manifest.add_artifact(
            surface_quality_path,
            role="object_surface_quality",
            media_type="application/json",
        )
        volume = None
        volume_quality = None
        if config.generate_fem:
            fem_surface = mesh
            if not surface_quality.watertight and config.clean_mesh:
                fem_surface = repair_surface_for_fem(mesh)
                surface_quality = triangle_quality(fem_surface)
                if surface_quality.watertight:
                    mesh = fem_surface
            volume = tetrahedralize_surface(fem_surface, config.fem)
            volume_quality = tetra_quality(volume)
            if not volume_quality.fem_ready:
                raise ValueError(
                    "FEM quality gate failed: "
                    f"inverted={volume_quality.inverted_cells}, "
                    f"degenerate={volume_quality.degenerate_cells}"
                )
            volume_quality_path = dump_json(
                run_directory / "quality" / "volume.json", volume_quality
            )
            manifest.add_artifact(
                volume_quality_path,
                role="object_fem_quality",
                media_type="application/json",
            )
            manifest.event(
                "fem_generated",
                cells=volume.n_cells,
                meshing_method=volume.metadata.get("meshing_method"),
                minimum_mean_ratio=volume_quality.minimum_mean_ratio,
                fem_ready=volume_quality.fem_ready,
            )
        surface_files = _export_surface(
            mesh,
            backend_output.native_mesh,
            run_directory / "exports" / "surface",
            config.export_formats,
        )
        volume_files: list[Path] = []
        if volume is not None:
            volume_directory = run_directory / "exports" / "volume"
            for fmt in dict.fromkeys(
                item.lower().lstrip(".") for item in config.fem_export_formats
            ):
                volume_files.append(
                    export_volume_mesh(volume_directory / f"object_fem.{fmt}", volume)
                )
        auxiliary = run_directory / "exports" / "native"
        for role, source in backend_output.auxiliary_assets.items():
            auxiliary.mkdir(parents=True, exist_ok=True)
            destination = auxiliary / f"{role}{source.suffix}"
            shutil.copy2(source, destination)
            manifest.add_artifact(destination, role=f"object_{role}")
        if backend_output.preview is not None:
            auxiliary.mkdir(parents=True, exist_ok=True)
            destination = auxiliary / f"preview{backend_output.preview.suffix}"
            shutil.copy2(backend_output.preview, destination)
            manifest.add_artifact(destination, role="object_preview")
        for path in surface_files:
            manifest.add_artifact(path, role="object_surface")
        for path in volume_files:
            manifest.add_artifact(path, role="object_fem")
        manifest.event(
            "object_completed",
            vertices=int(mesh.vertices.shape[0]),
            triangles=int(mesh.faces.shape[0]),
            watertight=surface_quality.watertight,
            tetrahedra=0 if volume is None else int(volume.n_cells),
            native_pbr_preserved=backend_output.native_mesh is not None,
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(run_directory / "manifest.json")
        raise
    manifest_path = manifest.save(run_directory / "manifest.json")
    return ObjectResult(
        surface_mesh=mesh,
        volume_mesh=volume,
        run_directory=run_directory,
        exported_surface_files=tuple(surface_files),
        exported_volume_files=tuple(volume_files),
        manifest_path=manifest_path,
        metadata={
            "backend": backend.name,
            "surface_quality": to_jsonable(surface_quality),
            "volume_quality": None if volume_quality is None else to_jsonable(volume_quality),
            "native_asset_preserved": backend_output.native_mesh is not None,
            **backend_output.metadata,
        },
    )


def _export_surface(
    mesh: TriangleMesh,
    native: Path | None,
    directory: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    native_suffix = None if native is None else native.suffix.lower().lstrip(".")
    for fmt in dict.fromkeys(item.lower().lstrip(".") for item in formats):
        target = directory / f"object.{fmt}"
        if native is not None and fmt == native_suffix:
            shutil.copy2(native, target)
            paths.append(target)
        else:
            paths.append(export_triangle_mesh(target, mesh))
    if native is not None and native_suffix not in {
        item.lower().lstrip(".") for item in formats
    }:
        target = directory / f"object_native{native.suffix}"
        shutil.copy2(native, target)
        paths.append(target)
    return paths


__all__ = ["run_object"]
