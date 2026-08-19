"""End-to-end image-to-object, high-fidelity render export, and FEM workflow."""

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
    tetra_quality,
    triangle_quality,
)
from .meshing import tetrahedralize_surface
from .quality import audit_surface
from .registry import OBJECT_BACKENDS
from .repair import repair_surface_for_fem
from .scaling import (
    PhysicalScaleTransform,
    apply_physical_scale,
    compute_physical_scale_transform,
    transform_native_asset,
)
from .schema import ObjectConfig, ObjectRequest
from .types import ObjectResult


def run_object(
    request: ObjectRequest,
    config: ObjectConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
) -> ObjectResult:
    """Generate one canonical render asset and a separate FEM-safe surface/volume.

    The visible/PBR asset is never replaced by a topology-repaired FEM surface.  This
    separation is important for modern image-to-3D models whose UV/PBR detail is much
    richer than the geometry representation required by a tetrahedral solver.
    """

    config = config or ObjectConfig()
    options = dict(config.backend_options)
    options.update(backend_options or {})
    if request.model_id is not None:
        options.setdefault("model_id", request.model_id)
    options.setdefault("device", request.device)
    backend = OBJECT_BACKENDS.choose(request.backend, **options)
    provenance: dict[str, object] = {
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
        "physical_scale_basis": request.physical_scale_basis.value,
    }
    if request.physical_scale_evidence is not None:
        provenance["physical_scale_evidence"] = {
            "filename": request.physical_scale_evidence.name,
            "sha256": sha256_file(request.physical_scale_evidence),
        }
    manifest = RunManifest(
        workflow="zynnova.zynform.image_to_object",
        backend=backend.name,
        configuration={"request": to_jsonable(request), "config": to_jsonable(config)},
        provenance=provenance,
    )
    run_directory = Path(config.output_directory).expanduser().resolve() / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    scaled_native: Path | None = None
    fem_surface: TriangleMesh | None = None
    fem_surface_files: list[Path] = []
    try:
        backend_output = backend.run(request, config, run_directory / "backend")
        render_mesh = backend_output.mesh
        if render_mesh is None:
            assert backend_output.native_mesh is not None
            render_mesh = load_triangle_mesh(backend_output.native_mesh)

        if config.clean_mesh:
            render_mesh, repair_report = clean_triangle_mesh(
                render_mesh, weld_tolerance=config.weld_tolerance
            )
            manifest.event(
                "render_mesh_cleaned",
                output_vertices=repair_report.output_vertices,
                output_faces=repair_report.output_faces,
                watertight=repair_report.watertight,
            )

        transform: PhysicalScaleTransform | None = None
        target_extent = request.physical_extent_m or config.normalize_extent
        if target_extent is not None:
            transform = compute_physical_scale_transform(render_mesh, target_extent)
            render_mesh = apply_physical_scale(render_mesh, transform)
            manifest.event(
                "physical_scale_applied",
                target_extent_m=target_extent,
                basis=request.physical_scale_basis.value,
                scale_factor=transform.scale_factor,
            )
            if backend_output.native_mesh is not None:
                scaled_native = transform_native_asset(
                    backend_output.native_mesh,
                    run_directory / "backend" / "metric_native" / backend_output.native_mesh.name,
                    transform,
                )

        render_quality = triangle_quality(render_mesh)
        render_audit = audit_surface(render_mesh)
        if render_quality.degenerate_faces or render_quality.nonmanifold_edges:
            raise ValueError(
                "processed render surface failed geometry quality gate: "
                f"degenerate_faces={render_quality.degenerate_faces}, "
                f"nonmanifold_edges={render_quality.nonmanifold_edges}"
            )
        surface_quality_path = dump_json(
            run_directory / "quality" / "surface.json", render_audit
        )
        manifest.add_artifact(
            surface_quality_path,
            role="object_surface_quality",
            media_type="application/json",
        )

        volume = None
        volume_quality = None
        if config.generate_fem:
            fem_surface = render_mesh
            fem_surface_quality = triangle_quality(fem_surface)
            if not fem_surface_quality.watertight and config.repair_fem_surface:
                fem_surface = repair_surface_for_fem(fem_surface)
                fem_surface_quality = triangle_quality(fem_surface)
                manifest.event(
                    "fem_surface_repaired",
                    watertight=fem_surface_quality.watertight,
                    boundary_edges=fem_surface_quality.boundary_edges,
                    nonmanifold_edges=fem_surface_quality.nonmanifold_edges,
                )
            if config.fem.require_watertight and not fem_surface_quality.watertight:
                raise ValueError(
                    "FEM surface remains non-watertight after repair: "
                    f"boundary_edges={fem_surface_quality.boundary_edges}, "
                    f"nonmanifold_edges={fem_surface_quality.nonmanifold_edges}"
                )
            if config.export_fem_repair_surface:
                fem_dir = run_directory / "exports" / "fem_surface"
                fem_surface_files.extend(
                    [
                        export_triangle_mesh(fem_dir / "object_fem_surface.ply", fem_surface),
                        export_triangle_mesh(fem_dir / "object_fem_surface.stl", fem_surface),
                    ]
                )
            volume = tetrahedralize_surface(fem_surface, config.fem)
            volume_quality = tetra_quality(volume)
            if not volume_quality.fem_ready:
                raise ValueError(
                    "FEM quality gate failed: "
                    f"inverted={volume_quality.inverted_cells}, "
                    f"degenerate={volume_quality.degenerate_cells}"
                )
            if volume_quality.minimum_mean_ratio < config.fem.minimum_mean_ratio:
                raise ValueError(
                    "FEM quality gate failed: "
                    f"minimum_mean_ratio={volume_quality.minimum_mean_ratio:.6g} < "
                    f"{config.fem.minimum_mean_ratio:.6g}"
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
                backend=volume.metadata.get("backend"),
                minimum_mean_ratio=volume_quality.minimum_mean_ratio,
                fem_ready=volume_quality.fem_ready,
            )

        surface_files = _export_surface(
            render_mesh,
            scaled_native,
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
                    _export_volume(
                        volume_directory / f"object_fem.{fmt}",
                        volume,
                    )
                )

        auxiliary = run_directory / "exports" / "native"
        if backend_output.native_mesh is not None:
            auxiliary.mkdir(parents=True, exist_ok=True)
            raw_native = auxiliary / f"source_asset{backend_output.native_mesh.suffix}"
            shutil.copy2(backend_output.native_mesh, raw_native)
            manifest.add_artifact(raw_native, role="object_native_source")
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
        for path in fem_surface_files:
            manifest.add_artifact(path, role="object_fem_surface")
        for path in volume_files:
            manifest.add_artifact(path, role="object_fem")
        manifest.event(
            "object_completed",
            vertices=render_mesh.n_vertices,
            triangles=render_mesh.n_faces,
            render_watertight=render_quality.watertight,
            tetrahedra=0 if volume is None else volume.n_cells,
            native_pbr_preserved=backend_output.native_mesh is not None,
            metric_native_asset=scaled_native is not None,
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(run_directory / "manifest.json")
        raise
    manifest_path = manifest.save(run_directory / "manifest.json")
    return ObjectResult(
        surface_mesh=render_mesh,
        volume_mesh=volume,
        run_directory=run_directory,
        exported_surface_files=tuple(surface_files),
        exported_volume_files=tuple(volume_files),
        manifest_path=manifest_path,
        fem_surface_mesh=fem_surface,
        exported_fem_surface_files=tuple(fem_surface_files),
        metadata={
            "backend": backend.name,
            "surface_quality": to_jsonable(render_quality),
            "surface_audit": to_jsonable(render_audit),
            "volume_quality": None if volume_quality is None else to_jsonable(volume_quality),
            "native_asset_preserved": backend_output.native_mesh is not None,
            "metric_native_asset": None if scaled_native is None else str(scaled_native),
            "physical_scale": None if transform is None else to_jsonable(transform),
            **backend_output.metadata,
        },
    )


def _export_surface(
    mesh: TriangleMesh,
    metric_native: Path | None,
    directory: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    native_suffix = None if metric_native is None else metric_native.suffix.lower().lstrip(".")
    for fmt in dict.fromkeys(item.lower().lstrip(".") for item in formats):
        target = directory / f"object.{fmt}"
        if metric_native is not None and fmt == native_suffix:
            shutil.copy2(metric_native, target)
            paths.append(target)
        else:
            paths.append(export_triangle_mesh(target, mesh))
    return paths


def _export_volume(path: Path, volume: object) -> Path:
    if path.suffix.lower() == ".mphtxt":
        from ..zynmorph.comsol import export_comsol_mphtxt

        report = export_comsol_mphtxt(path, volume)  # type: ignore[arg-type]
        # The exporter returns a report; its target path is deterministic.
        return path if path.is_file() else Path(report.path)  # type: ignore[attr-defined]
    return export_volume_mesh(path, volume)  # type: ignore[arg-type]


__all__ = ["run_object"]
