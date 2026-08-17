"""End-to-end ZynVista orchestration with reproducibility manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..core import RunManifest, sha256_file
from ..geometry import load_triangle_mesh
from ..core.serialization import to_jsonable
from .export import export_scene_output
from .fusion import dense_views_to_mesh, fuse_dense_views
from .registry import GENERATION_BACKENDS, RECONSTRUCTION_BACKENDS
from .schema import SceneConfig, SceneMode, SceneRequest
from .styles import STYLE_BACKENDS
from .types import SceneBackendOutput


@dataclass(frozen=True, slots=True)
class SceneResult:
    output: SceneBackendOutput
    run_directory: Path
    exported_files: tuple[Path, ...]
    manifest_path: Path


def run_scene(
    request: SceneRequest,
    config: SceneConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
    style_options: Mapping[str, object] | None = None,
) -> SceneResult:
    """Run reconstruction/generation, optional styling, export, and audit logging."""

    config = config or SceneConfig()
    base = Path(config.output_directory).expanduser().resolve()
    registry = (
        RECONSTRUCTION_BACKENDS
        if request.mode == SceneMode.RECONSTRUCT
        else GENERATION_BACKENDS
    )
    options = dict(config.backend_options)
    options.update(backend_options or {})
    if request.model_id is not None:
        options.setdefault("model_id", request.model_id)
    options.setdefault("device", request.device)
    backend = registry.choose(request.backend, **options)
    manifest = RunManifest(
        workflow=f"zynnova.zynvista.{request.mode.value}",
        backend=backend.name,
        configuration={"request": to_jsonable(request), "config": to_jsonable(config)},
        provenance={
            "input_images": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in request.images
            ],
            "input_video": None
            if request.video is None
            else {"path": str(request.video.resolve()), "sha256": sha256_file(request.video)},
        },
    )
    run_directory = base / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest.event("backend_selected", backend=backend.name)
    try:
        output = backend.run(request, config, run_directory / "backend")
        native_mesh = output.native_assets.get("mesh")
        if output.mesh is None and native_mesh is not None:
            try:
                output = _replace(output, mesh=load_triangle_mesh(native_mesh))
                manifest.event("native_mesh_parsed", path=str(native_mesh))
            except Exception as exc:
                # Preserve the original PBR/native asset even when the lightweight
                # geometry importer cannot decode it in the current environment.
                manifest.event(
                    "native_mesh_parse_skipped",
                    path=str(native_mesh),
                    type=type(exc).__name__,
                    message=str(exc),
                )
        if output.dense_views and output.point_cloud is None:
            output = _replace(
                output,
                point_cloud=fuse_dense_views(
                    output.dense_views,
                    confidence_percentile=config.confidence_percentile,
                    voxel_size=config.fusion_voxel_size_m,
                    maximum_points=config.maximum_points,
                ),
            )
        if config.build_mesh and output.dense_views and output.mesh is None:
            output = _replace(
                output,
                mesh=dense_views_to_mesh(
                    output.dense_views,
                    edge_factor=config.mesh_edge_factor,
                    confidence_percentile=config.confidence_percentile,
                ),
            )
        if config.style_backend:
            style_kwargs = dict(config.style_options)
            style_kwargs.update(style_options or {})
            style = STYLE_BACKENDS.create(config.style_backend, **style_kwargs)
            output = style.apply(output, config, run_directory / "style")
            manifest.event("style_applied", backend=style.name)
        exported = export_scene_output(
            output,
            run_directory / "exports",
            formats=config.export_formats,
            export_colmap=config.export_colmap,
        )
        for path in exported:
            manifest.add_artifact(path, role=_role(path))
        manifest.event(
            "scene_completed",
            dense_views=len(output.dense_views),
            points=0 if output.point_cloud is None else int(output.point_cloud.points.shape[0]),
            triangles=0 if output.mesh is None else int(output.mesh.faces.shape[0]),
            exported=len(exported),
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(run_directory / "manifest.json")
        raise
    manifest_path = manifest.save(run_directory / "manifest.json")
    return SceneResult(output, run_directory, tuple(exported), manifest_path)


def _replace(output: SceneBackendOutput, **updates: object) -> SceneBackendOutput:
    values = {
        "backend": output.backend,
        "dense_views": output.dense_views,
        "point_cloud": output.point_cloud,
        "mesh": output.mesh,
        "scene": output.scene,
        "native_assets": output.native_assets,
        "metadata": output.metadata,
    }
    values.update(updates)
    return SceneBackendOutput(**values)  # type: ignore[arg-type]


def _role(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {
        ".obj",
        ".stl",
        ".glb",
        ".gltf",
        ".fbx",
        ".usd",
        ".usda",
        ".usdc",
        ".dae",
        ".abc",
    }:
        return "scene_mesh"
    if suffix == ".spz":
        return "gaussian_splat"
    if path.name in {"cameras.txt", "images.txt", "points3D.txt"}:
        return "colmap"
    return "scene_asset"


__all__ = ["SceneResult", "run_scene"]
