from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zynnova.core import Availability, BackendDescriptor
from zynnova.geometry import Camera
from zynnova.zynvista import (
    DenseView,
    RECONSTRUCTION_BACKENDS,
    SceneBackendOutput,
    SceneConfig,
    SceneRequest,
    run_scene,
)


class _DenseGridBackend:
    name = "test-dense-grid"

    def availability(self) -> Availability:
        return Availability(True)

    def run(self, request: SceneRequest, config: SceneConfig, work_directory: Path) -> SceneBackendOutput:
        del request, config
        work_directory.mkdir(parents=True, exist_ok=True)
        height, width = 6, 7
        y, x = np.mgrid[0:height, 0:width]
        points = np.stack((0.02 * x, 0.02 * y, 1.0 + 0.002 * x * y), axis=-1)
        colors = np.stack(
            (
                x / max(width - 1, 1),
                y / max(height - 1, 1),
                np.full_like(x, 0.35, dtype=float),
            ),
            axis=-1,
        )
        camera = Camera(
            pose=np.eye(4),
            intrinsics=np.asarray([[80.0, 0.0, 3.0], [0.0, 80.0, 2.5], [0.0, 0.0, 1.0]]),
            width=width,
            height=height,
            name="frame_000.png",
        )
        return SceneBackendOutput(
            backend=self.name,
            dense_views=(
                DenseView(
                    points_world=points,
                    image_rgb=colors,
                    confidence=np.ones((height, width), dtype=float),
                    camera=camera,
                    name="frame_000",
                ),
            ),
            metadata={"fixture": True},
        )


def test_scene_dense_reconstruction_style_export_and_colmap(tmp_path, image_factory) -> None:
    if "test-dense-grid" not in RECONSTRUCTION_BACKENDS:
        RECONSTRUCTION_BACKENDS.register(
            BackendDescriptor(
                name="test-dense-grid",
                task="scene-reconstruction",
                factory=_DenseGridBackend,
                summary="deterministic dense scene test backend",
                default_rank=999,
            )
        )
    image = image_factory(tmp_path / "input.png")
    style = image_factory(tmp_path / "style.png", style=True)
    result = run_scene(
        SceneRequest(images=(image,), backend="test-dense-grid"),
        SceneConfig(
            output_directory=str(tmp_path / "runs"),
            confidence_percentile=0.0,
            fusion_voxel_size_m=0.001,
            maximum_points=1_000,
            build_mesh=True,
            mesh_edge_factor=4.0,
            export_formats=("ply", "obj", "stl", "npz"),
            export_colmap=True,
            style_backend="statistical-color",
            style_reference=style,
            style_options={"strength": 0.65},
        ),
    )

    assert result.output.point_cloud is not None
    assert result.output.mesh is not None
    assert result.output.point_cloud.n_points == 42
    assert result.output.mesh.n_faces == 2 * 5 * 6
    assert result.output.mesh.vertex_colors is not None
    names = {path.name for path in result.exported_files}
    assert {"scene.ply", "scene.obj", "scene.stl", "scene.npz"} <= names
    assert {"cameras.txt", "images.txt", "points3D.txt"} <= names
    assert all(path.is_file() for path in result.exported_files)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert any(event["name"] == "style_applied" for event in manifest["events"])


def test_scene_emits_metric_audit_and_large_world_index(tmp_path, image_factory) -> None:
    if "test-dense-grid" not in RECONSTRUCTION_BACKENDS:
        RECONSTRUCTION_BACKENDS.register(
            BackendDescriptor(
                name="test-dense-grid",
                task="scene-reconstruction",
                factory=_DenseGridBackend,
                summary="deterministic dense scene test backend",
                default_rank=999,
            )
        )
    image = image_factory(tmp_path / "world_input.png")
    result = run_scene(
        SceneRequest(images=(image,), backend="test-dense-grid"),
        SceneConfig(
            output_directory=str(tmp_path / "world-runs"),
            confidence_percentile=0.0,
            fusion_voxel_size_m=0.001,
            build_mesh=True,
            export_formats=("ply",),
            export_colmap=False,
            build_world_hierarchy=True,
            world_chunk_size_m=0.05,
            world_lod_levels=2,
        ),
    )
    assert result.quality_path is not None and result.quality_path.is_file()
    quality = json.loads(result.quality_path.read_text(encoding="utf-8"))
    assert quality["metric_plausible"] is True
    assert quality["diagonal_m"] > 0.0
    assert result.world_index_path is not None and result.world_index_path.is_file()
    index = json.loads(result.world_index_path.read_text(encoding="utf-8"))
    assert index["schema"] == "zynnova.world-index/1.0"
    assert index["unit"] == "meter"
    assert index["levels"] == 2
    assert index["chunks"]
