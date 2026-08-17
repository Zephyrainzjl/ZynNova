"""Cross-DCC scene export, COLMAP text export, and native-asset preservation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from ..geometry import Camera, export_point_cloud, export_triangle_mesh
from .types import SceneBackendOutput


def export_scene_output(
    output: SceneBackendOutput,
    directory: str | Path,
    *,
    formats: Iterable[str],
    export_colmap: bool = True,
) -> list[Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    normalized_formats = tuple(
        dict.fromkeys(str(item).lower().lstrip(".") for item in formats)
    )
    for fmt in normalized_formats:
        if output.mesh is not None:
            paths.append(export_triangle_mesh(target / f"scene.{fmt}", output.mesh))
        elif output.point_cloud is not None and fmt in {"ply", "npz"}:
            paths.append(export_point_cloud(target / f"scene_points.{fmt}", output.point_cloud))
    if output.point_cloud is not None and not any(
        path.name.startswith("scene_points") for path in paths
    ):
        paths.append(export_point_cloud(target / "scene_points.ply", output.point_cloud))
    native_directory = target / "native"
    for role, source in output.native_assets.items():
        native_directory.mkdir(parents=True, exist_ok=True)
        destination = native_directory / f"{role}{source.suffix}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        paths.append(destination)
    cameras = tuple(view.camera for view in output.dense_views if view.camera is not None)
    if export_colmap and cameras:
        paths.extend(write_colmap_text(target / "colmap", cameras))
    return paths


def write_colmap_text(directory: str | Path, cameras: Iterable[Camera]) -> list[Path]:
    """Write pinhole camera poses in COLMAP's text interchange format."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# Camera list with one line of data per camera:\n"]
    image_lines = ["# Image list with two lines of data per image:\n"]
    for index, camera in enumerate(cameras, start=1):
        fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
        cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
        camera_lines.append(
            f"{index} PINHOLE {camera.width} {camera.height} "
            f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n"
        )
        world_to_camera = np.linalg.inv(camera.pose)
        quaternion = _rotation_to_quaternion(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        name = Path(camera.name).name
        image_lines.append(
            f"{index} {quaternion[0]:.17g} {quaternion[1]:.17g} "
            f"{quaternion[2]:.17g} {quaternion[3]:.17g} "
            f"{translation[0]:.17g} {translation[1]:.17g} "
            f"{translation[2]:.17g} {index} {name}\n\n"
        )
    camera_path = target / "cameras.txt"
    image_path = target / "images.txt"
    points_path = target / "points3D.txt"
    camera_path.write_text("".join(camera_lines), encoding="utf-8")
    image_path.write_text("".join(image_lines), encoding="utf-8")
    points_path.write_text(
        "# Empty sparse point list; dense geometry is exported separately.\n",
        encoding="utf-8",
    )
    return [camera_path, image_path, points_path]


def _rotation_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        values = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(
                max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1.0e-15)
            )
            values = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(
                max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1.0e-15)
            )
            values = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(
                max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1.0e-15)
            )
            values = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])


__all__ = ["export_scene_output", "write_colmap_text"]
