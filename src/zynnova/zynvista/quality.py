"""Geometry/metric audits used to keep ZynVista outputs physically coherent.

The learned reconstruction/generation backend is intentionally separated from these
checks.  A backend may provide very high visual quality while still returning a scene
with an accidental scale jump, an invalid camera transform, or geometry-changing style
post-processing.  The routines here provide dependency-light evidence for those failure
modes and a geometry-only fingerprint that ignores appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np

from ..geometry import PointCloud, SceneBundle, TriangleMesh
from .types import SceneBackendOutput


@dataclass(frozen=True, slots=True)
class SceneGeometryAudit:
    dense_views: int
    cameras: int
    points: int
    triangles: int
    native_assets: int
    bounds_min_xyz: tuple[float, float, float] | None
    bounds_max_xyz: tuple[float, float, float] | None
    extent_xyz: tuple[float, float, float] | None
    diagonal_m: float | None
    minimum_camera_baseline_m: float | None
    median_camera_baseline_m: float | None
    maximum_camera_baseline_m: float | None
    finite_geometry: bool
    metric_plausible: bool
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GeometryFingerprint:
    sha256: str
    point_count: int
    vertex_count: int
    face_count: int
    scene_assets: int


def audit_scene_geometry(
    output: SceneBackendOutput,
    *,
    require_metric: bool = True,
    minimum_extent_m: float = 1.0e-6,
    maximum_extent_m: float = 1.0e7,
) -> SceneGeometryAudit:
    """Return scale/topology evidence without depending on a renderer.

    The metric plausibility gate is deliberately broad: it catches numerical scale
    collapse/explosion, not semantic scene-size mistakes.  A laboratory object and a
    city can both be valid metric scenes.
    """

    point_sets = list(_geometry_point_sets(output))
    finite = all(np.all(np.isfinite(points)) for points in point_sets)
    if point_sets and sum(len(value) for value in point_sets):
        minimum = np.min(np.concatenate(point_sets, axis=0), axis=0)
        maximum = np.max(np.concatenate(point_sets, axis=0), axis=0)
        extent = maximum - minimum
        diagonal = float(np.linalg.norm(extent))
        bounds_min = tuple(float(v) for v in minimum)
        bounds_max = tuple(float(v) for v in maximum)
        extent_xyz = tuple(float(v) for v in extent)
    else:
        bounds_min = bounds_max = extent_xyz = None
        diagonal = None

    camera_centres = [
        view.camera.pose[:3, 3]
        for view in output.dense_views
        if view.camera is not None
    ]
    baselines = _pairwise_neighbour_baselines(camera_centres)
    if len(baselines):
        baseline_min = float(np.min(baselines))
        baseline_median = float(np.median(baselines))
        baseline_max = float(np.max(baselines))
    else:
        baseline_min = baseline_median = baseline_max = None

    has_metric_geometry = diagonal is not None and minimum_extent_m <= diagonal <= maximum_extent_m
    metric_plausible = bool(finite and (has_metric_geometry or not require_metric))
    return SceneGeometryAudit(
        dense_views=len(output.dense_views),
        cameras=len(camera_centres),
        points=0 if output.point_cloud is None else output.point_cloud.n_points,
        triangles=0 if output.mesh is None else output.mesh.n_faces,
        native_assets=len(output.native_assets),
        bounds_min_xyz=bounds_min,
        bounds_max_xyz=bounds_max,
        extent_xyz=extent_xyz,
        diagonal_m=diagonal,
        minimum_camera_baseline_m=baseline_min,
        median_camera_baseline_m=baseline_median,
        maximum_camera_baseline_m=baseline_max,
        finite_geometry=finite,
        metric_plausible=metric_plausible,
        metadata={
            "minimum_extent_m": float(minimum_extent_m),
            "maximum_extent_m": float(maximum_extent_m),
            "require_metric": bool(require_metric),
        },
    )


def geometry_fingerprint(output: SceneBackendOutput) -> GeometryFingerprint:
    """Hash only scene geometry/transforms, intentionally excluding style/appearance."""

    digest = hashlib.sha256()
    point_count = vertex_count = face_count = scene_assets = 0

    def add_array(tag: str, value: np.ndarray) -> None:
        array = np.ascontiguousarray(value)
        digest.update(tag.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))

    if output.point_cloud is not None:
        add_array("point_cloud.points", output.point_cloud.points)
        point_count += output.point_cloud.n_points
    if output.mesh is not None:
        add_array("mesh.vertices", output.mesh.vertices)
        add_array("mesh.faces", output.mesh.faces)
        vertex_count += output.mesh.n_vertices
        face_count += output.mesh.n_faces
    if output.scene is not None:
        for index, asset in enumerate(output.scene.assets):
            scene_assets += 1
            add_array(f"scene.{index}.transform", asset.transform)
            if asset.point_cloud is not None:
                add_array(f"scene.{index}.points", asset.point_cloud.points)
                point_count += asset.point_cloud.n_points
            if asset.mesh is not None:
                add_array(f"scene.{index}.vertices", asset.mesh.vertices)
                add_array(f"scene.{index}.faces", asset.mesh.faces)
                vertex_count += asset.mesh.n_vertices
                face_count += asset.mesh.n_faces
            if asset.gaussian_splat is not None:
                # Means/scales/rotations are geometry.  Opacity/features are appearance.
                add_array(f"scene.{index}.gs.means", asset.gaussian_splat.means)
                add_array(f"scene.{index}.gs.scales", asset.gaussian_splat.scales)
                add_array(f"scene.{index}.gs.rotations", asset.gaussian_splat.rotations)
                point_count += len(asset.gaussian_splat.means)
    return GeometryFingerprint(
        sha256=digest.hexdigest(),
        point_count=point_count,
        vertex_count=vertex_count,
        face_count=face_count,
        scene_assets=scene_assets,
    )


def assert_geometry_preserved(
    before: GeometryFingerprint,
    after: GeometryFingerprint,
    *,
    operation: str = "style transfer",
) -> None:
    """Reject an appearance-only operation that silently moves/remeshes geometry."""

    if before.sha256 != after.sha256:
        raise ValueError(
            f"{operation} changed canonical in-memory geometry while geometry lock is enabled; "
            f"before={before.sha256[:12]}, after={after.sha256[:12]}"
        )


def _geometry_point_sets(output: SceneBackendOutput) -> Iterable[np.ndarray]:
    if output.point_cloud is not None:
        yield output.point_cloud.points
    if output.mesh is not None:
        yield output.mesh.vertices
    for view in output.dense_views:
        valid = np.all(np.isfinite(view.points_world), axis=-1)
        if view.mask is not None:
            valid &= view.mask
        if np.any(valid):
            yield view.points_world[valid]
    if output.scene is not None:
        yield from _scene_point_sets(output.scene)


def _scene_point_sets(scene: SceneBundle) -> Iterable[np.ndarray]:
    for asset in scene.assets:
        rotation = asset.transform[:3, :3]
        translation = asset.transform[:3, 3]
        for geometry in (asset.point_cloud, asset.mesh, asset.gaussian_splat):
            if geometry is None:
                continue
            if isinstance(geometry, PointCloud):
                values = geometry.points
            elif isinstance(geometry, TriangleMesh):
                values = geometry.vertices
            else:
                values = geometry.means
            yield values @ rotation.T + translation


def _pairwise_neighbour_baselines(centres: list[np.ndarray]) -> np.ndarray:
    if len(centres) < 2:
        return np.empty(0, dtype=np.float64)
    values = np.asarray(centres, dtype=np.float64)
    # Consecutive-view baselines are more meaningful for video/image sequences than
    # every O(N^2) pair and remain linear for large captures.
    distances = np.linalg.norm(values[1:] - values[:-1], axis=1)
    return distances[np.isfinite(distances) & (distances > 0.0)]


__all__ = [
    "GeometryFingerprint",
    "SceneGeometryAudit",
    "assert_geometry_preserved",
    "audit_scene_geometry",
    "geometry_fingerprint",
]
