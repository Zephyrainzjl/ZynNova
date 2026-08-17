"""Confidence-aware dense-view fusion and metric surface construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..geometry import PointCloud, TriangleMesh, clean_triangle_mesh
from .types import DenseView


def fuse_dense_views(
    views: Sequence[DenseView],
    *,
    confidence_percentile: float = 10.0,
    voxel_size: float = 0.005,
    maximum_points: int = 2_000_000,
) -> PointCloud:
    """Fuse world-space point maps by confidence filtering and voxel averaging."""

    if not views:
        raise ValueError("at least one dense view is required")
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    for view in views:
        valid = np.ones(view.points_world.shape[:2], dtype=bool)
        if view.mask is not None:
            valid &= view.mask
        confidence = (
            np.ones(view.points_world.shape[:2], dtype=np.float64)
            if view.confidence is None
            else view.confidence
        )
        finite = np.all(np.isfinite(view.points_world), axis=-1) & np.isfinite(confidence)
        valid &= finite
        selected_confidence = confidence[valid]
        if len(selected_confidence) and confidence_percentile > 0.0:
            threshold = float(np.percentile(selected_confidence, confidence_percentile))
            valid &= confidence >= threshold
        point_parts.append(view.points_world[valid])
        color_parts.append(view.image_rgb[valid])
        confidence_parts.append(confidence[valid])
    points = np.concatenate(point_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    confidence = np.concatenate(confidence_parts, axis=0)
    if not len(points):
        raise ValueError("confidence/mask filtering removed all scene points")
    if voxel_size > 0.0:
        points, colors, confidence = _voxel_average(points, colors, confidence, voxel_size)
    if len(points) > maximum_points:
        order = np.argsort(-confidence, kind="stable")[:maximum_points]
        points, colors, confidence = points[order], colors[order], confidence[order]
    return PointCloud(
        points=points,
        colors=colors,
        confidence=confidence,
        metadata={
            "source_views": len(views),
            "voxel_size": voxel_size,
            "confidence_percentile": confidence_percentile,
        },
    )


def dense_views_to_mesh(
    views: Sequence[DenseView],
    *,
    edge_factor: float = 3.5,
    confidence_percentile: float = 10.0,
    weld_tolerance: float = 1.0e-6,
) -> TriangleMesh:
    """Triangulate each metric point map and merge valid discontinuity-aware faces."""

    meshes = [
        dense_view_to_mesh(
            view,
            edge_factor=edge_factor,
            confidence_percentile=confidence_percentile,
        )
        for view in views
    ]
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        if not mesh.n_faces:
            continue
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        if mesh.vertex_colors is not None:
            colors.append(mesh.vertex_colors)
        offset += mesh.n_vertices
    if not vertices:
        raise ValueError("no valid triangles were produced from dense views")
    combined = TriangleMesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        vertex_colors=np.concatenate(colors, axis=0) if colors else None,
        metadata={"source_views": len(views)},
    )
    cleaned, _ = clean_triangle_mesh(combined, weld_tolerance=weld_tolerance)
    return cleaned


def dense_view_to_mesh(
    view: DenseView,
    *,
    edge_factor: float = 3.5,
    confidence_percentile: float = 10.0,
) -> TriangleMesh:
    points = view.points_world
    height, width = points.shape[:2]
    valid = np.ones((height, width), dtype=bool)
    if view.mask is not None:
        valid &= view.mask
    if view.confidence is not None:
        selected = view.confidence[valid]
        if len(selected) and confidence_percentile > 0.0:
            valid &= view.confidence >= np.percentile(selected, confidence_percentile)
    valid &= np.all(np.isfinite(points), axis=-1)

    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    upper_left = index[:-1, :-1]
    upper_right = index[:-1, 1:]
    lower_left = index[1:, :-1]
    lower_right = index[1:, 1:]
    candidates = np.stack(
        (
            np.stack((upper_left, lower_left, upper_right), axis=-1),
            np.stack((upper_right, lower_left, lower_right), axis=-1),
        ),
        axis=-2,
    ).reshape(-1, 3)
    valid_flat = valid.ravel()
    candidates = candidates[np.all(valid_flat[candidates], axis=1)]
    flattened = points.reshape(-1, 3)
    if len(candidates):
        triangles = flattened[candidates]
        edge_lengths = np.stack(
            (
                np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
            ),
            axis=1,
        )
        local_scale = _local_spacing(points, valid)
        threshold = max(local_scale * edge_factor, np.finfo(float).eps)
        candidates = candidates[np.max(edge_lengths, axis=1) <= threshold]
    return TriangleMesh(
        vertices=flattened,
        faces=candidates,
        vertex_colors=view.image_rgb.reshape(-1, 3),
        metadata={"source_view": view.name, "edge_factor": edge_factor},
    )


def _local_spacing(points: np.ndarray, valid: np.ndarray) -> float:
    distances: list[np.ndarray] = []
    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    if np.any(horizontal_valid):
        distances.append(
            np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)[horizontal_valid]
        )
    if np.any(vertical_valid):
        distances.append(
            np.linalg.norm(points[1:] - points[:-1], axis=-1)[vertical_valid]
        )
    if not distances:
        return 0.0
    values = np.concatenate(distances)
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(values)) if len(values) else 0.0


def _voxel_average(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    weights = np.maximum(confidence, np.finfo(float).eps)
    weight_sum = np.bincount(inverse, weights=weights, minlength=count)
    fused_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis] * weights, minlength=count)
            / weight_sum
            for axis in range(3)
        ]
    )
    fused_colors = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, axis] * weights, minlength=count)
            / weight_sum
            for axis in range(3)
        ]
    )
    fused_confidence = np.bincount(inverse, weights=confidence, minlength=count) / np.bincount(
        inverse, minlength=count
    )
    return fused_points, fused_colors, fused_confidence


__all__ = ["dense_view_to_mesh", "dense_views_to_mesh", "fuse_dense_views"]
