"""Spatially chunked large-world/LOD export for persistent scene assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core import dump_json
from ..geometry import PointCloud, TriangleMesh, export_point_cloud, export_triangle_mesh
from .types import SceneBackendOutput


@dataclass(frozen=True, slots=True)
class WorldChunkRecord:
    level: int
    key_xyz: tuple[int, int, int]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    point_count: int
    triangle_count: int
    point_asset: str | None
    mesh_asset: str | None


@dataclass(frozen=True, slots=True)
class WorldIndex:
    schema: str
    unit: str
    up_axis: str
    base_chunk_size_m: float
    overlap_m: float
    levels: int
    origin_xyz: tuple[float, float, float] | None
    bounds_min_xyz: tuple[float, float, float] | None
    bounds_max_xyz: tuple[float, float, float] | None
    chunks: tuple[WorldChunkRecord, ...]
    native_assets: Mapping[str, str]
    metadata: Mapping[str, object]


def export_world_hierarchy(
    output: SceneBackendOutput,
    directory: str | Path,
    *,
    chunk_size_m: float,
    levels: int = 3,
    overlap_m: float = 0.25,
    up_axis: str = "Y",
) -> Path:
    """Write deterministic spatial chunks and a small world index.

    LOD levels increase spatial tile size rather than destructively decimating the
    canonical scene.  Renderers/DCC tools can stream fewer, larger tiles at distance
    while the original 3DGS/native mesh remains preserved separately.
    """

    if chunk_size_m <= 0.0:
        raise ValueError("chunk_size_m must be positive")
    if levels < 1:
        raise ValueError("levels must be positive")
    if overlap_m < 0.0:
        raise ValueError("overlap_m cannot be negative")
    axis = str(up_axis).strip().upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("up_axis must be X, Y, or Z")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    point_sets = []
    if output.point_cloud is not None and output.point_cloud.n_points:
        point_sets.append(output.point_cloud.points)
    if output.mesh is not None and output.mesh.n_vertices:
        point_sets.append(output.mesh.vertices)
    if point_sets:
        all_points = np.concatenate(point_sets, axis=0)
        bounds_min = np.min(all_points, axis=0)
        bounds_max = np.max(all_points, axis=0)
        origin = bounds_min.copy()
    else:
        bounds_min = bounds_max = origin = None

    records: list[WorldChunkRecord] = []
    for level in range(levels):
        size = float(chunk_size_m * (2**level))
        point_groups = _group_points(output.point_cloud, origin, size)
        face_groups = _group_faces(output.mesh, origin, size)
        keys = sorted(set(point_groups) | set(face_groups))
        for key in keys:
            chunk_dir = target / f"lod_{level}" / f"{key[0]}_{key[1]}_{key[2]}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            point_path = None
            mesh_path = None
            point_count = triangle_count = 0
            if key in point_groups and output.point_cloud is not None:
                indices = point_groups[key]
                cloud = _point_subset(output.point_cloud, indices)
                point_count = cloud.n_points
                point_path = export_point_cloud(chunk_dir / "points.ply", cloud)
            if key in face_groups and output.mesh is not None:
                mesh = _face_subset(output.mesh, face_groups[key])
                triangle_count = mesh.n_faces
                mesh_path = export_triangle_mesh(chunk_dir / "mesh.ply", mesh)
            lower = np.asarray(origin, dtype=float) + size * np.asarray(key, dtype=float)
            upper = lower + size
            records.append(
                WorldChunkRecord(
                    level=level,
                    key_xyz=key,
                    bounds_min_xyz=tuple(float(v) for v in lower - overlap_m),
                    bounds_max_xyz=tuple(float(v) for v in upper + overlap_m),
                    point_count=point_count,
                    triangle_count=triangle_count,
                    point_asset=None if point_path is None else str(point_path.relative_to(target)),
                    mesh_asset=None if mesh_path is None else str(mesh_path.relative_to(target)),
                )
            )

    native = {role: str(path) for role, path in output.native_assets.items()}
    index = WorldIndex(
        schema="zynnova.world-index/1.0",
        unit="meter",
        up_axis=axis,
        base_chunk_size_m=float(chunk_size_m),
        overlap_m=float(overlap_m),
        levels=int(levels),
        origin_xyz=None if origin is None else tuple(float(v) for v in origin),
        bounds_min_xyz=None if bounds_min is None else tuple(float(v) for v in bounds_min),
        bounds_max_xyz=None if bounds_max is None else tuple(float(v) for v in bounds_max),
        chunks=tuple(records),
        native_assets=native,
        metadata={
            "lod_policy": "non-destructive-spatial-hierarchy",
            "canonical_native_assets_preserved": bool(native),
        },
    )
    return dump_json(target / "world_index.json", asdict(index))


def _key(points: np.ndarray, origin: np.ndarray, size: float) -> np.ndarray:
    return np.floor((points - origin[None, :]) / size).astype(np.int64)


def _group_points(
    cloud: PointCloud | None,
    origin: np.ndarray | None,
    size: float,
) -> dict[tuple[int, int, int], np.ndarray]:
    if cloud is None or origin is None or not cloud.n_points:
        return {}
    keys = _key(cloud.points, origin, size)
    result: dict[tuple[int, int, int], list[int]] = {}
    for index, raw in enumerate(keys):
        key = tuple(int(v) for v in raw)
        result.setdefault(key, []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in result.items()}


def _group_faces(
    mesh: TriangleMesh | None,
    origin: np.ndarray | None,
    size: float,
) -> dict[tuple[int, int, int], np.ndarray]:
    if mesh is None or origin is None or not mesh.n_faces:
        return {}
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    keys = _key(centroids, origin, size)
    result: dict[tuple[int, int, int], list[int]] = {}
    for index, raw in enumerate(keys):
        key = tuple(int(v) for v in raw)
        result.setdefault(key, []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in result.items()}


def _point_subset(cloud: PointCloud, indices: np.ndarray) -> PointCloud:
    return PointCloud(
        points=cloud.points[indices],
        colors=None if cloud.colors is None else cloud.colors[indices],
        normals=None if cloud.normals is None else cloud.normals[indices],
        confidence=None if cloud.confidence is None else cloud.confidence[indices],
        metadata={**cloud.metadata, "world_chunk": True},
    )


def _face_subset(mesh: TriangleMesh, face_indices: np.ndarray) -> TriangleMesh:
    faces = mesh.faces[face_indices]
    used = np.unique(faces)
    remap = np.full(mesh.n_vertices, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return TriangleMesh(
        vertices=mesh.vertices[used],
        faces=remap[faces],
        vertex_colors=None if mesh.vertex_colors is None else mesh.vertex_colors[used],
        vertex_normals=None if mesh.vertex_normals is None else mesh.vertex_normals[used],
        uv=None if mesh.uv is None else mesh.uv[used],
        face_materials=None if mesh.face_materials is None else mesh.face_materials[face_indices],
        texture_paths=mesh.texture_paths,
        metadata={**mesh.metadata, "world_chunk": True},
    )


__all__ = ["WorldChunkRecord", "WorldIndex", "export_world_hierarchy"]
