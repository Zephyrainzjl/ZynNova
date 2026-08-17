"""Validated geometry containers shared by reconstruction, generation, and FEM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.exceptions import GeometryError


def _points3(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise GeometryError(f"{name} must have shape (N, 3), received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GeometryError(f"{name} contains non-finite coordinates")
    return array


def _indices(values: np.ndarray, *, width: int, upper: int, name: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != width:
        raise GeometryError(f"{name} must have shape (N, {width}), received {array.shape}")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= upper):
        raise GeometryError(f"{name} contains an out-of-range vertex index")
    return array


def _optional_vectors(
    values: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    name: str,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.shape != shape:
        raise GeometryError(f"{name} must have shape {shape}, received {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise GeometryError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    confidence: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _points3(self.points, name="points")
        count = len(points)
        colors = _optional_vectors(self.colors, shape=(count, 3), name="colors")
        if colors is not None and colors.size and float(colors.max()) > 1.0:
            colors = colors / 255.0
        normals = _optional_vectors(self.normals, shape=(count, 3), name="normals")
        confidence = _optional_vectors(
            self.confidence,
            shape=(count,),
            name="confidence",
        )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None
    vertex_normals: np.ndarray | None = None
    uv: np.ndarray | None = None
    face_materials: np.ndarray | None = None
    texture_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vertices = _points3(self.vertices, name="vertices")
        faces = _indices(self.faces, width=3, upper=len(vertices), name="faces")
        colors = _optional_vectors(
            self.vertex_colors,
            shape=(len(vertices), 3),
            name="vertex_colors",
        )
        if colors is not None and colors.size and float(colors.max()) > 1.0:
            colors = colors / 255.0
        normals = _optional_vectors(
            self.vertex_normals,
            shape=(len(vertices), 3),
            name="vertex_normals",
        )
        uv = _optional_vectors(self.uv, shape=(len(vertices), 2), name="uv")
        face_materials = _optional_vectors(
            self.face_materials,
            shape=(len(faces),),
            name="face_materials",
            dtype=np.dtype(np.int32),
        )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "vertex_colors", colors)
        object.__setattr__(self, "vertex_normals", normals)
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "face_materials", face_materials)
        object.__setattr__(
            self,
            "texture_paths",
            tuple(Path(path) for path in self.texture_paths),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)


@dataclass(frozen=True, slots=True)
class VolumeMesh:
    nodes: np.ndarray
    tetrahedra: np.ndarray
    cell_regions: np.ndarray | None = None
    region_names: Mapping[int, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = _points3(self.nodes, name="nodes")
        tetrahedra = _indices(
            self.tetrahedra,
            width=4,
            upper=len(nodes),
            name="tetrahedra",
        )
        if self.cell_regions is None:
            regions = np.zeros(len(tetrahedra), dtype=np.int32)
        else:
            regions = np.ascontiguousarray(self.cell_regions, dtype=np.int32)
            if regions.shape != (len(tetrahedra),):
                raise GeometryError(
                    "cell_regions must have one integer per tetrahedron, "
                    f"received {regions.shape}"
                )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "tetrahedra", tetrahedra)
        object.__setattr__(self, "cell_regions", regions)
        object.__setattr__(
            self,
            "region_names",
            {int(key): str(value) for key, value in self.region_names.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        return len(self.tetrahedra)


@dataclass(frozen=True, slots=True)
class Camera:
    """World-from-camera pose and pinhole intrinsics."""

    pose: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    name: str = "camera"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pose = np.ascontiguousarray(self.pose, dtype=np.float64)
        intrinsics = np.ascontiguousarray(self.intrinsics, dtype=np.float64)
        if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
            raise GeometryError("camera pose/intrinsics must have shapes (4,4)/(3,3)")
        if self.width < 1 or self.height < 1:
            raise GeometryError("camera width and height must be positive")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(intrinsics)):
            raise GeometryError("camera matrices contain non-finite values")
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class GaussianSplat:
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacity: np.ndarray
    features: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        means = _points3(self.means, name="means")
        count = len(means)
        scales = _optional_vectors(self.scales, shape=(count, 3), name="scales")
        rotations = _optional_vectors(self.rotations, shape=(count, 4), name="rotations")
        opacity = _optional_vectors(self.opacity, shape=(count,), name="opacity")
        features = np.ascontiguousarray(self.features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != count:
            raise GeometryError("features must have shape (N, C)")
        assert scales is not None and rotations is not None and opacity is not None
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SceneAsset:
    """One scene node with optional geometry and an affine world transform."""

    name: str
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    mesh: TriangleMesh | None = None
    point_cloud: PointCloud | None = None
    gaussian_splat: GaussianSplat | None = None
    source_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transform = np.ascontiguousarray(self.transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise GeometryError("scene transform must be a finite (4,4) matrix")
        if self.mesh is None and self.point_cloud is None and self.gaussian_splat is None:
            if not self.source_paths:
                raise GeometryError("a scene asset needs geometry or at least one source path")
        object.__setattr__(self, "transform", transform)
        object.__setattr__(
            self,
            "source_paths",
            tuple(Path(path) for path in self.source_paths),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SceneBundle:
    assets: tuple[SceneAsset, ...]
    cameras: tuple[Camera, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.assets:
            raise GeometryError("scene bundle must contain at least one asset")
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(self, "cameras", tuple(self.cameras))
        object.__setattr__(self, "metadata", dict(self.metadata))


__all__ = [
    "Camera",
    "GaussianSplat",
    "PointCloud",
    "SceneAsset",
    "SceneBundle",
    "TriangleMesh",
    "VolumeMesh",
]
