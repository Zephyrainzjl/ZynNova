"""Validated first-order tetrahedral meshes and structured mesh generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

import numpy as np

from ..exceptions import MeshError


def _as_float_nodes(value: object) -> np.ndarray:
    nodes = np.ascontiguousarray(value, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise MeshError("nodes must have shape (n_nodes, 3)")
    if not np.all(np.isfinite(nodes)):
        raise MeshError("nodes contain non-finite coordinates")
    return nodes


def _as_cells(value: object) -> np.ndarray:
    cells = np.ascontiguousarray(value, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise MeshError("cells must contain first-order tetrahedra with shape (n_cells, 4)")
    return cells


def _as_faces(value: object, *, name: str) -> np.ndarray:
    faces = np.ascontiguousarray(value, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MeshError(f"boundary set {name!r} must have shape (n_faces, 3)")
    return faces


@dataclass(slots=True)
class Mesh:
    """A 3-D first-order tetrahedral mesh.

    ``cell_regions`` uses integer labels so region-specific coefficients can be
    assembled without Python objects in the C++ hot path. ``boundary_faces``
    maps human-readable names to outward-oriented triangles.
    """

    nodes: np.ndarray
    cells: np.ndarray
    cell_regions: np.ndarray | None = None
    boundary_faces: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = _as_float_nodes(self.nodes)
        self.cells = _as_cells(self.cells)
        if len(self.nodes) < 4:
            raise MeshError("a tetrahedral mesh needs at least four nodes")
        if len(self.cells) < 1:
            raise MeshError("a tetrahedral mesh needs at least one cell")
        if np.min(self.cells) < 0 or np.max(self.cells) >= len(self.nodes):
            raise MeshError("cell connectivity contains an out-of-range node index")
        if np.any(
            np.asarray([len(set(map(int, cell))) for cell in self.cells], dtype=int) != 4
        ):
            raise MeshError("a tetrahedron cannot repeat a node")

        if self.cell_regions is None:
            self.cell_regions = np.zeros(len(self.cells), dtype=np.int32)
        else:
            regions = np.ascontiguousarray(self.cell_regions, dtype=np.int32)
            if regions.shape != (len(self.cells),):
                raise MeshError("cell_regions must have shape (n_cells,)")
            self.cell_regions = regions

        normalized_faces: dict[str, np.ndarray] = {}
        for name, raw_faces in self.boundary_faces.items():
            faces = _as_faces(raw_faces, name=str(name))
            if faces.size and (np.min(faces) < 0 or np.max(faces) >= len(self.nodes)):
                raise MeshError(f"boundary set {name!r} contains an invalid node index")
            normalized_faces[str(name)] = faces
        self.boundary_faces = normalized_faces
        self.metadata = dict(self.metadata)
        self._validate_volumes()

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def _validate_volumes(self, tolerance: float = 64.0 * np.finfo(float).eps) -> None:
        x = self.nodes[self.cells]
        jacobians = np.stack(
            (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
            axis=1,
        )
        determinants = np.linalg.det(jacobians)
        scales = np.max(np.abs(jacobians), axis=(1, 2))
        thresholds = tolerance * np.maximum(
            scales**3, np.finfo(float).tiny
        )
        invalid = ~np.isfinite(determinants) | (np.abs(determinants) <= thresholds)
        if np.any(invalid):
            indices = np.flatnonzero(invalid)[:8].tolist()
            raise MeshError(f"degenerate tetrahedra at cell indices {indices}")

    def cell_volumes(self) -> np.ndarray:
        x = self.nodes[self.cells]
        jacobians = np.stack(
            (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
            axis=1,
        )
        return np.abs(np.linalg.det(jacobians)) / 6.0

    def cell_centers(self) -> np.ndarray:
        return self.nodes[self.cells].mean(axis=1)

    def boundary_nodes(self, name: str) -> np.ndarray:
        try:
            faces = self.boundary_faces[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.boundary_faces)) or "<none>"
            raise MeshError(f"unknown boundary {name!r}; available: {available}") from exc
        return np.unique(faces.reshape(-1))

    def region_cells(self, region: int) -> np.ndarray:
        return np.flatnonzero(self.cell_regions == int(region))

    def region_nodes(self, region: int) -> np.ndarray:
        selected = self.region_cells(region)
        if selected.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.cells[selected].reshape(-1))

    def exterior_faces(self) -> np.ndarray:
        """Return all topological exterior faces.

        Orientation is inherited from the owning tetrahedron and corrected to
        point away from its opposite node.
        """

        owners: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        counts: dict[tuple[int, int, int], int] = {}
        local_faces = ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3))
        for cell in self.cells:
            for a, b, c, opposite in local_faces:
                face = np.asarray([cell[a], cell[b], cell[c]], dtype=np.int64)
                key = tuple(sorted(map(int, face)))
                counts[key] = counts.get(key, 0) + 1
                owners[key] = (face, int(cell[opposite]))

        exterior: list[np.ndarray] = []
        for key, count in counts.items():
            if count != 1:
                continue
            face, opposite = owners[key]
            p0, p1, p2 = self.nodes[face]
            normal = np.cross(p1 - p0, p2 - p0)
            if np.dot(normal, self.nodes[opposite] - p0) > 0.0:
                face = face[[0, 2, 1]]
            exterior.append(face)
        if not exterior:
            return np.empty((0, 3), dtype=np.int64)
        return np.ascontiguousarray(exterior, dtype=np.int64)

    def with_coordinate_boundaries(
        self,
        *,
        tolerance: float | None = None,
        overwrite: bool = False,
    ) -> Mesh:
        """Return a copy with ``xmin/xmax/.../zmax`` exterior face sets."""

        scale = max(float(np.ptp(self.nodes, axis=0).max()), 1.0)
        tol = 1.0e-10 * scale if tolerance is None else float(tolerance)
        faces = self.exterior_faces()
        coordinates = self.nodes[faces]
        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        generated: dict[str, np.ndarray] = {}
        axes = ("x", "y", "z")
        for axis, label in enumerate(axes):
            generated[f"{label}min"] = faces[
                np.all(np.abs(coordinates[:, :, axis] - mins[axis]) <= tol, axis=1)
            ]
            generated[f"{label}max"] = faces[
                np.all(np.abs(coordinates[:, :, axis] - maxs[axis]) <= tol, axis=1)
            ]
        merged = dict(self.boundary_faces)
        for name, selected in generated.items():
            if overwrite or name not in merged:
                merged[name] = selected
        return Mesh(
            nodes=self.nodes.copy(),
            cells=self.cells.copy(),
            cell_regions=self.cell_regions.copy(),
            boundary_faces=merged,
            metadata=self.metadata.copy(),
        )

    def validate_manifold(self) -> None:
        """Reject faces shared by more than two tetrahedra."""

        counts: dict[tuple[int, int, int], int] = {}
        for cell in self.cells:
            for face in combinations(map(int, cell), 3):
                key = tuple(sorted(face))
                counts[key] = counts.get(key, 0) + 1
        non_manifold = [face for face, count in counts.items() if count > 2]
        if non_manifold:
            raise MeshError(f"non-manifold tetrahedral faces detected: {non_manifold[:8]}")


def box_tetrahedral_mesh(
    lengths: tuple[float, float, float],
    shape: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """Generate a conforming six-tetrahedra-per-voxel box mesh.

    The Freudenthal split uses the same body diagonal in every voxel, avoiding
    hanging or mismatched faces between neighbors.
    """

    if len(lengths) != 3 or any(float(value) <= 0.0 for value in lengths):
        raise MeshError("lengths must contain three positive values")
    if len(shape) != 3 or any(int(value) < 1 for value in shape):
        raise MeshError("shape must contain three positive cell counts")
    nx, ny, nz = map(int, shape)
    lx, ly, lz = map(float, lengths)
    ox, oy, oz = map(float, origin)
    xs = np.linspace(ox, ox + lx, nx + 1)
    ys = np.linspace(oy, oy + ly, ny + 1)
    zs = np.linspace(oz, oz + lz, nz + 1)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    nodes = grid.reshape(-1, 3)

    def index(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    cells: list[tuple[int, int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v000 = index(i, j, k)
                v100 = index(i + 1, j, k)
                v010 = index(i, j + 1, k)
                v110 = index(i + 1, j + 1, k)
                v001 = index(i, j, k + 1)
                v101 = index(i + 1, j, k + 1)
                v011 = index(i, j + 1, k + 1)
                v111 = index(i + 1, j + 1, k + 1)
                cells.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )
    mesh = Mesh(nodes=nodes, cells=np.asarray(cells, dtype=np.int64))
    return mesh.with_coordinate_boundaries()


__all__ = ["Mesh", "box_tetrahedral_mesh"]
