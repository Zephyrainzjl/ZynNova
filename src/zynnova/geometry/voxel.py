"""Exact structured voxel-to-Tet4 conversion and material interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.exceptions import GeometryError
from .quality import tetrahedron_signed_volumes
from .types import TriangleMesh, VolumeMesh


@dataclass(frozen=True, slots=True)
class VoxelMeshResult:
    volume_mesh: VolumeMesh
    boundary_mesh: TriangleMesh
    interface_faces: Mapping[tuple[int, int], np.ndarray]
    cell_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]


def validate_voxels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 3 or min(values.shape) < 1:
        raise GeometryError("voxel labels must be a non-empty three-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(values == np.rint(values)):
            raise GeometryError("voxel labels must contain integers")
    return np.ascontiguousarray(values, dtype=np.int32)


def normalize_spacing(spacing: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        result = (float(spacing),) * 3
    else:
        result = tuple(float(item) for item in spacing)
    if len(result) != 3 or min(result) <= 0.0 or not np.all(np.isfinite(result)):
        raise GeometryError("spacing must contain three finite positive numbers")
    return result


def voxel_to_tetrahedra(
    labels: np.ndarray,
    *,
    spacing: float | tuple[float, float, float] = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    region_names: Mapping[int, str] | None = None,
) -> VoxelMeshResult:
    """Split every voxel into six conforming Tet4 cells around one body diagonal."""

    phase = validate_voxels(labels)
    dz, dy, dx = normalize_spacing(spacing)
    oz, oy, ox = (float(item) for item in origin)
    nz, ny, nx = phase.shape

    z, y, x = np.meshgrid(
        oz + np.arange(nz + 1) * dz,
        oy + np.arange(ny + 1) * dy,
        ox + np.arange(nx + 1) * dx,
        indexing="ij",
    )
    nodes = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    def node_index(iz: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        return (iz * (ny + 1) + iy) * (nx + 1) + ix

    iz, iy, ix = np.meshgrid(
        np.arange(nz),
        np.arange(ny),
        np.arange(nx),
        indexing="ij",
    )
    corners = np.column_stack(
        [
            node_index(iz, iy, ix).ravel(),
            node_index(iz, iy, ix + 1).ravel(),
            node_index(iz, iy + 1, ix).ravel(),
            node_index(iz, iy + 1, ix + 1).ravel(),
            node_index(iz + 1, iy, ix).ravel(),
            node_index(iz + 1, iy, ix + 1).ravel(),
            node_index(iz + 1, iy + 1, ix).ravel(),
            node_index(iz + 1, iy + 1, ix + 1).ravel(),
        ]
    )
    local = np.asarray(
        [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ],
        dtype=np.int64,
    )
    tetrahedra = corners[:, local].reshape(-1, 4)
    regions = np.repeat(phase.ravel(), len(local)).astype(np.int32, copy=False)
    tentative = VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=regions,
        region_names=dict(region_names or {}),
        metadata={"source": "structured-voxel", "voxel_shape": phase.shape},
    )
    signed = tetrahedron_signed_volumes(tentative)
    negative = signed < 0.0
    if np.any(negative):
        tetrahedra[negative, 0], tetrahedra[negative, 1] = (
            tetrahedra[negative, 1].copy(),
            tetrahedra[negative, 0].copy(),
        )
    volume_mesh = VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=regions,
        region_names=dict(region_names or {}),
        metadata={
            "source": "structured-voxel",
            "voxel_shape": tuple(int(item) for item in phase.shape),
            "spacing": (dz, dy, dx),
        },
    )
    boundary, interfaces = extract_material_surfaces(volume_mesh)
    return VoxelMeshResult(
        volume_mesh=volume_mesh,
        boundary_mesh=boundary,
        interface_faces=interfaces,
        cell_shape=(nz, ny, nx),
        spacing=(dz, dy, dx),
    )



def select_volume_regions(mesh: VolumeMesh, regions: set[int] | tuple[int, ...] | list[int]) -> VolumeMesh:
    """Extract selected material cells and compact their node numbering."""

    selected = np.asarray(tuple(int(value) for value in regions), dtype=np.int32)
    if selected.size == 0:
        raise GeometryError("at least one region must be selected")
    keep = np.isin(mesh.cell_regions, selected)
    tetrahedra = mesh.tetrahedra[keep]
    cell_regions = mesh.cell_regions[keep]
    if not len(tetrahedra):
        raise GeometryError(f"none of the requested regions are present: {selected.tolist()}")
    referenced = np.unique(tetrahedra.ravel())
    remap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced), dtype=np.int64)
    return VolumeMesh(
        nodes=mesh.nodes[referenced],
        tetrahedra=remap[tetrahedra],
        cell_regions=cell_regions,
        region_names={
            int(key): value for key, value in mesh.region_names.items() if int(key) in selected
        },
        metadata={**mesh.metadata, "selected_regions": selected.tolist()},
    )


def extract_material_surfaces(
    mesh: VolumeMesh,
) -> tuple[TriangleMesh, Mapping[tuple[int, int], np.ndarray]]:
    """Extract external boundary and conforming internal material interfaces."""

    local_faces = np.asarray(
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        dtype=np.int64,
    )
    all_faces = mesh.tetrahedra[:, local_faces].reshape(-1, 3)
    owners = np.repeat(np.arange(mesh.n_cells, dtype=np.int64), 4)
    canonical = np.sort(all_faces, axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    canonical_sorted = canonical[order]
    faces_sorted = all_faces[order]
    owners_sorted = owners[order]

    boundary_faces: list[np.ndarray] = []
    interfaces: dict[tuple[int, int], list[np.ndarray]] = {}
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and np.array_equal(
            canonical_sorted[stop], canonical_sorted[start]
        ):
            stop += 1
        count = stop - start
        if count == 1:
            boundary_faces.append(faces_sorted[start])
        elif count == 2:
            left_cell = int(owners_sorted[start])
            right_cell = int(owners_sorted[start + 1])
            left_region = int(mesh.cell_regions[left_cell])
            right_region = int(mesh.cell_regions[right_cell])
            if left_region != right_region:
                pair = tuple(sorted((left_region, right_region)))
                interfaces.setdefault(pair, []).append(faces_sorted[start])
        elif count > 2:
            # Keep evidence rather than silently treating a non-manifold face as boundary.
            regions = sorted(
                {int(mesh.cell_regions[int(owner)]) for owner in owners_sorted[start:stop]}
            )
            for left, right in zip(regions[:-1], regions[1:], strict=False):
                interfaces.setdefault((left, right), []).append(faces_sorted[start])
        start = stop

    faces = (
        np.ascontiguousarray(boundary_faces, dtype=np.int64)
        if boundary_faces
        else np.empty((0, 3), dtype=np.int64)
    )
    boundary = TriangleMesh(
        vertices=mesh.nodes,
        faces=faces,
        metadata={"source": "volume-boundary"},
    )
    arrays = {
        pair: np.ascontiguousarray(values, dtype=np.int64)
        for pair, values in interfaces.items()
    }
    return boundary, arrays


__all__ = [
    "VoxelMeshResult",
    "extract_material_surfaces",
    "normalize_spacing",
    "select_volume_regions",
    "validate_voxels",
    "voxel_to_tetrahedra",
]
