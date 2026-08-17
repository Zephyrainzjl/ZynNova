"""Voxel-labelled microstructures and conforming tetrahedral meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..exceptions import MeshError
from .mesh import Mesh, box_tetrahedral_mesh


@dataclass(frozen=True, slots=True)
class VoxelMeshResult:
    """A tetrahedralized voxel microstructure and exact voxel statistics."""

    mesh: Mesh
    voxel_size_m: tuple[float, float, float]
    phase_volume_fractions: Mapping[int, float]
    interface_area_m2: Mapping[tuple[int, int], float]

    @property
    def volume_m3(self) -> float:
        return float(np.sum(self.mesh.cell_volumes()))

    def specific_interface_area_m_inv(
        self,
        phase_a: int,
        phase_b: int,
    ) -> float:
        key = tuple(sorted((int(phase_a), int(phase_b))))
        return float(self.interface_area_m2.get(key, 0.0) / self.volume_m3)


def voxel_to_tetrahedral_mesh(
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | tuple[float, float, float] = 1.0,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VoxelMeshResult:
    """Convert a 3-D integer phase image to a conforming Tet4 mesh.

    Every voxel is split into six tetrahedra using the globally consistent
    Freudenthal diagonal.  Phase labels are copied to all six tetrahedra, so
    there are no mixed or ambiguous cells at phase boundaries.
    """

    labels = np.asarray(phase_labels)
    if labels.ndim != 3 or min(labels.shape, default=0) < 1:
        raise MeshError("phase_labels must be a non-empty 3-D array")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
            raise MeshError("phase_labels must contain finite integer values")
    labels = np.asarray(labels, dtype=np.int32)

    spacing = _voxel_spacing(voxel_size_m)
    shape = tuple(map(int, labels.shape))
    lengths = tuple(shape[axis] * spacing[axis] for axis in range(3))
    base = box_tetrahedral_mesh(lengths, shape, origin=origin_m)
    regions = np.repeat(labels.reshape(-1), 6)
    mesh = Mesh(
        nodes=base.nodes,
        cells=base.cells,
        cell_regions=regions,
        boundary_faces=base.boundary_faces,
        metadata={
            **base.metadata,
            "source": "voxel_to_tetrahedral_mesh",
            "voxel_shape": shape,
            "voxel_size_m": spacing,
            "phase_labels": sorted(map(int, np.unique(labels))),
        },
    )
    values, counts = np.unique(labels, return_counts=True)
    total = float(labels.size)
    fractions = {
        int(value): float(count / total) for value, count in zip(values, counts, strict=True)
    }
    return VoxelMeshResult(
        mesh=mesh,
        voxel_size_m=spacing,
        phase_volume_fractions=fractions,
        interface_area_m2=voxel_interface_areas(labels, voxel_size_m=spacing),
    )


def voxel_interface_areas(
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | tuple[float, float, float] = 1.0,
) -> dict[tuple[int, int], float]:
    """Return face-based phase-pair interface areas from a labelled image."""

    labels = np.asarray(phase_labels)
    if labels.ndim != 3:
        raise MeshError("phase_labels must be a 3-D array")
    spacing = _voxel_spacing(voxel_size_m)
    face_areas = (
        spacing[1] * spacing[2],
        spacing[0] * spacing[2],
        spacing[0] * spacing[1],
    )
    result: dict[tuple[int, int], float] = {}
    for axis, face_area in enumerate(face_areas):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        a = labels[tuple(lower)]
        b = labels[tuple(upper)]
        changed = a != b
        if not np.any(changed):
            continue
        pairs = np.stack((a[changed], b[changed]), axis=1).astype(np.int64)
        pairs.sort(axis=1)
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        for pair, count in zip(unique, counts, strict=True):
            key = (int(pair[0]), int(pair[1]))
            result[key] = result.get(key, 0.0) + float(count * face_area)
    return result


def _voxel_spacing(
    value: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise MeshError("voxel_size_m must contain one or three positive values")
    return tuple(map(float, raw))


__all__ = [
    "VoxelMeshResult",
    "voxel_interface_areas",
    "voxel_to_tetrahedral_mesh",
]
