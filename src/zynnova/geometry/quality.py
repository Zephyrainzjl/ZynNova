"""Geometry quality, topology, and FEM-readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .types import TriangleMesh, VolumeMesh


@dataclass(frozen=True, slots=True)
class TriangleQuality:
    minimum_area: float
    median_area: float
    maximum_area: float
    degenerate_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    watertight: bool


@dataclass(frozen=True, slots=True)
class TetraQuality:
    minimum_volume: float
    median_volume: float
    maximum_volume: float
    inverted_cells: int
    degenerate_cells: int
    minimum_mean_ratio: float
    median_mean_ratio: float
    fem_ready: bool


def triangle_areas(mesh: TriangleMesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def edge_incidence(mesh: TriangleMesh) -> Mapping[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(left, right)), int(max(left, right)))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def triangle_quality(
    mesh: TriangleMesh,
    *,
    tolerance: float | None = None,
) -> TriangleQuality:
    """Return scale-invariant surface quality diagnostics.

    ``tolerance=None`` derives an area threshold from the median non-zero edge
    length. This avoids rejecting physically valid nanometre-scale meshes merely
    because their coordinates are expressed in SI metres.
    """

    areas = triangle_areas(mesh)
    if tolerance is None:
        triangles = mesh.vertices[mesh.faces]
        lengths = np.concatenate(
            [
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ]
        )
        positive = lengths[lengths > 0.0]
        characteristic = float(np.median(positive)) if len(positive) else 1.0
        tolerance = max(1.0e-12 * characteristic**2, np.finfo(float).tiny)
    elif tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    edges = edge_incidence(mesh)
    boundary = sum(count == 1 for count in edges.values())
    nonmanifold = sum(count > 2 for count in edges.values())
    if len(areas):
        minimum, median, maximum = (
            float(np.min(areas)),
            float(np.median(areas)),
            float(np.max(areas)),
        )
    else:
        minimum = median = maximum = 0.0
    return TriangleQuality(
        minimum_area=minimum,
        median_area=median,
        maximum_area=maximum,
        degenerate_faces=int(np.count_nonzero(areas <= tolerance)),
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        watertight=bool(edges) and boundary == 0 and nonmanifold == 0,
    )


def tetrahedron_signed_volumes(mesh: VolumeMesh) -> np.ndarray:
    points = mesh.nodes[mesh.tetrahedra]
    matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=1,
    )
    return np.linalg.det(matrices) / 6.0


def tetrahedron_mean_ratio(mesh: VolumeMesh) -> np.ndarray:
    points = mesh.nodes[mesh.tetrahedra]
    signed = tetrahedron_signed_volumes(mesh)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_square_sum = np.zeros(len(points), dtype=np.float64)
    for left, right in edge_pairs:
        edge_square_sum += np.sum((points[:, left] - points[:, right]) ** 2, axis=1)
    numerator = 12.0 * np.power(3.0 * np.maximum(signed, 0.0), 2.0 / 3.0)
    return np.divide(
        numerator,
        edge_square_sum,
        out=np.zeros_like(numerator),
        where=edge_square_sum > 0.0,
    )


def tetra_quality(
    mesh: VolumeMesh,
    *,
    tolerance: float | None = None,
) -> TetraQuality:
    """Return orientation and shape evidence with a scale-adaptive volume threshold.

    A fixed absolute threshold is invalid for SI-scale microstructures: a healthy
    tetrahedron inside a 100 nm voxel has volume around ``1e-22 m^3``. The default
    therefore derives a numerical-degeneracy threshold from the median local edge
    length cubed.
    """

    signed = tetrahedron_signed_volumes(mesh)
    if tolerance is None:
        points = mesh.nodes[mesh.tetrahedra]
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        lengths = np.concatenate(
            [np.linalg.norm(points[:, left] - points[:, right], axis=1) for left, right in edge_pairs]
        )
        positive = lengths[lengths > 0.0]
        characteristic = float(np.median(positive)) if len(positive) else 1.0
        tolerance = max(1.0e-12 * characteristic**3, np.finfo(float).tiny)
    elif tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    absolute = np.abs(signed)
    ratios = tetrahedron_mean_ratio(mesh)
    if len(absolute):
        minimum, median, maximum = (
            float(np.min(absolute)),
            float(np.median(absolute)),
            float(np.max(absolute)),
        )
        ratio_min, ratio_median = float(np.min(ratios)), float(np.median(ratios))
    else:
        minimum = median = maximum = ratio_min = ratio_median = 0.0
    inverted = int(np.count_nonzero(signed < -tolerance))
    degenerate = int(np.count_nonzero(absolute <= tolerance))
    return TetraQuality(
        minimum_volume=minimum,
        median_volume=median,
        maximum_volume=maximum,
        inverted_cells=inverted,
        degenerate_cells=degenerate,
        minimum_mean_ratio=ratio_min,
        median_mean_ratio=ratio_median,
        fem_ready=bool(len(signed)) and inverted == 0 and degenerate == 0,
    )


__all__ = [
    "TetraQuality",
    "TriangleQuality",
    "edge_incidence",
    "tetra_quality",
    "tetrahedron_mean_ratio",
    "tetrahedron_signed_volumes",
    "triangle_areas",
    "triangle_quality",
]
