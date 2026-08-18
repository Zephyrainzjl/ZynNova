"""COMSOL first-order element ordering and topology validation.

The internal ZynNova mesh containers use the widespread cyclic/VTK ordering
for Hex8 and Quad4 elements.  COMSOL's native Mesh-v4 text format uses a
tensor-product ordering for these two element families.  Treating the two
orders as interchangeable makes adjacent Hex8 cells appear on the same side
of their common face during COMSOL import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Internal Hex8 (VTK/cyclic):
#   000, 100, 110, 010, 001, 101, 111, 011
# COMSOL Mesh-v4 Hex8 (tensor product):
#   000, 100, 010, 110, 001, 101, 011, 111
COMSOL_HEX8_FROM_INTERNAL = np.asarray((0, 1, 3, 2, 4, 5, 7, 6), dtype=np.int64)

# Internal Quad4 is a perimeter cycle: 00, 10, 11, 01.
# COMSOL Mesh-v4 Quad4 is tensor product: 00, 10, 01, 11.
COMSOL_QUAD4_FROM_INTERNAL = np.asarray((0, 1, 3, 2), dtype=np.int64)

# Six positive Tet4 cells around the 000--111 body diagonal, indexing the
# COMSOL tensor-product Hex8 order above.
COMSOL_HEX8_TO_TET4 = np.asarray(
    (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    ),
    dtype=np.int64,
)

# Outward-oriented local face cycles for COMSOL's tensor-product Hex8 order.
COMSOL_HEX8_OUTWARD_FACES: dict[str, np.ndarray] = {
    "xmin": np.asarray((0, 4, 6, 2), dtype=np.int64),
    "xmax": np.asarray((1, 3, 7, 5), dtype=np.int64),
    "ymin": np.asarray((0, 1, 5, 4), dtype=np.int64),
    "ymax": np.asarray((2, 6, 7, 3), dtype=np.int64),
    "zmin": np.asarray((0, 2, 3, 1), dtype=np.int64),
    "zmax": np.asarray((4, 5, 7, 6), dtype=np.int64),
}


@dataclass(frozen=True, slots=True)
class HexTopologyValidationReport:
    """Evidence that a structured COMSOL Hex8 field is consistently oriented."""

    element_count: int
    sampled_elements: int
    shared_faces_checked: int
    same_side_shared_faces: int
    minimum_jacobian: float
    failure_coordinate: tuple[float, float, float] | None
    valid: bool
    sample_failure_coordinates: tuple[tuple[float, float, float], ...] = ()


def to_comsol_connectivity(element_type: str, connectivity: np.ndarray) -> np.ndarray:
    """Return connectivity in native COMSOL Mesh-v4 local-node order."""

    values = np.asarray(connectivity, dtype=np.int64)
    key = str(element_type).lower()
    if key in {"hex", "hex8", "hexahedron"}:
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError("Hex8 connectivity must have shape (n, 8)")
        return np.ascontiguousarray(values[:, COMSOL_HEX8_FROM_INTERNAL])
    if key in {"quad", "quad4", "quadrilateral"}:
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Quad4 connectivity must have shape (n, 4)")
        return np.ascontiguousarray(values[:, COMSOL_QUAD4_FROM_INTERNAL])
    return np.ascontiguousarray(values)


def structured_comsol_hex_rows(
    shape: Sequence[int],
    start: int,
    stop: int,
) -> np.ndarray:
    """Generate structured Hex8 rows directly in COMSOL tensor-product order."""

    nx, ny, nz = _shape3(shape)
    total = nx * ny * nz
    first = int(start)
    last = int(stop)
    if first < 0 or last < first or last > total:
        raise ValueError("invalid structured Hex8 cell range")
    ids = np.arange(first, last, dtype=np.int64)
    i, rem = np.divmod(ids, ny * nz)
    j, k = np.divmod(rem, nz)
    stride_yz = (ny + 1) * (nz + 1)
    base = i * stride_yz + j * (nz + 1) + k
    n100 = base + stride_yz
    n010 = base + (nz + 1)
    n110 = n100 + (nz + 1)
    return np.ascontiguousarray(
        np.column_stack(
            (
                base,
                n100,
                n010,
                n110,
                base + 1,
                n100 + 1,
                n010 + 1,
                n110 + 1,
            )
        ),
        dtype=np.int64,
    )



def validate_comsol_hex_connectivity(
    nodes: np.ndarray,
    elements: np.ndarray,
    *,
    maximum_shared_faces: int | None = None,
) -> HexTopologyValidationReport:
    """Validate arbitrary Hex8 rows interpreted in COMSOL Mesh-v4 order.

    This diagnostic is intentionally independent of material/domain indices.
    It therefore reproduces COMSOL's topology failure even when geometric
    entity indices are omitted.
    """

    points = np.ascontiguousarray(nodes, dtype=np.float64)
    cells = np.ascontiguousarray(elements, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("nodes must have shape (n, 3) and contain finite coordinates")
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError("elements must have shape (m, 8)")
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= len(points)):
        raise ValueError("elements contain an invalid node index")
    if not len(cells):
        return HexTopologyValidationReport(0, 0, 0, 0, 0.0, None, False)

    geometry = points[cells]
    jacobians = np.linalg.det(
        np.stack(
            (
                geometry[:, 1] - geometry[:, 0],
                geometry[:, 2] - geometry[:, 0],
                geometry[:, 4] - geometry[:, 0],
            ),
            axis=2,
        )
    )
    minimum_jacobian = float(np.min(jacobians))

    face_names = tuple(COMSOL_HEX8_OUTWARD_FACES)
    faces = np.concatenate(
        [cells[:, COMSOL_HEX8_OUTWARD_FACES[name]] for name in face_names],
        axis=0,
    )
    owners = np.concatenate(
        [np.arange(len(cells), dtype=np.int64) for _ in face_names]
    )
    canonical = np.sort(faces, axis=1)
    order = np.lexsort((canonical[:, 3], canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    canonical = canonical[order]
    faces = faces[order]
    owners = owners[order]
    changes = np.r_[True, np.any(canonical[1:] != canonical[:-1], axis=1)]
    starts = np.flatnonzero(changes)
    stops = np.r_[starts[1:], len(faces)]
    shared = np.flatnonzero((stops - starts) == 2)
    if maximum_shared_faces is not None and len(shared) > int(maximum_shared_faces):
        positions = np.unique(
            np.linspace(0, len(shared) - 1, int(maximum_shared_faces), dtype=np.int64)
        )
        shared = shared[positions]

    same_side = 0
    failure_coordinate: tuple[float, float, float] | None = None
    failure_samples: list[tuple[float, float, float]] = []
    for group_index in shared:
        start, stop = int(starts[group_index]), int(stops[group_index])
        first, second = faces[start:stop]
        first_points = points[first]
        second_points = points[second]
        first_normal = np.cross(
            first_points[1] - first_points[0],
            first_points[2] - first_points[0],
        )
        second_normal = np.cross(
            second_points[1] - second_points[0],
            second_points[2] - second_points[0],
        )
        # A shared face must be traversed in opposite directions by its owners.
        # The centroid-side test catches warped faces whose first triangle alone
        # is nearly singular.
        owner_a, owner_b = int(owners[start]), int(owners[start + 1])
        face_center = np.mean(first_points, axis=0)
        centroid_a = np.mean(geometry[owner_a], axis=0)
        centroid_b = np.mean(geometry[owner_b], axis=0)
        opposite_normals = float(np.dot(first_normal, second_normal)) < 0.0
        opposite_sides = (
            float(np.dot(first_normal, centroid_a - face_center))
            * float(np.dot(first_normal, centroid_b - face_center))
            < 0.0
        )
        if not (opposite_normals and opposite_sides):
            same_side += 1
            coordinate = tuple(map(float, face_center))
            if failure_coordinate is None:
                failure_coordinate = coordinate
            if len(failure_samples) < 32:
                failure_samples.append(coordinate)

    return HexTopologyValidationReport(
        element_count=int(len(cells)),
        sampled_elements=int(len(cells)),
        shared_faces_checked=int(len(shared)),
        same_side_shared_faces=int(same_side),
        minimum_jacobian=minimum_jacobian,
        failure_coordinate=failure_coordinate,
        valid=bool(minimum_jacobian > 0.0 and same_side == 0),
        sample_failure_coordinates=tuple(failure_samples),
    )

def validate_structured_hex_topology(
    shape: Sequence[int],
    *,
    spacing: float | Sequence[float] = 1.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    maximum_shared_faces: int = 100_000,
) -> HexTopologyValidationReport:
    """Validate Jacobians and opposite-sided ownership of shared Hex8 faces.

    The check is deterministic.  Small meshes are checked exhaustively; large
    meshes use evenly spaced samples along each adjacency family so validation
    remains bounded-memory and bounded-time.
    """

    nx, ny, nz = _shape3(shape)
    dx, dy, dz = _positive3(spacing, name="spacing")
    ox, oy, oz = _finite3(origin, name="origin")
    total = nx * ny * nz

    local = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (dx, 0.0, 0.0),
            (0.0, dy, 0.0),
            (dx, dy, 0.0),
            (0.0, 0.0, dz),
            (dx, 0.0, dz),
            (0.0, dy, dz),
            (dx, dy, dz),
        ),
        dtype=np.float64,
    )
    jacobian = float(np.linalg.det(np.column_stack((local[1], local[2], local[4]))))
    minimum_jacobian = jacobian
    failure_coordinate: tuple[float, float, float] | None = None
    failure_samples: list[tuple[float, float, float]] = []

    families = (
        (0, (max(nx - 1, 0), ny, nz), "xmax", "xmin"),
        (1, (nx, max(ny - 1, 0), nz), "ymax", "ymin"),
        (2, (nx, ny, max(nz - 1, 0)), "zmax", "zmin"),
    )
    total_shared = sum(int(np.prod(size, dtype=np.int64)) for _, size, _, _ in families)
    budget = max(0, int(maximum_shared_faces))
    shared_checked = 0
    same_side = 0

    def cell_id(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    def node_coordinates(node_ids: np.ndarray) -> np.ndarray:
        yz = (ny + 1) * (nz + 1)
        ii, rem = np.divmod(node_ids, yz)
        jj, kk = np.divmod(rem, nz + 1)
        return np.column_stack((ox + dx * ii, oy + dy * jj, oz + dz * kk))

    remaining_budget = min(total_shared, budget) if budget else 0
    for axis, family_shape, left_name, right_name in families:
        family_count = int(np.prod(family_shape, dtype=np.int64))
        if family_count == 0 or remaining_budget == 0:
            continue
        # Allocate the remaining global budget proportionally, while ensuring
        # that every non-empty family receives at least one check.
        share = max(1, min(family_count, round(remaining_budget * family_count / max(total_shared, 1))))
        flat_ids = (
            np.arange(family_count, dtype=np.int64)
            if share >= family_count
            else np.unique(np.linspace(0, family_count - 1, share, dtype=np.int64))
        )
        a_size, b_size, c_size = family_shape
        del a_size  # only b/c strides are needed below
        aa, rem = np.divmod(flat_ids, b_size * c_size)
        bb, cc = np.divmod(rem, c_size)
        for a, b, c in zip(aa, bb, cc, strict=True):
            i, j, k = int(a), int(b), int(c)
            neighbour = [i, j, k]
            neighbour[axis] += 1
            left = structured_comsol_hex_rows((nx, ny, nz), cell_id(i, j, k), cell_id(i, j, k) + 1)[0]
            right_id = cell_id(neighbour[0], neighbour[1], neighbour[2])
            right = structured_comsol_hex_rows((nx, ny, nz), right_id, right_id + 1)[0]
            left_face = left[COMSOL_HEX8_OUTWARD_FACES[left_name]]
            right_face = right[COMSOL_HEX8_OUTWARD_FACES[right_name]]
            if not np.array_equal(np.sort(left_face), np.sort(right_face)):
                same_side += 1
                if failure_coordinate is None:
                    points = node_coordinates(left_face)
                    failure_coordinate = tuple(map(float, np.mean(points, axis=0)))
                if len(failure_samples) < 32:
                    failure_samples.append(tuple(map(float, np.mean(points, axis=0))))
                continue
            lp = node_coordinates(left_face)
            rp = node_coordinates(right_face)
            ln = np.cross(lp[1] - lp[0], lp[2] - lp[0])
            rn = np.cross(rp[1] - rp[0], rp[2] - rp[0])
            if float(np.dot(ln, rn)) >= 0.0:
                same_side += 1
                coordinate = tuple(map(float, np.mean(lp, axis=0)))
                if failure_coordinate is None:
                    failure_coordinate = coordinate
                if len(failure_samples) < 32:
                    failure_samples.append(coordinate)
            shared_checked += 1
        remaining_budget = max(0, remaining_budget - len(flat_ids))

    return HexTopologyValidationReport(
        element_count=total,
        sampled_elements=min(total, max(1, shared_checked * 2) if total else 0),
        shared_faces_checked=shared_checked,
        same_side_shared_faces=same_side,
        minimum_jacobian=minimum_jacobian,
        failure_coordinate=failure_coordinate,
        valid=bool(total > 0 and jacobian > 0.0 and same_side == 0),
        sample_failure_coordinates=tuple(failure_samples),
    )


def _shape3(value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or min(result) < 1:
        raise ValueError("shape must contain three positive integers")
    return result  # type: ignore[return-value]


def _finite3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


def _positive3(value: float | Sequence[float], *, name: str) -> tuple[float, float, float]:
    result = (float(value),) * 3 if np.isscalar(value) else _finite3(value, name=name)
    if min(result) <= 0.0:
        raise ValueError(f"{name} must contain three positive values")
    return result  # type: ignore[return-value]


__all__ = [
    "COMSOL_HEX8_FROM_INTERNAL",
    "COMSOL_HEX8_OUTWARD_FACES",
    "COMSOL_HEX8_TO_TET4",
    "COMSOL_QUAD4_FROM_INTERNAL",
    "HexTopologyValidationReport",
    "structured_comsol_hex_rows",
    "to_comsol_connectivity",
    "validate_comsol_hex_connectivity",
    "validate_structured_hex_topology",
]
