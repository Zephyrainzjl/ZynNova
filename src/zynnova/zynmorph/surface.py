"""Conforming multi-material PLC extraction and topology-safe smoothing.

The TetGen path must not tetrahedralize each label independently.  This module
builds one global piecewise-linear complex (PLC): every exterior or material
interface is represented exactly once, while the phase on both sides of every
triangle remains explicit.  That is the representation required for a
region-partitioned, interface-conforming tetrahedral mesh.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.exceptions import GeometryError
from ..geometry import TriangleMesh
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class MultiphasePLC:
    """Triangular PLC with a material label on both sides of every facet."""

    vertices: np.ndarray
    triangles: np.ndarray
    facet_markers: np.ndarray
    left_regions: np.ndarray
    right_regions: np.ndarray
    outside_region: int
    marker_names: Mapping[int, str] = field(default_factory=dict)
    marker_region_pairs: Mapping[int, tuple[int | None, int | None]] = field(
        default_factory=dict
    )
    fixed_vertices: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        markers = np.ascontiguousarray(self.facet_markers, dtype=np.int32)
        left = np.ascontiguousarray(self.left_regions, dtype=np.int32)
        right = np.ascontiguousarray(self.right_regions, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise GeometryError("PLC vertices must have shape (N, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise GeometryError("PLC triangles must have shape (M, 3)")
        if triangles.size and (
            int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices)
        ):
            raise GeometryError("PLC triangle contains an out-of-range vertex")
        expected = (len(triangles),)
        for name, value in (
            ("facet_markers", markers),
            ("left_regions", left),
            ("right_regions", right),
        ):
            if value.shape != expected:
                raise GeometryError(f"{name} must have shape {expected}")
        if np.any(left == right):
            raise GeometryError("every PLC triangle must separate two distinct regions")
        if not np.all(np.isfinite(vertices)):
            raise GeometryError("PLC contains non-finite coordinates")
        if self.fixed_vertices is None:
            fixed = np.zeros(len(vertices), dtype=bool)
        else:
            fixed = np.ascontiguousarray(self.fixed_vertices, dtype=bool)
            if fixed.shape != (len(vertices),):
                raise GeometryError("fixed_vertices must have one boolean per PLC vertex")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "facet_markers", markers)
        object.__setattr__(self, "left_regions", left)
        object.__setattr__(self, "right_regions", right)
        object.__setattr__(self, "outside_region", int(self.outside_region))
        object.__setattr__(
            self, "marker_names", {int(key): str(value) for key, value in self.marker_names.items()}
        )
        object.__setattr__(
            self,
            "marker_region_pairs",
            {
                int(key): (
                    None if value[0] is None else int(value[0]),
                    None if value[1] is None else int(value[1]),
                )
                for key, value in self.marker_region_pairs.items()
            },
        )
        object.__setattr__(self, "fixed_vertices", fixed)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def surface_mesh(self) -> TriangleMesh:
        return TriangleMesh(
            vertices=self.vertices,
            faces=self.triangles,
            face_materials=self.facet_markers,
            metadata={**self.metadata, "source": "multiphase-plc"},
        )

    @property
    def regions(self) -> tuple[int, ...]:
        values = np.unique(np.concatenate((self.left_regions, self.right_regions)))
        return tuple(int(value) for value in values if int(value) != self.outside_region)


@dataclass(frozen=True, slots=True)
class SurfacePLCAudit:
    n_vertices: int
    n_triangles: int
    degenerate_faces: int
    duplicate_faces: int
    unreferenced_vertices: int
    open_region_edges: int
    nonmanifold_region_edges: int
    orientation_conflicts: int
    region_edge_issues: Mapping[int, tuple[int, int, int]]
    valid: bool


@dataclass(frozen=True, slots=True)
class JunctionRegularizationReport:
    ambiguous_edges_before: int
    ambiguous_edges_after: int
    changed_voxels: int
    changed_fraction: float
    iterations: int
    converged: bool
    phase_counts_before: Mapping[int, int]
    phase_counts_after: Mapping[int, int]
    changes: tuple[tuple[int, int, int, int, int], ...]
    initial_change_budget_fraction: float = 0.0
    final_change_budget_fraction: float = 0.0
    hard_change_budget_fraction: float = 0.0
    budget_expansions: int = 0
    termination_reason: str = "unknown"
    maximum_phase_fraction_drift: float = 0.0


@dataclass(frozen=True, slots=True)
class _QuadBatch:
    grid_corners_zyx: np.ndarray
    left_regions: np.ndarray
    right_regions: np.ndarray
    marker_keys: tuple[tuple[object, ...], ...]
    parity: np.ndarray


def extract_multiphase_plc(
    volume: MicrostructureVolume,
    *,
    checkerboard_diagonals: bool = True,
    preserve_outer_boundary: bool = True,
    preserve_multiphase_junctions: bool = True,
    strict: bool = True,
) -> MultiphasePLC:
    """Extract one conforming PLC from a ``(z, y, x)`` label volume.

    Internal interfaces are emitted once, not once per material.  Triangle
    normals point from ``left_regions`` to ``right_regions``.  Alternating
    quad diagonals suppress a global directional bias without breaking
    conformity because every material interface has only one owner here.
    """

    labels = np.ascontiguousarray(volume.labels, dtype=np.int32)
    nz, ny, nx = map(int, labels.shape)
    outside = int(labels.min()) - 1
    if outside in set(map(int, np.unique(labels))):  # defensive for exotic integers
        outside = int(labels.max()) + 1

    batches: list[_QuadBatch] = []

    def add_batch(
        corners: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        keys: Sequence[tuple[object, ...]],
        parity: np.ndarray,
    ) -> None:
        if not len(corners):
            return
        batches.append(
            _QuadBatch(
                np.ascontiguousarray(corners, dtype=np.int64),
                np.ascontiguousarray(left, dtype=np.int32),
                np.ascontiguousarray(right, dtype=np.int32),
                tuple(keys),
                np.ascontiguousarray(parity, dtype=np.int8),
            )
        )

    # Internal +z faces.
    mask = labels[:-1, :, :] != labels[1:, :, :]
    iz, iy, ix = np.nonzero(mask)
    if len(iz):
        z = iz + 1
        corners = np.stack(
            (
                np.column_stack((z, iy, ix)),
                np.column_stack((z, iy, ix + 1)),
                np.column_stack((z, iy + 1, ix + 1)),
                np.column_stack((z, iy + 1, ix)),
            ),
            axis=1,
        )
        left = labels[iz, iy, ix]
        right = labels[iz + 1, iy, ix]
        keys = [("interface", min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(left, right, strict=True)]
        add_batch(corners, left, right, keys, (iz + iy + ix) & 1)

    # Internal +y faces.
    mask = labels[:, :-1, :] != labels[:, 1:, :]
    iz, iy, ix = np.nonzero(mask)
    if len(iz):
        y = iy + 1
        corners = np.stack(
            (
                np.column_stack((iz, y, ix)),
                np.column_stack((iz + 1, y, ix)),
                np.column_stack((iz + 1, y, ix + 1)),
                np.column_stack((iz, y, ix + 1)),
            ),
            axis=1,
        )
        left = labels[iz, iy, ix]
        right = labels[iz, iy + 1, ix]
        keys = [("interface", min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(left, right, strict=True)]
        add_batch(corners, left, right, keys, (iz + iy + ix) & 1)

    # Internal +x faces.
    mask = labels[:, :, :-1] != labels[:, :, 1:]
    iz, iy, ix = np.nonzero(mask)
    if len(iz):
        x = ix + 1
        corners = np.stack(
            (
                np.column_stack((iz, iy, x)),
                np.column_stack((iz, iy + 1, x)),
                np.column_stack((iz + 1, iy + 1, x)),
                np.column_stack((iz + 1, iy, x)),
            ),
            axis=1,
        )
        left = labels[iz, iy, ix]
        right = labels[iz, iy, ix + 1]
        keys = [("interface", min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(left, right, strict=True)]
        add_batch(corners, left, right, keys, (iz + iy + ix) & 1)

    def exterior_z(z: int, side: str, phase: np.ndarray, left_is_outside: bool) -> None:
        yy, xx = np.indices((ny, nx))
        yy = yy.ravel()
        xx = xx.ravel()
        zz = np.full_like(yy, z)
        corners = np.stack(
            (
                np.column_stack((zz, yy, xx)),
                np.column_stack((zz, yy, xx + 1)),
                np.column_stack((zz, yy + 1, xx + 1)),
                np.column_stack((zz, yy + 1, xx)),
            ),
            axis=1,
        )
        values = phase.ravel()
        left = np.full(len(values), outside, dtype=np.int32) if left_is_outside else values
        right = values if left_is_outside else np.full(len(values), outside, dtype=np.int32)
        keys = [("exterior", side, int(value)) for value in values]
        add_batch(corners, left, right, keys, (yy + xx) & 1)

    def exterior_y(y: int, side: str, phase: np.ndarray, left_is_outside: bool) -> None:
        zz, xx = np.indices((nz, nx))
        zz = zz.ravel()
        xx = xx.ravel()
        yy = np.full_like(zz, y)
        corners = np.stack(
            (
                np.column_stack((zz, yy, xx)),
                np.column_stack((zz + 1, yy, xx)),
                np.column_stack((zz + 1, yy, xx + 1)),
                np.column_stack((zz, yy, xx + 1)),
            ),
            axis=1,
        )
        values = phase.ravel()
        left = np.full(len(values), outside, dtype=np.int32) if left_is_outside else values
        right = values if left_is_outside else np.full(len(values), outside, dtype=np.int32)
        keys = [("exterior", side, int(value)) for value in values]
        add_batch(corners, left, right, keys, (zz + xx) & 1)

    def exterior_x(x: int, side: str, phase: np.ndarray, left_is_outside: bool) -> None:
        zz, yy = np.indices((nz, ny))
        zz = zz.ravel()
        yy = yy.ravel()
        xx = np.full_like(zz, x)
        corners = np.stack(
            (
                np.column_stack((zz, yy, xx)),
                np.column_stack((zz, yy + 1, xx)),
                np.column_stack((zz + 1, yy + 1, xx)),
                np.column_stack((zz + 1, yy, xx)),
            ),
            axis=1,
        )
        values = phase.ravel()
        left = np.full(len(values), outside, dtype=np.int32) if left_is_outside else values
        right = values if left_is_outside else np.full(len(values), outside, dtype=np.int32)
        keys = [("exterior", side, int(value)) for value in values]
        add_batch(corners, left, right, keys, (zz + yy) & 1)

    exterior_z(0, "zmin", labels[0], True)
    exterior_z(nz, "zmax", labels[-1], False)
    exterior_y(0, "ymin", labels[:, 0, :], True)
    exterior_y(ny, "ymax", labels[:, -1, :], False)
    exterior_x(0, "xmin", labels[:, :, 0], True)
    exterior_x(nx, "xmax", labels[:, :, -1], False)

    if not batches:
        raise GeometryError("cannot extract an empty PLC")

    all_keys = sorted(
        {key for batch in batches for key in batch.marker_keys},
        key=lambda item: tuple(str(value) for value in item),
    )
    marker_for_key = {key: index for index, key in enumerate(all_keys, start=1)}
    marker_names: dict[int, str] = {}
    marker_pairs: dict[int, tuple[int | None, int | None]] = {}
    for key, marker in marker_for_key.items():
        if key[0] == "interface":
            a, b = int(key[1]), int(key[2])
            marker_names[marker] = f"interface_{_phase_name(volume, a)}_{_phase_name(volume, b)}"
            marker_pairs[marker] = (a, b)
        else:
            side, phase = str(key[1]), int(key[2])
            marker_names[marker] = f"{side}_{_phase_name(volume, phase)}"
            marker_pairs[marker] = (phase, None)

    grid_quads = np.concatenate([batch.grid_corners_zyx for batch in batches], axis=0)
    left_quads = np.concatenate([batch.left_regions for batch in batches])
    right_quads = np.concatenate([batch.right_regions for batch in batches])
    parity = np.concatenate([batch.parity for batch in batches])
    quad_markers = np.fromiter(
        (
            marker_for_key[key]
            for batch in batches
            for key in batch.marker_keys
        ),
        dtype=np.int32,
        count=len(grid_quads),
    )

    flat_grid = grid_quads.reshape(-1, 3)
    unique_grid, inverse = np.unique(flat_grid, axis=0, return_inverse=True)
    quad_indices = inverse.reshape(-1, 4)
    dz, dy, dx = map(float, volume.voxel_size_m)
    oz, oy, ox = map(float, volume.origin_m)
    vertices = np.column_stack(
        (
            ox + unique_grid[:, 2] * dx,
            oy + unique_grid[:, 1] * dy,
            oz + unique_grid[:, 0] * dz,
        )
    )

    use_alternate = checkerboard_diagonals & (parity.astype(bool))
    triangles = np.empty((2 * len(quad_indices), 3), dtype=np.int64)
    primary = ~use_alternate
    q = quad_indices[primary]
    primary_rows = np.nonzero(primary)[0]
    triangles[2 * primary_rows] = q[:, [0, 1, 2]]
    triangles[2 * primary_rows + 1] = q[:, [0, 2, 3]]
    q = quad_indices[use_alternate]
    alternate_rows = np.nonzero(use_alternate)[0]
    triangles[2 * alternate_rows] = q[:, [0, 1, 3]]
    triangles[2 * alternate_rows + 1] = q[:, [1, 2, 3]]

    facet_markers = np.repeat(quad_markers, 2)
    left_regions = np.repeat(left_quads, 2)
    right_regions = np.repeat(right_quads, 2)

    fixed = np.zeros(len(vertices), dtype=bool)
    if preserve_outer_boundary:
        fixed |= np.any(
            (unique_grid == np.array((0, 0, 0)))
            | (unique_grid == np.array((nz, ny, nx))),
            axis=1,
        )
    if preserve_multiphase_junctions:
        repeated_vertices = np.repeat(triangles.reshape(-1), 2)
        face_regions = np.column_stack((left_regions, right_regions))
        repeated_regions = np.repeat(face_regions, 3, axis=0).reshape(-1)
        incidence = np.column_stack((repeated_vertices, repeated_regions))
        unique_incidence = np.unique(incidence, axis=0)
        region_counts = np.bincount(unique_incidence[:, 0], minlength=len(vertices))
        fixed |= region_counts > 2

    plc = MultiphasePLC(
        vertices=vertices,
        triangles=triangles,
        facet_markers=facet_markers,
        left_regions=left_regions,
        right_regions=right_regions,
        outside_region=outside,
        marker_names=marker_names,
        marker_region_pairs=marker_pairs,
        fixed_vertices=fixed,
        metadata={
            "source": "multiphase-voxel-plc",
            "voxel_shape_zyx": tuple(map(int, labels.shape)),
            "voxel_size_m_zyx": tuple(map(float, volume.voxel_size_m)),
            "origin_m_zyx": tuple(map(float, volume.origin_m)),
            "checkerboard_diagonals": bool(checkerboard_diagonals),
            "preserve_outer_boundary": bool(preserve_outer_boundary),
            "preserve_multiphase_junctions": bool(preserve_multiphase_junctions),
        },
    )
    audit = audit_multiphase_plc(plc)
    if strict and not audit.valid:
        raise GeometryError(
            "multi-material PLC is not a closed manifold for every region: "
            f"degenerate={audit.degenerate_faces}, duplicates={audit.duplicate_faces}, "
            f"open_region_edges={audit.open_region_edges}, "
            f"nonmanifold_region_edges={audit.nonmanifold_region_edges}, "
            f"orientation_conflicts={audit.orientation_conflicts}. "
            "Run regularize_nonmanifold_junctions() before TetGen meshing."
        )
    return plc


def audit_multiphase_plc(
    plc: MultiphasePLC,
    *,
    area_tolerance: float | None = None,
) -> SurfacePLCAudit:
    """Audit every phase shell independently, including edge orientation."""

    triangles_xyz = plc.vertices[plc.triangles]
    normals = np.cross(
        triangles_xyz[:, 1] - triangles_xyz[:, 0],
        triangles_xyz[:, 2] - triangles_xyz[:, 0],
    )
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    if area_tolerance is None:
        edge_lengths = np.concatenate(
            (
                np.linalg.norm(triangles_xyz[:, 1] - triangles_xyz[:, 0], axis=1),
                np.linalg.norm(triangles_xyz[:, 2] - triangles_xyz[:, 1], axis=1),
                np.linalg.norm(triangles_xyz[:, 0] - triangles_xyz[:, 2], axis=1),
            )
        )
        positive = edge_lengths[edge_lengths > 0.0]
        characteristic = float(np.median(positive)) if len(positive) else 1.0
        area_tolerance = max(
            np.finfo(float).eps * 4096.0 * characteristic**2,
            np.finfo(float).tiny,
        )
    if area_tolerance < 0.0:
        raise ValueError("area_tolerance cannot be negative")
    degenerate = int(np.count_nonzero(areas <= area_tolerance))

    canonical = np.sort(plc.triangles, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    duplicate = int(np.sum(np.maximum(counts - 1, 0)))
    referenced = np.unique(plc.triangles.ravel())
    unreferenced = int(len(plc.vertices) - len(referenced))

    issues: dict[int, tuple[int, int, int]] = {}
    open_total = nonmanifold_total = orientation_total = 0
    for region in plc.regions:
        left_mask = plc.left_regions == region
        right_mask = plc.right_regions == region
        phase_faces = np.concatenate(
            (
                plc.triangles[left_mask],
                plc.triangles[right_mask][:, [0, 2, 1]],
            ),
            axis=0,
        )
        if not len(phase_faces):
            issues[region] = (0, 0, 0)
            continue
        directed = np.concatenate(
            (
                phase_faces[:, [0, 1]],
                phase_faces[:, [1, 2]],
                phase_faces[:, [2, 0]],
            ),
            axis=0,
        )
        canonical_edges = np.sort(directed, axis=1)
        unique_edges, inverse, edge_counts = np.unique(
            canonical_edges, axis=0, return_inverse=True, return_counts=True
        )
        del unique_edges
        open_edges = int(np.count_nonzero(edge_counts == 1))
        nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
        signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
        signed_balance = np.bincount(
            inverse, weights=signs, minlength=len(edge_counts)
        )
        orientation_conflicts = int(
            np.count_nonzero((edge_counts == 2) & (np.abs(signed_balance) > 0.5))
        )
        issues[region] = (open_edges, nonmanifold_edges, orientation_conflicts)
        open_total += open_edges
        nonmanifold_total += nonmanifold_edges
        orientation_total += orientation_conflicts

    valid = (
        degenerate == 0
        and duplicate == 0
        and unreferenced == 0
        and open_total == 0
        and nonmanifold_total == 0
        and orientation_total == 0
    )
    return SurfacePLCAudit(
        n_vertices=int(len(plc.vertices)),
        n_triangles=int(len(plc.triangles)),
        degenerate_faces=degenerate,
        duplicate_faces=duplicate,
        unreferenced_vertices=unreferenced,
        open_region_edges=open_total,
        nonmanifold_region_edges=nonmanifold_total,
        orientation_conflicts=orientation_total,
        region_edge_issues=issues,
        valid=valid,
    )




def lock_plc_interfaces(
    plc: MultiphasePLC,
    region_pairs: Sequence[tuple[int, int]],
    *,
    require_present: bool = True,
) -> MultiphasePLC:
    """Freeze all vertices on selected material interfaces before smoothing.

    The region-pair order is ignored.  This is useful for geometrically exact
    interfaces such as cathode/separator and separator/anode planes, but is
    intentionally generic and works for any pair of material IDs in any
    complex PLC.
    """

    normalized = tuple(
        sorted(
            {
                tuple(sorted((int(pair[0]), int(pair[1]))))
                for pair in region_pairs
                if len(pair) == 2 and int(pair[0]) != int(pair[1])
            }
        )
    )
    if len(normalized) != len(region_pairs):
        for pair in region_pairs:
            if len(pair) != 2 or int(pair[0]) == int(pair[1]):
                raise ValueError(
                    "each locked interface must contain two distinct region IDs"
                )
    if not normalized:
        return plc

    fixed = np.asarray(plc.fixed_vertices, dtype=bool).copy()
    matched_pairs: set[tuple[int, int]] = set()
    matched_faces = 0
    wanted = set(normalized)
    for face_index, (left, right) in enumerate(
        zip(plc.left_regions, plc.right_regions, strict=True)
    ):
        pair = tuple(sorted((int(left), int(right))))
        if pair not in wanted:
            continue
        fixed[plc.triangles[face_index]] = True
        matched_pairs.add(pair)
        matched_faces += 1

    missing = wanted - matched_pairs
    if missing and require_present:
        raise GeometryError(
            "requested locked PLC interfaces are absent: "
            f"{sorted(missing)}"
        )

    return MultiphasePLC(
        vertices=plc.vertices,
        triangles=plc.triangles,
        facet_markers=plc.facet_markers,
        left_regions=plc.left_regions,
        right_regions=plc.right_regions,
        outside_region=plc.outside_region,
        marker_names=plc.marker_names,
        marker_region_pairs=plc.marker_region_pairs,
        fixed_vertices=fixed,
        metadata={
            **plc.metadata,
            "locked_interface_pairs": normalized,
            "locked_interface_faces": int(matched_faces),
            "locked_interface_vertices": int(np.count_nonzero(fixed)),
        },
    )


def smooth_multiphase_plc(
    plc: MultiphasePLC,
    *,
    iterations: int = 8,
    relaxation: float = 0.34,
    taubin_mu: float = -0.36,
    maximum_displacement_m: float | None = None,
    line_search_steps: int = 10,
) -> MultiphasePLC:
    """Taubin-smooth movable PLC vertices without changing connectivity.

    Outer boundaries and multiphase junctions are fixed by the extractor.  A
    line search rejects any step that collapses or flips an input triangle.
    """

    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    if not 0.0 <= relaxation < 1.0:
        raise ValueError("relaxation must lie in [0, 1)")
    if not -1.0 < taubin_mu <= 0.0:
        raise ValueError("taubin_mu must lie in (-1, 0]")
    if maximum_displacement_m is not None and maximum_displacement_m <= 0.0:
        raise ValueError("maximum_displacement_m must be positive")
    if line_search_steps < 1:
        raise ValueError("line_search_steps must be positive")
    if iterations == 0 or relaxation == 0.0:
        return plc

    edges = np.concatenate(
        (
            plc.triangles[:, [0, 1]],
            plc.triangles[:, [1, 2]],
            plc.triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(plc.vertices)).astype(np.float64)
    movable = ~np.asarray(plc.fixed_vertices, dtype=bool)
    movable &= degree > 0
    reference = plc.vertices.copy()
    current = reference.copy()
    reference_triangles = reference[plc.triangles]
    reference_normals = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    reference_area2 = np.linalg.norm(reference_normals, axis=1)
    area_floor = max(
        np.finfo(float).eps * 4096.0 * float(np.median(reference_area2[reference_area2 > 0.0]))
        if np.any(reference_area2 > 0.0)
        else np.finfo(float).tiny,
        np.finfo(float).tiny,
    )

    def laplacian(values: np.ndarray) -> np.ndarray:
        sums = np.zeros_like(values)
        np.add.at(sums, edges[:, 0], values[edges[:, 1]])
        np.add.at(sums, edges[:, 1], values[edges[:, 0]])
        averages = np.divide(
            sums,
            degree[:, None],
            out=values.copy(),
            where=degree[:, None] > 0.0,
        )
        return averages - values

    def admissible(candidate: np.ndarray) -> bool:
        xyz = candidate[plc.triangles]
        normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        area2 = np.linalg.norm(normals, axis=1)
        if np.any(area2 <= area_floor):
            return False
        orientation = np.einsum("ij,ij->i", normals, reference_normals)
        return bool(np.all(orientation > 0.0))

    for _ in range(iterations):
        for coefficient in (relaxation, taubin_mu):
            delta = coefficient * laplacian(current)
            delta[~movable] = 0.0
            scale = 1.0
            accepted = False
            for _line in range(line_search_steps):
                candidate = current + scale * delta
                if maximum_displacement_m is not None:
                    displacement = candidate - reference
                    length = np.linalg.norm(displacement, axis=1)
                    excess = length > maximum_displacement_m
                    if np.any(excess):
                        displacement[excess] *= (
                            maximum_displacement_m / length[excess]
                        )[:, None]
                        candidate = reference + displacement
                candidate[~movable] = reference[~movable]
                if admissible(candidate):
                    current = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                continue

    result = MultiphasePLC(
        vertices=current,
        triangles=plc.triangles,
        facet_markers=plc.facet_markers,
        left_regions=plc.left_regions,
        right_regions=plc.right_regions,
        outside_region=plc.outside_region,
        marker_names=plc.marker_names,
        marker_region_pairs=plc.marker_region_pairs,
        fixed_vertices=plc.fixed_vertices,
        metadata={
            **plc.metadata,
            "smoothing": {
                "method": "taubin-interface-graph",
                "iterations": int(iterations),
                "relaxation": float(relaxation),
                "taubin_mu": float(taubin_mu),
                "maximum_displacement_m": maximum_displacement_m,
                "maximum_observed_displacement_m": float(
                    np.max(np.linalg.norm(current - reference, axis=1))
                ),
            },
        },
    )
    audit = audit_multiphase_plc(result)
    if not audit.valid:
        raise GeometryError(
            "surface smoothing violated the PLC manifold gate: "
            f"{audit}"
        )
    return result


def count_nonmanifold_voxel_edges(labels: np.ndarray) -> int:
    """Count diagonal material contacts around physical voxel-grid edges."""

    return len(_ambiguous_edge_events(np.ascontiguousarray(labels, dtype=np.int32)))


def regularize_nonmanifold_junctions(
    volume: MicrostructureVolume,
    *,
    maximum_changed_fraction: float = 0.005,
    maximum_iterations: int = 10_000,
    preserve_outer_layer: bool = False,
    minimum_phase_voxels: int = 8,
    phase_change_penalties: Mapping[int, float] | None = None,
    adaptive_budget: bool = True,
    hard_maximum_changed_fraction: float = 0.05,
    budget_growth_factor: float = 2.0,
    strict: bool = True,
) -> tuple[MicrostructureVolume, JunctionRegularizationReport]:
    """Remove edge-only diagonal contacts with deterministic minimal relabeling.

    Complex stochastic microstructures can contain hundreds or thousands of
    checkerboard edge contacts.  ``maximum_changed_fraction`` is therefore the
    *initial* conservative budget.  With ``adaptive_budget=True`` (the default),
    the budget grows only when it is actually exhausted, up to
    ``hard_maximum_changed_fraction``.  This preserves the old conservative
    behaviour for easy geometries while allowing difficult MCS-style particle /
    CBD / electrolyte volumes to become a valid manifold PLC without forcing
    every caller to guess a suitable budget.

    The repair always changes a voxel to an already adjacent material, honours
    per-phase penalties and minimum phase populations, records every edit, and
    never silently accepts a non-manifold result when ``strict=True``.
    """

    if not 0.0 <= maximum_changed_fraction <= 1.0:
        raise ValueError("maximum_changed_fraction must lie in [0, 1]")
    if not 0.0 <= hard_maximum_changed_fraction <= 1.0:
        raise ValueError("hard_maximum_changed_fraction must lie in [0, 1]")
    if hard_maximum_changed_fraction < maximum_changed_fraction:
        raise ValueError(
            "hard_maximum_changed_fraction cannot be smaller than "
            "maximum_changed_fraction"
        )
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if minimum_phase_voxels < 1:
        raise ValueError("minimum_phase_voxels must be positive")
    if adaptive_budget and (not np.isfinite(budget_growth_factor) or budget_growth_factor <= 1.0):
        raise ValueError("budget_growth_factor must be finite and greater than one")

    original = np.ascontiguousarray(volume.labels, dtype=np.int32)
    labels = original.copy()
    values, counts = np.unique(labels, return_counts=True)
    phase_counts = {
        int(value): int(count)
        for value, count in zip(values, counts, strict=True)
    }
    before_counts = dict(phase_counts)
    penalties = {
        int(key): float(value)
        for key, value in (phase_change_penalties or {}).items()
    }
    before_events = _ambiguous_edge_events(labels)
    before = len(before_events)

    initial_maximum_changes = int(np.floor(maximum_changed_fraction * labels.size))
    hard_maximum_changes = int(np.floor(hard_maximum_changed_fraction * labels.size))
    if before and initial_maximum_changes < 1:
        initial_maximum_changes = 1
    if before and hard_maximum_changes < 1:
        hard_maximum_changes = 1
    hard_maximum_changes = max(initial_maximum_changes, hard_maximum_changes)
    current_maximum_changes = initial_maximum_changes

    changes: list[tuple[int, int, int, int, int]] = []
    operations = 0
    budget_expansions = 0
    termination_reason = "not-started"

    changed_coordinates: set[tuple[int, int, int]] = set()
    previous_ambiguity = before
    while operations < maximum_iterations:
        events = _ambiguous_edge_events(labels)
        if not events:
            termination_reason = "converged"
            break

        if len(changed_coordinates) >= current_maximum_changes:
            if adaptive_budget and current_maximum_changes < hard_maximum_changes:
                # Grow conservatively, but use the current ambiguity count to avoid
                # dozens of tiny expansions for highly disordered multiphase data.
                ambiguity_allowance = int(np.ceil(0.35 * len(events)))
                grown = max(
                    current_maximum_changes + 1,
                    int(np.ceil(current_maximum_changes * budget_growth_factor)),
                    len(changed_coordinates) + ambiguity_allowance,
                )
                current_maximum_changes = min(hard_maximum_changes, grown)
                budget_expansions += 1
            else:
                termination_reason = "change-budget-exhausted"
                break

        # Aggregate every legal repair proposal. A proposal may resolve several
        # checkerboard edges at once, so event support is included in its score.
        proposals: dict[tuple[tuple[int, int, int], int], list[float]] = {}
        for event in events:
            for coordinate, target in _junction_candidates(labels, event):
                if coordinate in changed_coordinates:
                    continue
                z, y, x = coordinate
                source = int(labels[z, y, x])
                target = int(target)
                if source == target:
                    continue
                if preserve_outer_layer and (
                    z in {0, labels.shape[0] - 1}
                    or y in {0, labels.shape[1] - 1}
                    or x in {0, labels.shape[2] - 1}
                ):
                    continue
                if phase_counts.get(source, 0) <= minimum_phase_voxels:
                    continue
                neighbors = _neighbor_values(labels, coordinate)
                support_target = int(np.count_nonzero(neighbors == target))
                support_source = int(np.count_nonzero(neighbors == source))

                # Dynamic balance term discourages systematic material-fraction
                # drift while still allowing the local topology to converge.
                source_delta = phase_counts.get(source, 0) - before_counts.get(source, 0)
                target_delta = phase_counts.get(target, 0) - before_counts.get(target, 0)
                balance = 0.35 * (target_delta - source_delta) / max(1, labels.size)
                score = (
                    4.0 * support_target
                    - 2.0 * support_source
                    - penalties.get(source, 0.0)
                    - 0.25 * penalties.get(target, 0.0)
                    - balance
                )
                proposals.setdefault((coordinate, target), []).append(score)

        ranked: list[
            tuple[int, float, tuple[int, int, int], int, int]
        ] = []
        plateau: list[
            tuple[int, float, tuple[int, int, int], int, int]
        ] = []
        for (coordinate, target), scores in proposals.items():
            source = int(labels[coordinate])
            local_before, local_after = _local_ambiguity_before_after(
                labels, coordinate, target
            )
            gain = local_before - local_after
            item = (
                gain,
                float(max(scores) + 0.5 * len(scores)),
                coordinate,
                source,
                target,
            )
            if gain > 0:
                ranked.append(item)
            elif gain == 0:
                plateau.append(item)
        ranked.sort(
            key=lambda item: (-item[0], -item[1], item[2], item[3], item[4])
        )
        plateau.sort(key=lambda item: (-item[1], item[2], item[3], item[4]))
        use_plateau_escape = not ranked and bool(plateau)
        if not ranked and not plateau:
            termination_reason = "no-legal-repair-proposal"
            break
        if use_plateau_escape:
            # Some multi-label junctions require one topology-neutral move before
            # a strictly improving edit exists. Apply exactly one deterministic
            # zero-gain move and never revisit that voxel.
            ranked = [plateau[0]]

        # Select an independent batch. Chebyshev radius two prevents the local
        # 3x3x3 ambiguity neighborhoods of two edits from overlapping.
        selected: list[tuple[int, float, tuple[int, int, int], int, int]] = []
        blocked: set[tuple[int, int, int]] = set()
        remaining_budget = current_maximum_changes - len(changed_coordinates)
        remaining_operations = maximum_iterations - operations
        batch_limit = min(
            1 if use_plateau_escape else 512,
            remaining_budget,
            remaining_operations,
        )
        for item in ranked:
            coordinate = item[2]
            if coordinate in blocked:
                continue
            selected.append(item)
            z, y, x = coordinate
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        blocked.add((z + dz, y + dy, x + dx))
            if len(selected) >= batch_limit:
                break
        if not selected:
            termination_reason = "no-independent-repair-batch"
            break

        snapshot: list[tuple[tuple[int, int, int], int, int]] = []
        for _gain, _score, coordinate, source, target in selected:
            if int(labels[coordinate]) != source:
                continue
            snapshot.append((coordinate, source, target))
            labels[coordinate] = target
            phase_counts[source] -= 1
            phase_counts[target] = phase_counts.get(target, 0) + 1

        current_ambiguity = count_nonmanifold_voxel_edges(labels)
        if current_ambiguity > previous_ambiguity:
            for coordinate, source, target in reversed(snapshot):
                labels[coordinate] = source
                phase_counts[source] += 1
                phase_counts[target] -= 1
            termination_reason = "repair-batch-increased-ambiguity"
            break

        for coordinate, source, target in snapshot:
            z, y, x = coordinate
            changed_coordinates.add(coordinate)
            changes.append((z, y, x, source, target))
        operations += len(snapshot)
        previous_ambiguity = current_ambiguity
    else:
        termination_reason = "maximum-iterations-exhausted"

    after = count_nonmanifold_voxel_edges(labels)
    changed_mask = labels != original
    changed_voxels = int(np.count_nonzero(changed_mask))
    converged = after == 0
    if converged:
        termination_reason = "converged"

    after_counts = {
        int(value): int(count)
        for value, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    all_phases = set(before_counts) | set(after_counts)
    maximum_phase_fraction_drift = max(
        (
            abs(after_counts.get(phase, 0) - before_counts.get(phase, 0))
            / labels.size
            for phase in all_phases
        ),
        default=0.0,
    )
    report = JunctionRegularizationReport(
        ambiguous_edges_before=before,
        ambiguous_edges_after=after,
        changed_voxels=changed_voxels,
        changed_fraction=float(changed_voxels / labels.size),
        iterations=operations,
        converged=converged,
        phase_counts_before=before_counts,
        phase_counts_after=after_counts,
        changes=tuple(changes),
        initial_change_budget_fraction=float(initial_maximum_changes / labels.size),
        final_change_budget_fraction=float(current_maximum_changes / labels.size),
        hard_change_budget_fraction=float(hard_maximum_changes / labels.size),
        budget_expansions=budget_expansions,
        termination_reason=termination_reason,
        maximum_phase_fraction_drift=float(maximum_phase_fraction_drift),
    )
    if strict and not converged:
        raise GeometryError(
            "could not regularize non-manifold voxel junctions: "
            f"before={before}, after={after}, changed={changed_voxels}/"
            f"{labels.size} ({report.changed_fraction:.3%}), "
            f"initial_budget={report.initial_change_budget_fraction:.3%}, "
            f"final_budget={report.final_change_budget_fraction:.3%}, "
            f"hard_budget={report.hard_change_budget_fraction:.3%}, "
            f"expansions={budget_expansions}, reason={termination_reason}. "
            "For deliberately topology-dense inputs, increase "
            "junction_hard_maximum_changed_fraction or protect critical phases "
            "with junction_phase_change_penalties."
        )
    regularized = MicrostructureVolume(
        labels=labels,
        voxel_size_m=volume.voxel_size_m,
        origin_m=volume.origin_m,
        phase_names=volume.phase_names,
        metadata={
            **volume.metadata,
            "junction_regularization": {
                "ambiguous_edges_before": before,
                "ambiguous_edges_after": after,
                "changed_voxels": changed_voxels,
                "changed_fraction": report.changed_fraction,
                "converged": converged,
                "initial_change_budget_fraction": report.initial_change_budget_fraction,
                "final_change_budget_fraction": report.final_change_budget_fraction,
                "hard_change_budget_fraction": report.hard_change_budget_fraction,
                "budget_expansions": budget_expansions,
                "termination_reason": termination_reason,
                "maximum_phase_fraction_drift": report.maximum_phase_fraction_drift,
            },
        },
    )
    return regularized, report



def _local_ambiguity_before_after(
    labels: np.ndarray,
    coordinate: tuple[int, int, int],
    target: int,
) -> tuple[int, int]:
    """Return ambiguity counts in the complete local influence neighborhood."""

    z, y, x = coordinate
    slices = (
        slice(max(0, z - 1), min(labels.shape[0], z + 2)),
        slice(max(0, y - 1), min(labels.shape[1], y + 2)),
        slice(max(0, x - 1), min(labels.shape[2], x + 2)),
    )
    patch = np.ascontiguousarray(labels[slices])
    before = count_nonmanifold_voxel_edges(patch)
    local = (z - slices[0].start, y - slices[1].start, x - slices[2].start)
    original = int(patch[local])
    patch[local] = int(target)
    after = count_nonmanifold_voxel_edges(patch)
    patch[local] = original
    return before, after

def _ambiguous_edge_events(labels: np.ndarray) -> list[tuple[int, int, int, int]]:
    events: list[tuple[int, int, int, int]] = []

    def collect(axis: int, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> None:
        mask = ((a == d) & (a != b) & (a != c)) | ((b == c) & (b != a) & (b != d))
        for z, y, x in np.argwhere(mask):
            events.append((axis, int(z), int(y), int(x)))

    # Axis 2: physical edge parallel to x, 2x2 stencil in z-y.
    collect(
        2,
        labels[:-1, :-1, :],
        labels[1:, :-1, :],
        labels[:-1, 1:, :],
        labels[1:, 1:, :],
    )
    # Axis 1: physical edge parallel to y, 2x2 stencil in z-x.
    collect(
        1,
        labels[:-1, :, :-1],
        labels[1:, :, :-1],
        labels[:-1, :, 1:],
        labels[1:, :, 1:],
    )
    # Axis 0: physical edge parallel to z, 2x2 stencil in y-x.
    collect(
        0,
        labels[:, :-1, :-1],
        labels[:, 1:, :-1],
        labels[:, :-1, 1:],
        labels[:, 1:, 1:],
    )
    events.sort()
    return events


def _junction_candidates(
    labels: np.ndarray,
    event: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int, int], int]]:
    axis, z, y, x = event
    if axis == 2:
        coordinates = ((z, y, x), (z + 1, y, x), (z, y + 1, x), (z + 1, y + 1, x))
    elif axis == 1:
        coordinates = ((z, y, x), (z + 1, y, x), (z, y, x + 1), (z + 1, y, x + 1))
    else:
        coordinates = ((z, y, x), (z, y + 1, x), (z, y, x + 1), (z, y + 1, x + 1))
    values = [int(labels[coordinate]) for coordinate in coordinates]
    candidates: set[tuple[tuple[int, int, int], int]] = set()
    if values[0] == values[3]:
        candidates.add((coordinates[1], values[0]))
        candidates.add((coordinates[2], values[0]))
        candidates.add((coordinates[0], values[1]))
        candidates.add((coordinates[3], values[2]))
    if values[1] == values[2]:
        candidates.add((coordinates[0], values[1]))
        candidates.add((coordinates[3], values[1]))
        candidates.add((coordinates[1], values[0]))
        candidates.add((coordinates[2], values[3]))
    return sorted(candidates)


def _neighbor_values(labels: np.ndarray, coordinate: tuple[int, int, int]) -> np.ndarray:
    z, y, x = coordinate
    values: list[int] = []
    for dz, dy, dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
        zz, yy, xx = z + dz, y + dy, x + dx
        if 0 <= zz < labels.shape[0] and 0 <= yy < labels.shape[1] and 0 <= xx < labels.shape[2]:
            values.append(int(labels[zz, yy, xx]))
    return np.asarray(values, dtype=np.int32)


def _phase_name(volume: MicrostructureVolume, phase: int) -> str:
    value = str(volume.phase_names.get(int(phase), f"phase_{int(phase)}"))
    safe = "".join(character if character.isalnum() or character in "_-" else "_" for character in value)
    return safe or f"phase_{int(phase)}"


__all__ = [
    "JunctionRegularizationReport",
    "MultiphasePLC",
    "SurfacePLCAudit",
    "audit_multiphase_plc",
    "count_nonmanifold_voxel_edges",
    "extract_multiphase_plc",
    "lock_plc_interfaces",
    "regularize_nonmanifold_junctions",
    "smooth_multiphase_plc",
]
