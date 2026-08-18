"""Free-form, multi-domain, fully unstructured TetGen meshing.

Unlike the voxel compatibility path, this module never assumes a rectangular
bounding box or a six-tetrahedra-per-voxel topology.  Geometry is described by
closed triangular interface shells.  Each shell separates an ``inside_region``
from an ``outside_region``; an outer shell uses ``outside_region=None``.
Arbitrary watertight STL/OBJ/PLY/NPZ surfaces can therefore become the outer
boundary or internal material interfaces of one conforming TetGen PLC.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..core.exceptions import GeometryError
from ..geometry import (
    TetraQuality,
    TriangleMesh,
    VolumeMesh,
    extract_material_surfaces,
    load_triangle_mesh,
    tetra_quality,
    tetrahedron_signed_volumes,
)
from .surface import MultiphasePLC, SurfacePLCAudit, audit_multiphase_plc
from .region_recovery import (
    TetGenRegionRecoveryReport,
    recover_tetgen_material_regions,
)
from .tetgen import (
    LocalRefinementZone,
    TetGenMeshingConfig,
    _load_tetgen_native,
    _normalize_tetgen_input,
    _resolve_facet_constraints,
    tetgen_native_status,
)


@dataclass(frozen=True, slots=True)
class SurfaceShell:
    """One closed triangular interface separating two material regions.

    Surface normals are normalized to point from ``inside_region`` toward
    ``outside_region``.  For the exterior shell set ``outside_region=None``.
    """

    surface: TriangleMesh
    inside_region: int
    outside_region: int | None = None
    name: str = "interface"
    marker: int | None = None
    maximum_triangle_area_m2: float | None = None

    def __post_init__(self) -> None:
        if self.outside_region is not None and int(self.outside_region) == int(self.inside_region):
            raise ValueError("inside_region and outside_region must differ")
        if self.marker is not None and int(self.marker) <= 0:
            raise ValueError("marker must be positive when supplied")
        if self.maximum_triangle_area_m2 is not None and (
            not np.isfinite(self.maximum_triangle_area_m2)
            or self.maximum_triangle_area_m2 <= 0.0
        ):
            raise ValueError("maximum_triangle_area_m2 must be positive and finite")
        object.__setattr__(self, "inside_region", int(self.inside_region))
        object.__setattr__(
            self,
            "outside_region",
            None if self.outside_region is None else int(self.outside_region),
        )
        object.__setattr__(self, "name", str(self.name))
        if self.marker is not None:
            object.__setattr__(self, "marker", int(self.marker))


@dataclass(frozen=True, slots=True)
class FreeformRegion:
    """TetGen region seed and optional local volume constraint."""

    region: int
    seed_m_xyz: tuple[float, float, float]
    name: str = "domain"
    maximum_tetra_volume_m3: float | None = None

    def __post_init__(self) -> None:
        seed = tuple(float(value) for value in self.seed_m_xyz)
        if len(seed) != 3 or not np.all(np.isfinite(seed)):
            raise ValueError("seed_m_xyz must contain three finite values")
        maximum = self.maximum_tetra_volume_m3
        if maximum is not None and (not np.isfinite(maximum) or maximum <= 0.0):
            raise ValueError("maximum_tetra_volume_m3 must be positive and finite")
        object.__setattr__(self, "region", int(self.region))
        object.__setattr__(self, "seed_m_xyz", seed)
        object.__setattr__(self, "name", str(self.name))
        if maximum is not None:
            object.__setattr__(self, "maximum_tetra_volume_m3", float(maximum))


@dataclass(frozen=True, slots=True)
class FreeformTetMeshResult:
    mesh: VolumeMesh
    boundary: TriangleMesh
    interface_faces: Mapping[tuple[int, int], np.ndarray]
    quality: TetraQuality
    plc: MultiphasePLC
    plc_audit: SurfacePLCAudit
    output_surface: TriangleMesh | None
    regions: tuple[FreeformRegion, ...]
    switches: str
    native_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShellClearancePair:
    left_name: str
    right_name: str
    minimum_sample_distance_m: float
    reference_edge_length_m: float
    clearance_ratio: float
    required_clearance_m: float
    valid: bool


@dataclass(frozen=True, slots=True)
class FreeformClearanceAudit:
    pair_results: tuple[ShellClearancePair, ...]
    minimum_sample_distance_m: float
    minimum_clearance_ratio: float
    required_clearance_factor: float
    valid: bool


def _surface_sample_points(
    surface: TriangleMesh,
    *,
    maximum_samples: int = 60_000,
) -> np.ndarray:
    """Return deterministic geometry samples for inter-shell clearance checks.

    Vertices alone can miss an edge passing close to the middle of another
    triangle.  We therefore sample triangle centroids and all three edge
    midpoints.  Large implicit surfaces can contain tens of thousands of
    triangles, so the face-derived samples are deterministically thinned to
    keep preflight cost bounded without dropping the complete vertex set.
    """

    if maximum_samples < surface.n_vertices:
        # Keep a deterministic subset of vertices if the caller gives an
        # unusually small cap; default settings always retain all vertices.
        vertex_ids = np.linspace(
            0, surface.n_vertices - 1, maximum_samples, dtype=np.int64
        )
        return np.ascontiguousarray(surface.vertices[vertex_ids], dtype=np.float64)

    vertices = np.ascontiguousarray(surface.vertices, dtype=np.float64)
    tri = surface.vertices[surface.faces]
    remaining = maximum_samples - len(vertices)
    if remaining <= 0 or not len(tri):
        return vertices

    # Every selected face contributes centroid + three edge midpoints.
    face_budget = max(1, remaining // 4)
    if len(tri) > face_budget:
        face_ids = np.linspace(0, len(tri) - 1, face_budget, dtype=np.int64)
        tri = tri[face_ids]

    derived = np.concatenate(
        (
            tri.mean(axis=1),
            0.5 * (tri[:, 0] + tri[:, 1]),
            0.5 * (tri[:, 1] + tri[:, 2]),
            0.5 * (tri[:, 2] + tri[:, 0]),
        ),
        axis=0,
    )
    return np.ascontiguousarray(
        np.concatenate((vertices, derived), axis=0)[:maximum_samples],
        dtype=np.float64,
    )


def _median_surface_edge_length(surface: TriangleMesh) -> float:
    tri = surface.vertices[surface.faces]
    edges = np.concatenate(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        )
    )
    edges = edges[np.isfinite(edges) & (edges > 0.0)]
    if not len(edges):
        raise GeometryError("cannot estimate edge length for an empty/degenerate shell")
    return float(np.median(edges))


def _ensure_oriented_closed_surface(surface: TriangleMesh) -> TriangleMesh:
    if bool(surface.metadata.get("oriented_closed_surface", False)):
        return surface
    return orient_closed_surface(surface)


def audit_freeform_shell_clearance(
    shells: Sequence[SurfaceShell],
    *,
    minimum_clearance_factor: float = 0.35,
    minimum_clearance_m: float | None = None,
) -> FreeformClearanceAudit:
    """Audit separation between independently supplied closed PLC shells.

    TetGen's facet recovery is sensitive to nearly overlapping facets.  For two
    independent closed shells, a gap substantially smaller than their surface
    discretization scale is usually a modelling error: touching materials must
    be represented by one conforming shared interface instead of two almost
    coincident shells.

    The distance estimate is conservative and dependency-light.  It samples
    vertices, edge midpoints and triangle centroids and uses a KD-tree.  It is
    not a replacement for TetGen's exact PLC self-intersection detection, but
    catches the thin-gap failure mode that otherwise often appears as an
    internal ``split_subface`` error.
    """

    factor = float(minimum_clearance_factor)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("minimum_clearance_factor must be finite and non-negative")
    if minimum_clearance_m is not None and (
        not np.isfinite(minimum_clearance_m) or minimum_clearance_m < 0.0
    ):
        raise ValueError("minimum_clearance_m must be finite and non-negative")

    prepared = [
        (shell, _ensure_oriented_closed_surface(shell.surface))
        for shell in tuple(shells)
    ]
    samples = [_surface_sample_points(surface) for _, surface in prepared]
    medians = [_median_surface_edge_length(surface) for _, surface in prepared]

    results: list[ShellClearancePair] = []
    for left_index in range(len(prepared)):
        for right_index in range(left_index + 1, len(prepared)):
            left_shell, _ = prepared[left_index]
            right_shell, _ = prepared[right_index]
            left_points = samples[left_index]
            right_points = samples[right_index]

            # Query the smaller cloud against the larger one.
            if len(left_points) <= len(right_points):
                distances, _ = cKDTree(right_points).query(left_points, k=1)
            else:
                distances, _ = cKDTree(left_points).query(right_points, k=1)
            minimum = float(np.min(distances)) if len(distances) else float("inf")
            reference = float(min(medians[left_index], medians[right_index]))
            required = max(
                0.0 if minimum_clearance_m is None else float(minimum_clearance_m),
                factor * reference,
            )
            ratio = minimum / reference if reference > 0.0 else float("inf")
            results.append(
                ShellClearancePair(
                    left_name=left_shell.name,
                    right_name=right_shell.name,
                    minimum_sample_distance_m=minimum,
                    reference_edge_length_m=reference,
                    clearance_ratio=ratio,
                    required_clearance_m=required,
                    valid=minimum + np.finfo(float).eps >= required,
                )
            )

    if results:
        minimum_distance = min(item.minimum_sample_distance_m for item in results)
        minimum_ratio = min(item.clearance_ratio for item in results)
        valid = all(item.valid for item in results)
    else:
        minimum_distance = float("inf")
        minimum_ratio = float("inf")
        valid = True
    return FreeformClearanceAudit(
        pair_results=tuple(results),
        minimum_sample_distance_m=minimum_distance,
        minimum_clearance_ratio=minimum_ratio,
        required_clearance_factor=factor,
        valid=valid,
    )


def load_surface_shell(
    path: str | Path,
    *,
    inside_region: int,
    outside_region: int | None = None,
    name: str | None = None,
    marker: int | None = None,
    coordinate_scale: float = 1.0,
    maximum_triangle_area_m2: float | None = None,
) -> SurfaceShell:
    """Load OBJ/PLY/STL/NPZ/etc. and turn it into a free-form interface shell."""

    mesh = load_triangle_mesh(path)
    scale = float(coordinate_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale must be positive and finite")
    if scale != 1.0:
        mesh = TriangleMesh(
            vertices=mesh.vertices * scale,
            faces=mesh.faces,
            face_materials=mesh.face_materials,
            metadata={**mesh.metadata, "coordinate_scale": scale},
        )
    return SurfaceShell(
        surface=mesh,
        inside_region=inside_region,
        outside_region=outside_region,
        name=Path(path).stem if name is None else name,
        marker=marker,
        maximum_triangle_area_m2=maximum_triangle_area_m2,
    )


def orient_closed_surface(surface: TriangleMesh) -> TriangleMesh:
    """Return a consistently oriented, outward-facing copy of a closed surface.

    The routine is dependency-light and rejects open/non-manifold shells instead
    of silently filling them.  Adjacent triangles are propagated so their shared
    edge directions oppose one another.  The final signed enclosed volume then
    selects the outward global orientation.
    """

    faces = np.ascontiguousarray(surface.faces, dtype=np.int64).copy()
    vertices = surface.vertices
    if not len(faces):
        raise GeometryError("free-form shell contains no triangles")

    tri = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    extent = float(np.max(np.ptp(vertices, axis=0)))
    tolerance = max(np.finfo(float).eps * max(extent**2, np.finfo(float).tiny) * 128.0, np.finfo(float).tiny)
    if np.any(double_area <= tolerance):
        raise GeometryError(
            f"free-form shell contains {int(np.count_nonzero(double_area <= tolerance))} degenerate triangles"
        )

    edge_refs: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            sign = 1 if (u, v) == key else -1
            edge_refs[key].append((face_index, sign))

    open_edges = sum(len(refs) == 1 for refs in edge_refs.values())
    nonmanifold = sum(len(refs) != 2 for refs in edge_refs.values())
    if open_edges or nonmanifold:
        raise GeometryError(
            "free-form shell must be closed and 2-manifold: "
            f"open_edges={open_edges}, nonmanifold_edges={nonmanifold}"
        )

    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for refs in edge_refs.values():
        (fa, sa), (fb, sb) = refs
        # orientation factor x in {+1,-1}; require sa*xa == -(sb*xb)
        relation = -sa * sb
        neighbors[fa].append((fb, relation))
        neighbors[fb].append((fa, relation))

    state = np.zeros(len(faces), dtype=np.int8)
    component_faces: list[list[int]] = []
    for root in range(len(faces)):
        if state[root]:
            continue
        current_component: list[int] = []
        state[root] = 1
        queue: deque[int] = deque([root])
        while queue:
            current = queue.popleft()
            current_component.append(current)
            for other, relation in neighbors[current]:
                wanted = int(state[current]) * int(relation)
                if state[other] == 0:
                    state[other] = wanted
                    queue.append(other)
                elif int(state[other]) != wanted:
                    raise GeometryError("free-form shell is not consistently orientable")
        component_faces.append(current_component)

    flip = state < 0
    if np.any(flip):
        faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()

    component_volumes: list[float] = []
    volume_tolerance = max(extent**3, np.finfo(float).tiny) * np.finfo(float).eps * 256.0
    # Connected components must each be outward, not merely the sum of all
    # components.  Marching-cubes fields commonly contain many disconnected
    # particles/CBD bodies.
    for indices in component_faces:
        index_array = np.asarray(indices, dtype=np.int64)
        tri_component = vertices[faces[index_array]]
        component_volume = float(
            np.einsum(
                "ij,ij->i",
                tri_component[:, 0],
                np.cross(tri_component[:, 1], tri_component[:, 2]),
            ).sum()
            / 6.0
        )
        if abs(component_volume) <= volume_tolerance:
            raise GeometryError(
                "free-form shell contains a connected component with zero or "
                "numerically unresolved enclosed volume"
            )
        if component_volume < 0.0:
            faces[index_array, 1], faces[index_array, 2] = (
                faces[index_array, 2].copy(),
                faces[index_array, 1].copy(),
            )
            component_volume = -component_volume
        component_volumes.append(component_volume)
    signed_volume = float(sum(component_volumes))
    components = len(component_faces)

    return TriangleMesh(
        vertices=vertices,
        faces=faces,
        metadata={
            **surface.metadata,
            "oriented_closed_surface": True,
            "connected_surface_components": components,
            "signed_enclosed_volume_m3": signed_volume,
        },
    )


def _weld_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld points within ``tolerance`` using KD-tree + union-find.

    The previous pure-Python 27-neighbour spatial hash became a bottleneck for
    20k--100k-vertex free-form PLCs.  ``cKDTree.query_pairs`` performs the broad
    phase in compiled code while union-find preserves transitive equivalence.
    """

    xyz = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(triangles, dtype=np.int64)
    if tolerance <= 0.0 or not len(xyz):
        return xyz, faces

    pairs = cKDTree(xyz).query_pairs(float(tolerance), output_type="ndarray")
    if pairs.size == 0:
        return xyz, faces

    parent = np.arange(len(xyz), dtype=np.int64)
    rank = np.zeros(len(xyz), dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def union(left: int, right: int) -> None:
        a = find(left)
        b = find(right)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    for left, right in np.asarray(pairs, dtype=np.int64):
        union(int(left), int(right))

    roots = np.fromiter((find(i) for i in range(len(xyz))), dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    welded_vertices = xyz[unique_roots]
    welded_triangles = inverse[faces]

    # Welding can collapse a triangle if the input contains a feature below the
    # tolerance.  Reject it here instead of letting TetGen fail during facet
    # recovery.
    degenerate = (
        (welded_triangles[:, 0] == welded_triangles[:, 1])
        | (welded_triangles[:, 1] == welded_triangles[:, 2])
        | (welded_triangles[:, 2] == welded_triangles[:, 0])
    )
    if np.any(degenerate):
        raise GeometryError(
            "vertex welding collapsed "
            f"{int(np.count_nonzero(degenerate))} PLC triangles; reduce weld_tolerance_m"
        )

    return (
        np.ascontiguousarray(welded_vertices, dtype=np.float64),
        np.ascontiguousarray(welded_triangles, dtype=np.int64),
    )


def assemble_freeform_plc(
    shells: Sequence[SurfaceShell],
    *,
    outside_region: int = -1,
    weld_tolerance_m: float | None = None,
    strict: bool = True,
    minimum_clearance_factor: float = 0.0,
    minimum_clearance_m: float | None = None,
    strict_clearance: bool = False,
) -> MultiphasePLC:
    """Assemble arbitrary closed shells into one multi-domain TetGen PLC.

    Examples include a curved outer casing containing pores/inclusions, multiple
    particles inside an electrolyte domain, hollow objects, and nested layers.
    Shells must not self-intersect; a material interface should be supplied only
    once, with its two adjacent region IDs.
    """

    if not shells:
        raise ValueError("at least one free-form shell is required")
    shells = tuple(shells)
    clearance_audit = audit_freeform_shell_clearance(
        shells,
        minimum_clearance_factor=minimum_clearance_factor,
        minimum_clearance_m=minimum_clearance_m,
    )
    if strict_clearance and not clearance_audit.valid:
        failing = [item for item in clearance_audit.pair_results if not item.valid]
        details = "; ".join(
            f"{item.left_name}<->{item.right_name}: "
            f"gap={item.minimum_sample_distance_m:.6g} m, "
            f"required>={item.required_clearance_m:.6g} m"
            for item in failing[:8]
        )
        raise GeometryError(
            "free-form shells are too close for robust TetGen facet recovery; "
            "increase physical clearance or construct touching materials as one "
            f"conforming shared interface. {details}"
        )
    outside_region = int(outside_region)
    prepared: list[tuple[SurfaceShell, TriangleMesh, int]] = []
    used_markers: set[int] = set()
    next_marker = 1
    for shell in shells:
        marker = shell.marker
        if marker is None:
            while next_marker in used_markers:
                next_marker += 1
            marker = next_marker
            next_marker += 1
        marker = int(marker)
        if marker in used_markers:
            raise ValueError(f"duplicate free-form facet marker: {marker}")
        used_markers.add(marker)
        prepared.append((shell, _ensure_oriented_closed_surface(shell.surface), marker))

    all_vertices: list[np.ndarray] = []
    all_triangles: list[np.ndarray] = []
    all_markers: list[np.ndarray] = []
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    marker_names: dict[int, str] = {}
    marker_pairs: dict[int, tuple[int | None, int | None]] = {}
    facet_area_constraints: dict[int, float] = {}
    offset = 0
    for shell, surface, marker in prepared:
        nfaces = surface.n_faces
        outer = outside_region if shell.outside_region is None else shell.outside_region
        all_vertices.append(surface.vertices)
        all_triangles.append(surface.faces + offset)
        all_markers.append(np.full(nfaces, marker, dtype=np.int32))
        left.append(np.full(nfaces, shell.inside_region, dtype=np.int32))
        right.append(np.full(nfaces, outer, dtype=np.int32))
        marker_names[marker] = shell.name
        marker_pairs[marker] = (
            shell.inside_region,
            None if shell.outside_region is None else shell.outside_region,
        )
        if shell.maximum_triangle_area_m2 is not None:
            facet_area_constraints[marker] = float(shell.maximum_triangle_area_m2)
        offset += surface.n_vertices

    vertices = np.concatenate(all_vertices, axis=0)
    triangles = np.concatenate(all_triangles, axis=0)
    if weld_tolerance_m is None:
        extent = float(np.max(np.ptp(vertices, axis=0)))
        weld_tolerance_m = max(extent * 1.0e-12, 0.0)
    vertices, triangles = _weld_vertices(vertices, triangles, float(weld_tolerance_m))

    plc = MultiphasePLC(
        vertices=vertices,
        triangles=triangles,
        facet_markers=np.concatenate(all_markers),
        left_regions=np.concatenate(left),
        right_regions=np.concatenate(right),
        outside_region=outside_region,
        marker_names=marker_names,
        marker_region_pairs=marker_pairs,
        fixed_vertices=np.ones(len(vertices), dtype=bool),
        metadata={
            "source": "freeform-shells",
            "shell_count": len(shells),
            "facet_area_constraints_m2": facet_area_constraints,
            "weld_tolerance_m": float(weld_tolerance_m),
            "clearance_audit": {
                "valid": clearance_audit.valid,
                "minimum_sample_distance_m": clearance_audit.minimum_sample_distance_m,
                "minimum_clearance_ratio": clearance_audit.minimum_clearance_ratio,
                "required_clearance_factor": clearance_audit.required_clearance_factor,
                "pairs": [
                    {
                        "left": item.left_name,
                        "right": item.right_name,
                        "gap_m": item.minimum_sample_distance_m,
                        "reference_edge_m": item.reference_edge_length_m,
                        "ratio": item.clearance_ratio,
                        "required_gap_m": item.required_clearance_m,
                        "valid": item.valid,
                    }
                    for item in clearance_audit.pair_results
                ],
            },
        },
    )
    audit = audit_multiphase_plc(plc)
    if strict and not audit.valid:
        raise GeometryError(f"free-form PLC failed topology audit: {audit}")
    return plc


def _region_seed_array(
    regions: Sequence[FreeformRegion],
    config: TetGenMeshingConfig,
) -> np.ndarray:
    if not regions:
        raise ValueError("at least one FreeformRegion is required")
    rows = []
    for region in regions:
        maximum = region.maximum_tetra_volume_m3
        if maximum is None:
            maximum = config.phase_maximum_tetra_volume_m3.get(
                region.region, config.global_maximum_tetra_volume_m3
            )
        rows.append((*region.seed_m_xyz, float(region.region), -1.0 if maximum is None else maximum))
    return np.asarray(rows, dtype=np.float64)


def mesh_freeform_tetgen(
    plc_or_shells: MultiphasePLC | Sequence[SurfaceShell],
    regions: Sequence[FreeformRegion],
    *,
    config: TetGenMeshingConfig | None = None,
    holes_m_xyz: Sequence[tuple[float, float, float]] = (),
    void_regions: Sequence[int] = (),
    maximum_tetrahedra: int = 20_000_000,
    weld_tolerance_m: float | None = None,
) -> FreeformTetMeshResult:
    """Generate a quality, spatially graded Tet4 mesh for arbitrary 3D geometry."""

    resolved = config or TetGenMeshingConfig()
    if maximum_tetrahedra < 1:
        raise ValueError("maximum_tetrahedra must be positive")
    plc = (
        plc_or_shells
        if isinstance(plc_or_shells, MultiphasePLC)
        else assemble_freeform_plc(
            tuple(plc_or_shells),
            weld_tolerance_m=weld_tolerance_m,
            strict=True,
            minimum_clearance_factor=resolved.freeform_minimum_clearance_factor,
            minimum_clearance_m=resolved.freeform_minimum_clearance_m,
            strict_clearance=resolved.freeform_strict_clearance,
        )
    )
    audit = audit_multiphase_plc(plc)
    if not audit.valid:
        raise GeometryError(f"free-form TetGen input PLC is invalid: {audit}")

    configured_ids = {region.region for region in regions}
    void_ids = {int(value) for value in void_regions}
    if configured_ids & void_ids:
        raise GeometryError("a region cannot be both meshed and declared void")
    plc_ids = set(plc.regions)
    if configured_ids | void_ids != plc_ids:
        raise GeometryError(
            "free-form seeds plus void regions must cover exactly the PLC regions: "
            f"plc={sorted(plc_ids)}, seeds={sorted(configured_ids)}, void={sorted(void_ids)}"
        )
    if void_ids and len(holes_m_xyz) < len(void_ids):
        raise GeometryError(
            "void_regions require at least one TetGen hole point per void region"
        )

    status = tetgen_native_status()
    if not status.available:
        raise RuntimeError(
            "free-form TetGen meshing requires the compiled TetGen extension; "
            f"details: {status.reason}"
        )
    native = _load_tetgen_native()

    region_seeds = _region_seed_array(regions, resolved)
    configured_facet_limits = dict(resolved.facet_maximum_area_m2)
    for marker, value in dict(plc.metadata.get("facet_area_constraints_m2", {})).items():
        configured_facet_limits.setdefault(int(marker), float(value))
    facet_constraints = _resolve_facet_constraints(plc, configured_facet_limits)
    zones = np.asarray(
        [
            (*zone.center_m_xyz, zone.radius_m, zone.maximum_tetra_volume_m3)
            for zone in resolved.local_refinement_zones
        ],
        dtype=np.float64,
    ).reshape(-1, 5)
    holes = np.asarray(tuple(holes_m_xyz), dtype=np.float64).reshape(-1, 3)
    if holes.size and not np.all(np.isfinite(holes)):
        raise ValueError("holes_m_xyz contains non-finite coordinates")

    if resolved.normalize_coordinates:
        native_points, native_seeds, native_constraints, native_zones, transform = _normalize_tetgen_input(
            plc.vertices, region_seeds, facet_constraints, zones
        )
        native_holes = np.ascontiguousarray(holes.copy(), dtype=np.float64)
        if len(native_holes):
            native_holes[:] = (native_holes - transform[0]) / transform[1]
    else:
        native_points, native_seeds, native_constraints, native_zones = (
            plc.vertices,
            region_seeds,
            facet_constraints,
            zones,
        )
        native_holes = holes
        transform = None

    def _native_call(
        *,
        preserve_boundary_facets: bool,
        conforming_delaunay: bool,
        constraints: np.ndarray,
    ):
        try:
            return native.tetrahedralize(
                native_points,
                plc.triangles,
                plc.facet_markers,
                native_seeds,
                native_holes,
                constraints,
                native_zones,
                float(resolved.radius_edge_ratio),
                float(resolved.minimum_dihedral_degrees),
                int(resolved.optimization_level),
                int(resolved.maximum_steiner_points),
                bool(resolved.consistency_check),
                bool(conforming_delaunay),
                bool(resolved.quiet),
                bool(preserve_boundary_facets),
            )
        except TypeError as exc:
            if "incompatible function arguments" in str(exc) or "tetrahedralize" in str(exc):
                raise RuntimeError(
                    "the installed _zynmorph_tetgen_native extension uses an older "
                    "ABI and does not support robust boundary recovery. Rebuild ZynNova "
                    "with `python -m pip install -e .[zynmorph-all] -v`, then restart "
                    "the Jupyter kernel."
                ) from exc
            raise

    recovery_mode = "none"
    primary_error: str | None = None
    try:
        raw = _native_call(
            preserve_boundary_facets=resolved.preserve_boundary_facets,
            conforming_delaunay=resolved.conforming_delaunay,
            constraints=native_constraints,
        )
    except RuntimeError as exc:
        primary_error = str(exc)
        lowered = primary_error.lower()
        recoverable = any(
            token in lowered
            for token in (
                "split_subface",
                "very close input facets",
                "two very close",
                "internal tetgen error",
                "facet recovery",
            )
        )
        if not (
            resolved.retry_preserve_boundary_on_facet_error
            and recoverable
            and not resolved.preserve_boundary_facets
        ):
            raise GeometryError(
                "TetGen failed during free-form PLC recovery. The PLC is topologically "
                "manifold but may contain geometric self-intersections or sub-resolution "
                f"facet gaps. Native error: {primary_error}"
            ) from exc

        # TetGen's documented -Y recovery avoids insertion of Steiner points on
        # PLC boundary facets. Facet-area constraints are intentionally removed
        # in this retry because they request exactly such boundary splitting;
        # volumetric and local-zone sizing remain active.
        recovery_mode = "preserve-boundary-facets"
        try:
            raw = _native_call(
                preserve_boundary_facets=True,
                conforming_delaunay=False,
                constraints=np.empty((0, 2), dtype=np.float64),
            )
        except RuntimeError as retry_exc:
            clearance = plc.metadata.get("clearance_audit", {})
            raise GeometryError(
                "TetGen facet recovery failed in both normal and -Y boundary-preserving "
                "modes. This strongly indicates intersecting or under-resolved close "
                "facets. Rebuild the implicit geometry with a larger material clearance "
                "or a finer surface sampling grid. "
                f"primary={primary_error!r}; retry={str(retry_exc)!r}; "
                f"clearance_audit={clearance}"
            ) from retry_exc

    nodes = np.ascontiguousarray(raw["points"], dtype=np.float64)
    if transform is not None:
        nodes = np.ascontiguousarray(nodes * transform[1] + transform[0])
    tetrahedra = np.ascontiguousarray(raw["tetrahedra"], dtype=np.int64)
    if len(tetrahedra) > maximum_tetrahedra:
        raise GeometryError(
            f"TetGen produced {len(tetrahedra):,} tetrahedra, exceeding "
            f"maximum_tetrahedra={maximum_tetrahedra:,}"
        )
    attributes = np.asarray(raw["region_attributes"], dtype=np.float64)
    if attributes.shape != (len(tetrahedra),):
        raise GeometryError("TetGen did not return one region attribute per tetrahedron")
    rounded = np.rint(attributes)
    if not np.allclose(attributes, rounded, rtol=0.0, atol=1.0e-8):
        raise GeometryError("TetGen returned non-integral free-form region attributes")
    raw_cell_regions = np.ascontiguousarray(rounded, dtype=np.int32)

    raw_output_ids = set(map(int, np.unique(raw_cell_regions)))
    if raw_output_ids == configured_ids:
        cell_regions = raw_cell_regions
        raw_counts = {
            int(key): int(value)
            for key, value in zip(
                *np.unique(raw_cell_regions, return_counts=True),
                strict=True,
            )
        }
        region_recovery = TetGenRegionRecoveryReport(
            requested_material_ids=tuple(sorted(configured_ids)),
            raw_region_ids=tuple(sorted(raw_output_ids)),
            automatic_region_ids=(),
            raw_to_material={value: value for value in sorted(raw_output_ids)},
            raw_region_cell_counts=raw_counts,
            material_cell_counts=raw_counts,
            boundary_anchors=0,
            interface_constraints=0,
            recovered_automatic_regions=0,
            recovered_cells=0,
            unresolved_region_ids=(),
            valid=True,
        )
    else:
        trifaces_for_recovery = np.asarray(
            raw.get("trifaces", np.empty((0, 3))), dtype=np.int64
        )
        trimarkers_for_recovery = np.asarray(
            raw.get("triface_markers", np.empty(0)), dtype=np.int32
        )
        cell_regions, region_recovery = recover_tetgen_material_regions(
            tetrahedra,
            raw_cell_regions,
            trifaces_for_recovery,
            trimarkers_for_recovery,
            plc.marker_region_pairs,
            requested_material_ids=tuple(sorted(configured_ids)),
            outside_region=plc.outside_region,
            void_region_ids=tuple(sorted(void_ids)),
        )

    region_names: dict[int, str] = {}
    for region in regions:
        region_names.setdefault(region.region, region.name.split("_component_")[0])

    tentative = VolumeMesh(nodes, tetrahedra, cell_regions, region_names)
    signed = tetrahedron_signed_volumes(tentative)
    negative = signed < 0.0
    if np.any(negative):
        tetrahedra = tetrahedra.copy()
        tetrahedra[negative, 0], tetrahedra[negative, 1] = (
            tetrahedra[negative, 1].copy(),
            tetrahedra[negative, 0].copy(),
        )
    mesh = VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=cell_regions,
        region_names=region_names,
        metadata={
            "source": "tetgen-1.6.0-freeform-plc",
            "switches": str(raw["switches"]),
            "adaptive": True,
            "rectangular_domain_assumed": False,
            "tetgen_raw_region_ids": tuple(region_recovery.raw_region_ids),
            "tetgen_automatic_region_ids": tuple(region_recovery.automatic_region_ids),
            "tetgen_raw_to_material_region": dict(region_recovery.raw_to_material),
        },
    )
    quality = tetra_quality(mesh)
    if not quality.fem_ready:
        raise GeometryError(
            "free-form TetGen output failed FEM readiness: "
            f"inverted={quality.inverted_cells}, degenerate={quality.degenerate_cells}"
        )
    output_ids = set(map(int, np.unique(cell_regions)))
    if output_ids != configured_ids:
        raise GeometryError(
            "TetGen free-form material recovery mismatch: "
            f"requested={sorted(configured_ids)}, recovered={sorted(output_ids)}, "
            f"raw={sorted(raw_output_ids)}, mapping={dict(region_recovery.raw_to_material)}"
        )

    boundary, interfaces = extract_material_surfaces(mesh)
    trifaces = np.asarray(raw.get("trifaces", np.empty((0, 3))), dtype=np.int64)
    trimarkers = np.asarray(raw.get("triface_markers", np.empty(0)), dtype=np.int32)
    output_surface = None
    if trifaces.ndim == 2 and trifaces.shape[1:] == (3,) and len(trifaces):
        output_surface = TriangleMesh(
            vertices=nodes,
            faces=trifaces,
            face_materials=trimarkers if trimarkers.shape == (len(trifaces),) else None,
            metadata={"source": "tetgen-output-subfaces", "freeform": True},
        )

    return FreeformTetMeshResult(
        mesh=mesh,
        boundary=boundary,
        interface_faces=interfaces,
        quality=quality,
        plc=plc,
        plc_audit=audit,
        output_surface=output_surface,
        regions=tuple(regions),
        switches=str(raw["switches"]),
        native_version=str(raw.get("version", status.version)),
        metadata={
            "geometry_mode": "arbitrary-closed-shell-plc",
            "shell_count": int(plc.metadata.get("shell_count", 0)),
            "region_count": len(regions),
            "hole_count": len(holes),
            "void_regions": tuple(sorted(void_ids)),
            "local_refinement_zone_count": len(resolved.local_refinement_zones),
            "facet_constraint_count": len(facet_constraints),
            "tetgen_boundary_recovery_mode": recovery_mode,
            "tetgen_primary_error": primary_error,
            "tetgen_raw_region_ids": tuple(region_recovery.raw_region_ids),
            "tetgen_automatic_region_ids": tuple(region_recovery.automatic_region_ids),
            "tetgen_raw_to_material_region": dict(region_recovery.raw_to_material),
            "tetgen_region_recovery": {
                "requested_material_ids": tuple(region_recovery.requested_material_ids),
                "raw_region_ids": tuple(region_recovery.raw_region_ids),
                "automatic_region_ids": tuple(region_recovery.automatic_region_ids),
                "raw_to_material": dict(region_recovery.raw_to_material),
                "raw_region_cell_counts": dict(region_recovery.raw_region_cell_counts),
                "material_cell_counts": dict(region_recovery.material_cell_counts),
                "boundary_anchors": region_recovery.boundary_anchors,
                "interface_constraints": region_recovery.interface_constraints,
                "recovered_automatic_regions": region_recovery.recovered_automatic_regions,
                "recovered_cells": region_recovery.recovered_cells,
                "valid": region_recovery.valid,
            },
            "clearance_audit": plc.metadata.get("clearance_audit"),
            "coordinate_normalization": None
            if transform is None
            else {
                "offset_m_xyz": transform[0].tolist(),
                "scale_m": float(transform[1]),
            },
        },
    )


def mesh_closed_surface_tetgen(
    surface: TriangleMesh | str | Path,
    *,
    region: int = 1,
    region_name: str = "domain",
    seed_m_xyz: tuple[float, float, float],
    maximum_tetra_volume_m3: float | None = None,
    maximum_triangle_area_m2: float | None = None,
    config: TetGenMeshingConfig | None = None,
    coordinate_scale: float = 1.0,
    maximum_tetrahedra: int = 20_000_000,
) -> FreeformTetMeshResult:
    """Convenience API: arbitrary watertight surface -> nonstructured Tet4."""

    if isinstance(surface, (str, Path)):
        shell = load_surface_shell(
            surface,
            inside_region=region,
            name="outer_boundary",
            coordinate_scale=coordinate_scale,
            maximum_triangle_area_m2=maximum_triangle_area_m2,
        )
    else:
        shell = SurfaceShell(
            surface=surface,
            inside_region=region,
            name="outer_boundary",
            maximum_triangle_area_m2=maximum_triangle_area_m2,
        )
    return mesh_freeform_tetgen(
        (shell,),
        (
            FreeformRegion(
                region=region,
                seed_m_xyz=seed_m_xyz,
                name=region_name,
                maximum_tetra_volume_m3=maximum_tetra_volume_m3,
            ),
        ),
        config=config,
        maximum_tetrahedra=maximum_tetrahedra,
    )


__all__ = [
    "FreeformClearanceAudit",
    "ShellClearancePair",
    "FreeformRegion",
    "FreeformTetMeshResult",
    "SurfaceShell",
    "assemble_freeform_plc",
    "audit_freeform_shell_clearance",
    "load_surface_shell",
    "mesh_closed_surface_tetgen",
    "mesh_freeform_tetgen",
    "orient_closed_surface",
    "plc_from_volume_mesh",
    "region_seeds_from_volume_mesh",
]


def plc_from_volume_mesh(
    mesh: VolumeMesh,
    *,
    outside_region: int = -1,
    strict: bool = True,
) -> MultiphasePLC:
    """Recover one conforming multi-domain PLC from an existing Tet4 mesh.

    Exterior faces and material interfaces are extracted directly from Tet4
    adjacency.  Same-material interior faces disappear.  Every emitted face is
    oriented from its owner/left region toward the neighboring/right region or
    the exterior.  This enables COMSOL/Gmsh/legacy Tet4 meshes to be remeshed by
    the same free-form TetGen path without voxelization.
    """

    outside_region = int(outside_region)
    points = mesh.nodes
    tets = mesh.tetrahedra
    regions = mesh.cell_regions
    if not len(tets):
        raise GeometryError("cannot build a PLC from an empty Tet4 mesh")

    # Each candidate is oriented outward from its owning tetrahedron using a
    # geometric centroid test, so an internal owner face points at its neighbor.
    face_owner: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    emitted: list[tuple[tuple[int, int, int], int, int, tuple[str, int, int]]] = []
    local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    centroids = points[tets].mean(axis=1)

    for cell_index, tet in enumerate(tets):
        cell_center = centroids[cell_index]
        for local in local_faces:
            face = tuple(int(tet[index]) for index in local)
            xyz = points[np.asarray(face, dtype=np.int64)]
            face_center = xyz.mean(axis=0)
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            # Outward normal must point away from the tetra centroid.
            if float(np.dot(normal, cell_center - face_center)) > 0.0:
                face = (face[0], face[2], face[1])
            key = tuple(sorted(face))
            previous = face_owner.pop(key, None)
            if previous is None:
                face_owner[key] = (cell_index, face)
                continue
            other_index, other_face = previous
            left_region = int(regions[other_index])
            right_region = int(regions[cell_index])
            if left_region == right_region:
                continue
            # other_face is outward from other cell and therefore toward this cell.
            pair = tuple(sorted((left_region, right_region)))
            emitted.append((other_face, left_region, right_region, ("interface", *pair)))

    # Remaining owner faces are exterior boundary faces.
    for _, (cell_index, face) in face_owner.items():
        region = int(regions[cell_index])
        emitted.append((face, region, outside_region, ("exterior", region, outside_region)))

    if not emitted:
        raise GeometryError("Tet4 mesh yielded no exterior/interface PLC faces")

    keys = sorted({record[3] for record in emitted})
    marker_for = {key: index + 1 for index, key in enumerate(keys)}
    marker_names: dict[int, str] = {}
    marker_pairs: dict[int, tuple[int | None, int | None]] = {}
    for key, marker in marker_for.items():
        if key[0] == "interface":
            a, b = int(key[1]), int(key[2])
            marker_names[marker] = f"interface_{a}_{b}"
            marker_pairs[marker] = (a, b)
        else:
            region = int(key[1])
            marker_names[marker] = f"exterior_{region}"
            marker_pairs[marker] = (region, None)

    faces = np.asarray([record[0] for record in emitted], dtype=np.int64)
    facet_markers = np.asarray([marker_for[record[3]] for record in emitted], dtype=np.int32)
    left = np.asarray([record[1] for record in emitted], dtype=np.int32)
    right = np.asarray([record[2] for record in emitted], dtype=np.int32)

    referenced = np.unique(faces)
    remap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced), dtype=np.int64)
    compact_faces = remap[faces]
    compact_vertices = points[referenced]

    plc = MultiphasePLC(
        vertices=compact_vertices,
        triangles=compact_faces,
        facet_markers=facet_markers,
        left_regions=left,
        right_regions=right,
        outside_region=outside_region,
        marker_names=marker_names,
        marker_region_pairs=marker_pairs,
        fixed_vertices=np.ones(len(compact_vertices), dtype=bool),
        metadata={
            "source": "existing-tet4-volume-mesh",
            "source_nodes": mesh.n_nodes,
            "source_tetrahedra": mesh.n_cells,
            "source_regions": tuple(map(int, np.unique(mesh.cell_regions))),
        },
    )
    audit = audit_multiphase_plc(plc)
    if strict and not audit.valid:
        raise GeometryError(f"recovered Tet4 PLC failed topology audit: {audit}")
    if not strict:
        plc = MultiphasePLC(
            vertices=plc.vertices,
            triangles=plc.triangles,
            facet_markers=plc.facet_markers,
            left_regions=plc.left_regions,
            right_regions=plc.right_regions,
            outside_region=plc.outside_region,
            marker_names=plc.marker_names,
            marker_region_pairs=plc.marker_region_pairs,
            fixed_vertices=plc.fixed_vertices,
            metadata={**plc.metadata, "plc_audit": audit, "strict_topology": False},
        )
    return plc


def region_seeds_from_volume_mesh(
    mesh: VolumeMesh,
    *,
    maximum_tetra_volume_m3: Mapping[int, float] | None = None,
) -> tuple[FreeformRegion, ...]:
    """Create one interior TetGen seed per connected material component.

    Disconnected particles may share the same region ID; TetGen receives a seed
    for every component while retaining a common material attribute.
    """

    limits = {int(k): float(v) for k, v in (maximum_tetra_volume_m3 or {}).items()}
    tets = mesh.tetrahedra
    regions = mesh.cell_regions
    cell_points = mesh.nodes[tets]
    centroids = cell_points.mean(axis=1)
    volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            cell_points[:, 1] - cell_points[:, 0],
            np.cross(cell_points[:, 2] - cell_points[:, 0], cell_points[:, 3] - cell_points[:, 0]),
        )
    ) / 6.0

    same_region_neighbors: list[list[int]] = [[] for _ in range(mesh.n_cells)]
    owner: dict[tuple[int, int, int], int] = {}
    for cell_index, tet in enumerate(tets):
        for local in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            key = tuple(sorted(int(tet[i]) for i in local))
            other = owner.pop(key, None)
            if other is None:
                owner[key] = cell_index
            elif int(regions[other]) == int(regions[cell_index]):
                same_region_neighbors[other].append(cell_index)
                same_region_neighbors[cell_index].append(other)

    seen = np.zeros(mesh.n_cells, dtype=bool)
    seeds: list[FreeformRegion] = []
    component_counts: dict[int, int] = defaultdict(int)
    for root in range(mesh.n_cells):
        if seen[root]:
            continue
        region = int(regions[root])
        stack = [root]
        seen[root] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in same_region_neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        component_counts[region] += 1
        # Any tetra centroid is strictly interior; selecting the largest cell
        # maximizes numerical distance from its faces among available candidates.
        best = component[int(np.argmax(volumes[np.asarray(component, dtype=np.int64)]))]
        base_name = mesh.region_names.get(region, f"domain_{region}")
        suffix = component_counts[region]
        seeds.append(
            FreeformRegion(
                region=region,
                seed_m_xyz=tuple(map(float, centroids[best])),
                name=base_name if suffix == 1 else f"{base_name}_component_{suffix}",
                maximum_tetra_volume_m3=limits.get(region),
            )
        )
    return tuple(seeds)
