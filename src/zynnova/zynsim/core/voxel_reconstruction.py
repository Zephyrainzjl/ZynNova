"""High-fidelity, quality-guarded multi-label voxel-to-Tet4 reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..exceptions import MeshError
from .mesh import Mesh
from .voxel import voxel_interface_areas, voxel_to_tetrahedral_mesh


RegionPair = tuple[int, int]
Refinement = int | tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class VoxelFEMReconstructionConfig:
    """Geometry fidelity, smoothing, and FEM-quality controls.

    ``smoothing`` is normalized to ``[0, 1]``. A value of zero reproduces the
    exact voxel geometry. Smoothing never changes a voxel label or tetrahedral
    region identifier. ``refinement`` subdivides each input voxel before
    smoothing and is useful when curved-looking interfaces are needed without
    discarding one-voxel features.
    """

    smoothing: float = 0.0
    refinement: Refinement = 1
    surface_iterations: int = 12
    volume_iterations: int = 24
    surface_relaxation: float = 0.45
    taubin_mu: float = -0.47
    volume_relaxation: float = 0.72
    max_displacement_voxels: float = 0.45
    preserve_outer_boundary: bool = True
    preserve_multiphase_junctions: bool = True
    minimum_volume_ratio: float = 0.15
    minimum_mean_ratio: float = 0.20
    minimum_quality_retention: float = 0.35
    maximum_region_volume_change: float = 0.03
    backtracking_steps: int = 12
    quality_failure: Literal["fallback", "raise"] = "fallback"
    validate_manifold: bool = True
    maximum_tetrahedra: int = 20_000_000

    def __post_init__(self) -> None:
        smoothing = float(self.smoothing)
        if not np.isfinite(smoothing) or not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must lie in [0, 1]")
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "refinement", _normalize_refinement(self.refinement))
        if self.surface_iterations < 0 or self.volume_iterations < 0:
            raise ValueError("smoothing iteration counts cannot be negative")
        if not 0.0 < self.surface_relaxation <= 0.8:
            raise ValueError("surface_relaxation must lie in (0, 0.8]")
        if not -0.8 <= self.taubin_mu < 0.0:
            raise ValueError("taubin_mu must lie in [-0.8, 0)")
        if not 0.0 < self.volume_relaxation <= 1.0:
            raise ValueError("volume_relaxation must lie in (0, 1]")
        if not 0.0 <= self.max_displacement_voxels <= 0.5:
            raise ValueError("max_displacement_voxels must lie in [0, 0.5]")
        for name, value in (
            ("minimum_volume_ratio", self.minimum_volume_ratio),
            ("minimum_mean_ratio", self.minimum_mean_ratio),
            ("minimum_quality_retention", self.minimum_quality_retention),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not 0.0 <= self.maximum_region_volume_change <= 1.0:
            raise ValueError("maximum_region_volume_change must lie in [0, 1]")
        if self.backtracking_steps < 0:
            raise ValueError("backtracking_steps cannot be negative")
        if self.quality_failure not in {"fallback", "raise"}:
            raise ValueError("quality_failure must be 'fallback' or 'raise'")
        if int(self.maximum_tetrahedra) < 6:
            raise ValueError("maximum_tetrahedra must be at least 6")
        object.__setattr__(self, "maximum_tetrahedra", int(self.maximum_tetrahedra))


@dataclass(frozen=True, slots=True)
class TetMeshQualitySummary:
    """Dimensionless Tet4 mean-ratio and physical volume statistics."""

    minimum_mean_ratio: float
    percentile_01_mean_ratio: float
    median_mean_ratio: float
    mean_mean_ratio: float
    maximum_mean_ratio: float
    minimum_volume_m3: float
    median_volume_m3: float
    maximum_volume_m3: float
    inverted_tetrahedra: int
    degenerate_tetrahedra: int


@dataclass(frozen=True, slots=True)
class VoxelFEMReconstructionReport:
    """Audit trail for fidelity, topology, smoothing, and FEM readiness."""

    requested_smoothing: float
    applied_smoothing_scale: float
    backtracking_steps_used: int
    refinement: tuple[int, int, int]
    input_voxel_shape: tuple[int, int, int]
    reconstructed_voxel_shape: tuple[int, int, int]
    interface_triangle_count: int
    interface_node_count: int
    movable_interface_node_count: int
    fixed_multiphase_junction_count: int
    maximum_displacement_m: float
    maximum_displacement_voxels: float
    minimum_cell_volume_ratio: float
    maximum_region_volume_change: float
    region_volume_change: Mapping[int, float]
    quality_before: TetMeshQualitySummary
    quality_after: TetMeshQualitySummary
    exact_region_assignment: bool
    topology_preserved: bool
    manifold: bool
    fem_ready: bool
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class VoxelFEMReconstructionResult:
    """A conforming multi-region Tet4 mesh and its reconstruction evidence."""

    mesh: Mesh
    refined_phase_labels: np.ndarray
    input_voxel_size_m: tuple[float, float, float]
    mesh_spacing_m: tuple[float, float, float]
    phase_volume_fractions: Mapping[int, float]
    exact_voxel_interface_area_m2: Mapping[RegionPair, float]
    reconstructed_interface_area_m2: Mapping[RegionPair, float]
    interface_faces: Mapping[RegionPair, np.ndarray]
    region_names: Mapping[int, str]
    domain_selections: Mapping[str, tuple[int, ...]]
    geometric_region_volumes_m3: Mapping[int, float]
    report: VoxelFEMReconstructionReport

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self.refined_phase_labels, dtype=np.int32)
        object.__setattr__(self, "refined_phase_labels", labels)
        object.__setattr__(
            self,
            "phase_volume_fractions",
            dict(self.phase_volume_fractions),
        )
        object.__setattr__(
            self,
            "exact_voxel_interface_area_m2",
            dict(self.exact_voxel_interface_area_m2),
        )
        object.__setattr__(
            self,
            "reconstructed_interface_area_m2",
            dict(self.reconstructed_interface_area_m2),
        )
        object.__setattr__(
            self,
            "interface_faces",
            {
                pair: np.ascontiguousarray(faces, dtype=np.int64)
                for pair, faces in self.interface_faces.items()
            },
        )
        object.__setattr__(self, "region_names", dict(self.region_names))
        object.__setattr__(
            self,
            "domain_selections",
            dict(self.domain_selections),
        )
        object.__setattr__(
            self,
            "geometric_region_volumes_m3",
            dict(self.geometric_region_volumes_m3),
        )


def reconstruct_voxel_fem_mesh(
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | tuple[float, float, float] = 1.0,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    region_names: Mapping[int, str] | None = None,
    config: VoxelFEMReconstructionConfig | None = None,
) -> VoxelFEMReconstructionResult:
    """Reconstruct a FEM-ready, multi-material Tet4 mesh from integer voxels.

    The material topology and cell-region assignment remain exact. Interface
    smoothing is implemented as non-shrinking surface graph smoothing followed
    by a harmonic volume deformation. A global line search rejects inverted,
    collapsed, low-quality, or excessive region-volume-changing candidates.
    """

    resolved = config or VoxelFEMReconstructionConfig()
    labels = _integer_labels(phase_labels)
    spacing = _normalize_spacing(voxel_size_m)
    refinement = resolved.refinement
    assert isinstance(refinement, tuple)
    refined_shape = tuple(
        int(labels.shape[axis]) * refinement[axis] for axis in range(3)
    )
    tetrahedra = 6
    for extent in refined_shape:
        tetrahedra *= extent
    if tetrahedra > resolved.maximum_tetrahedra:
        raise MeshError(
            "refined voxel mesh would contain "
            f"{tetrahedra:,} tetrahedra, exceeding maximum_tetrahedra="
            f"{resolved.maximum_tetrahedra:,}"
        )
    refined_labels = _refine_labels(labels, refinement)
    refined_spacing = tuple(
        spacing[axis] / refinement[axis] for axis in range(3)
    )
    base_result = voxel_to_tetrahedral_mesh(
        refined_labels,
        voxel_size_m=refined_spacing,
        origin_m=origin_m,
    )
    base_mesh = base_result.mesh
    interface_faces = voxel_interface_faces(refined_labels)
    interfaces = (
        np.concatenate(tuple(interface_faces.values()), axis=0)
        if interface_faces
        else np.empty((0, 3), dtype=np.int64)
    )
    interface_edges = _surface_edges(interfaces)
    interface_mask, junction_mask = _interface_node_masks(
        base_mesh.n_nodes,
        interface_faces,
    )
    outer_mask = _outer_node_mask(refined_labels.shape)
    movable_mask = interface_mask.copy()
    if resolved.preserve_outer_boundary:
        movable_mask &= ~outer_mask
    if resolved.preserve_multiphase_junctions:
        movable_mask &= ~junction_mask

    base_signed = tetrahedron_signed_six_volumes(base_mesh)
    base_quality_values = tetrahedron_mean_ratio(base_mesh)
    quality_before = _quality_summary(
        base_mesh,
        base_signed,
        reference_signed=base_signed,
    )
    base_region_volumes = _region_volumes(
        base_mesh.cell_regions,
        np.abs(base_signed) / 6.0,
    )

    proposed_nodes = base_mesh.nodes.copy()
    if (
        resolved.smoothing > 0.0
        and resolved.surface_iterations > 0
        and np.any(movable_mask)
    ):
        surface_nodes = _taubin_surface_smoothing(
            base_mesh.nodes,
            interface_edges,
            movable_mask,
            spacing,
            resolved,
        )
        proposed_nodes = _harmonic_volume_deformation(
            base_mesh.nodes,
            surface_nodes,
            interface_mask,
            outer_mask,
            refined_labels.shape,
            resolved,
        )

    raw_displacement = proposed_nodes - base_mesh.nodes
    accepted_nodes, accepted_scale, backtracks, fallback = _quality_guarded_nodes(
        base_mesh,
        raw_displacement,
        base_signed,
        base_quality_values,
        base_region_volumes,
        resolved,
    )
    final_mesh = Mesh(
        nodes=accepted_nodes,
        cells=base_mesh.cells.copy(),
        cell_regions=base_mesh.cell_regions.copy(),
        boundary_faces={
            name: faces.copy() for name, faces in base_mesh.boundary_faces.items()
        },
        metadata={
            **base_mesh.metadata,
            "source": "reconstruct_voxel_fem_mesh",
            "input_voxel_shape": tuple(map(int, labels.shape)),
            "refinement": refinement,
            "requested_smoothing": resolved.smoothing,
            "applied_smoothing_scale": accepted_scale,
            "topology_preserved": True,
        },
    )
    manifold = True
    if resolved.validate_manifold:
        final_mesh.validate_manifold()

    final_signed = tetrahedron_signed_six_volumes(final_mesh)
    quality_after = _quality_summary(
        final_mesh,
        final_signed,
        reference_signed=base_signed,
    )
    final_volumes = np.abs(final_signed) / 6.0
    final_region_volumes = _region_volumes(
        final_mesh.cell_regions,
        final_volumes,
    )
    region_volume_change = _relative_region_volume_change(
        base_region_volumes,
        final_region_volumes,
    )
    volume_ratio = np.divide(
        np.abs(final_signed),
        np.abs(base_signed),
        out=np.ones_like(final_signed),
        where=np.abs(base_signed) > 0.0,
    )
    displacement = final_mesh.nodes - base_mesh.nodes
    normalized_displacement = displacement / np.asarray(spacing)
    normalized_norms = np.linalg.norm(normalized_displacement, axis=1)
    physical_norms = np.linalg.norm(displacement, axis=1)
    reconstructed_area = _interface_areas_from_faces(
        final_mesh.nodes,
        interface_faces,
    )
    values, counts = np.unique(labels, return_counts=True)
    phase_fractions = {
        int(value): float(count / labels.size)
        for value, count in zip(values, counts, strict=True)
    }
    names, selections = _region_metadata(values, region_names)
    exact_assignment = bool(
        np.array_equal(
            final_mesh.cell_regions,
            np.repeat(refined_labels.reshape(-1), 6),
        )
    )
    fem_ready = bool(
        exact_assignment
        and manifold
        and quality_after.inverted_tetrahedra == 0
        and quality_after.degenerate_tetrahedra == 0
        and np.min(volume_ratio) >= resolved.minimum_volume_ratio
    )
    report = VoxelFEMReconstructionReport(
        requested_smoothing=resolved.smoothing,
        applied_smoothing_scale=accepted_scale,
        backtracking_steps_used=backtracks,
        refinement=refinement,
        input_voxel_shape=tuple(map(int, labels.shape)),
        reconstructed_voxel_shape=tuple(map(int, refined_labels.shape)),
        interface_triangle_count=int(len(interfaces)),
        interface_node_count=int(np.sum(interface_mask)),
        movable_interface_node_count=int(np.sum(movable_mask)),
        fixed_multiphase_junction_count=int(
            np.sum(junction_mask)
            if resolved.preserve_multiphase_junctions
            else 0
        ),
        maximum_displacement_m=float(np.max(physical_norms, initial=0.0)),
        maximum_displacement_voxels=float(
            np.max(normalized_norms, initial=0.0)
        ),
        minimum_cell_volume_ratio=float(np.min(volume_ratio, initial=1.0)),
        maximum_region_volume_change=float(
            max(region_volume_change.values(), default=0.0)
        ),
        region_volume_change=region_volume_change,
        quality_before=quality_before,
        quality_after=quality_after,
        exact_region_assignment=exact_assignment,
        topology_preserved=True,
        manifold=manifold,
        fem_ready=fem_ready,
        fallback_used=fallback,
    )
    if not fem_ready:
        raise MeshError("quality-guarded voxel reconstruction is not FEM-ready")
    return VoxelFEMReconstructionResult(
        mesh=final_mesh,
        refined_phase_labels=refined_labels,
        input_voxel_size_m=spacing,
        mesh_spacing_m=refined_spacing,
        phase_volume_fractions=phase_fractions,
        exact_voxel_interface_area_m2=voxel_interface_areas(
            labels,
            voxel_size_m=spacing,
        ),
        reconstructed_interface_area_m2=reconstructed_area,
        interface_faces=interface_faces,
        region_names=names,
        domain_selections=selections,
        geometric_region_volumes_m3=final_region_volumes,
        report=report,
    )


def voxel_to_fem_mesh(
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | tuple[float, float, float] = 1.0,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    region_names: Mapping[int, str] | None = None,
    smoothing: float = 0.0,
    refinement: Refinement = 1,
) -> VoxelFEMReconstructionResult:
    """Convenience interface for exact or smoothed FEM-ready reconstruction."""

    return reconstruct_voxel_fem_mesh(
        phase_labels,
        voxel_size_m=voxel_size_m,
        origin_m=origin_m,
        region_names=region_names,
        config=VoxelFEMReconstructionConfig(
            smoothing=smoothing,
            refinement=refinement,
        ),
    )


def voxel_interface_faces(
    phase_labels: np.ndarray,
) -> dict[RegionPair, np.ndarray]:
    """Return conforming interface triangles keyed by sorted region pairs.

    Connectivity matches the global Freudenthal split used by
    :func:`voxel_to_tetrahedral_mesh`.
    """

    labels = _integer_labels(phase_labels)
    nx, ny, nz = map(int, labels.shape)
    ny_nodes = ny + 1
    nz_nodes = nz + 1
    groups: dict[RegionPair, list[np.ndarray]] = {}

    def node(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (i * ny_nodes + j) * nz_nodes + k

    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        lower_labels = labels[tuple(lower_slice)]
        upper_labels = labels[tuple(upper_slice)]
        changed = lower_labels != upper_labels
        if not np.any(changed):
            continue
        i, j, k = np.nonzero(changed)
        pairs = np.stack(
            (lower_labels[changed], upper_labels[changed]),
            axis=1,
        ).astype(np.int64)
        pairs.sort(axis=1)
        if axis == 0:
            plane = i + 1
            c00 = node(plane, j, k)
            c10 = node(plane, j + 1, k)
            c11 = node(plane, j + 1, k + 1)
            c01 = node(plane, j, k + 1)
        elif axis == 1:
            plane = j + 1
            c00 = node(i, plane, k)
            c10 = node(i + 1, plane, k)
            c11 = node(i + 1, plane, k + 1)
            c01 = node(i, plane, k + 1)
        else:
            plane = k + 1
            c00 = node(i, j, plane)
            c10 = node(i + 1, j, plane)
            c11 = node(i + 1, j + 1, plane)
            c01 = node(i, j + 1, plane)
        triangles = np.stack(
            (
                np.stack((c00, c10, c11), axis=1),
                np.stack((c00, c11, c01), axis=1),
            ),
            axis=1,
        ).reshape(-1, 3)
        repeated_pairs = np.repeat(pairs, 2, axis=0)
        unique_pairs, inverse = np.unique(
            repeated_pairs,
            axis=0,
            return_inverse=True,
        )
        for pair_index, pair_values in enumerate(unique_pairs):
            pair = (int(pair_values[0]), int(pair_values[1]))
            groups.setdefault(pair, []).append(
                triangles[inverse == pair_index]
            )
    return {
        pair: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.int64)
        for pair, parts in sorted(groups.items())
    }


def tetrahedron_signed_six_volumes(mesh: Mesh) -> np.ndarray:
    """Return signed determinants, equal to six times Tet4 volumes."""

    return _signed_six_volumes(mesh.nodes, mesh.cells)


def _signed_six_volumes(
    nodes: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    coordinates = nodes[cells]
    jacobians = np.stack(
        (
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
            coordinates[:, 3] - coordinates[:, 0],
        ),
        axis=1,
    )
    return np.linalg.det(jacobians)


def tetrahedron_mean_ratio(mesh: Mesh) -> np.ndarray:
    """Return the standard Tet4 mean-ratio quality in ``[0, 1]``."""

    return _mean_ratio(mesh.nodes, mesh.cells)


def _mean_ratio(
    nodes: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    coordinates = nodes[cells]
    signed = _signed_six_volumes(nodes, cells)
    volumes = np.abs(signed) / 6.0
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared_edge_sum = np.zeros(len(cells), dtype=np.float64)
    for first, second in edge_pairs:
        difference = coordinates[:, first] - coordinates[:, second]
        squared_edge_sum += np.einsum(
            "ij,ij->i",
            difference,
            difference,
        )
    numerator = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0)
    return np.divide(
        numerator,
        squared_edge_sum,
        out=np.zeros_like(numerator),
        where=squared_edge_sum > 0.0,
    )


def _quality_summary(
    mesh: Mesh,
    signed: np.ndarray,
    *,
    reference_signed: np.ndarray,
) -> TetMeshQualitySummary:
    quality = tetrahedron_mean_ratio(mesh)
    volumes = np.abs(signed) / 6.0
    sign_product = signed * reference_signed
    scale = np.max(
        np.abs(mesh.nodes[mesh.cells] - mesh.nodes[mesh.cells][:, :1]),
        axis=(1, 2),
    )
    determinant_threshold = (
        64.0
        * np.finfo(float).eps
        * np.maximum(scale**3, np.finfo(float).tiny)
    )
    return TetMeshQualitySummary(
        minimum_mean_ratio=float(np.min(quality)),
        percentile_01_mean_ratio=float(np.percentile(quality, 1.0)),
        median_mean_ratio=float(np.median(quality)),
        mean_mean_ratio=float(np.mean(quality)),
        maximum_mean_ratio=float(np.max(quality)),
        minimum_volume_m3=float(np.min(volumes)),
        median_volume_m3=float(np.median(volumes)),
        maximum_volume_m3=float(np.max(volumes)),
        inverted_tetrahedra=int(np.sum(sign_product <= 0.0)),
        degenerate_tetrahedra=int(
            np.sum(np.abs(signed) <= determinant_threshold)
        ),
    )


def _taubin_surface_smoothing(
    base_nodes: np.ndarray,
    edges: np.ndarray,
    movable: np.ndarray,
    input_spacing: tuple[float, float, float],
    config: VoxelFEMReconstructionConfig,
) -> np.ndarray:
    points = base_nodes.copy()
    lam = config.surface_relaxation * config.smoothing
    mu = config.taubin_mu * config.smoothing
    max_displacement = config.max_displacement_voxels * config.smoothing
    for _ in range(config.surface_iterations):
        average = _edge_neighbor_average(points, edges)
        points[movable] += lam * (average[movable] - points[movable])
        points = _clamp_surface_displacement(
            points,
            base_nodes,
            movable,
            input_spacing,
            max_displacement,
        )
        average = _edge_neighbor_average(points, edges)
        points[movable] += mu * (average[movable] - points[movable])
        points = _clamp_surface_displacement(
            points,
            base_nodes,
            movable,
            input_spacing,
            max_displacement,
        )
    return points


def _harmonic_volume_deformation(
    base_nodes: np.ndarray,
    surface_nodes: np.ndarray,
    interface_mask: np.ndarray,
    outer_mask: np.ndarray,
    voxel_shape: tuple[int, int, int],
    config: VoxelFEMReconstructionConfig,
) -> np.ndarray:
    if config.volume_iterations == 0:
        result = base_nodes.copy()
        result[interface_mask] = surface_nodes[interface_mask]
        return result
    node_shape = tuple(value + 1 for value in voxel_shape)
    displacement = np.zeros((*node_shape, 3), dtype=np.float64)
    target = (surface_nodes - base_nodes).reshape((*node_shape, 3))
    fixed = interface_mask.reshape(node_shape).copy()
    if config.preserve_outer_boundary:
        fixed |= outer_mask.reshape(node_shape)
    displacement[fixed] = target[fixed]
    free = ~fixed
    counts = np.zeros(node_shape, dtype=np.float64)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        counts[tuple(lower)] += 1.0
        counts[tuple(upper)] += 1.0
    for _ in range(config.volume_iterations):
        neighbor_sum = np.zeros_like(displacement)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            neighbor_sum[tuple(lower)] += displacement[tuple(upper)]
            neighbor_sum[tuple(upper)] += displacement[tuple(lower)]
        average = neighbor_sum / counts[..., None]
        displacement[free] = (
            (1.0 - config.volume_relaxation) * displacement[free]
            + config.volume_relaxation * average[free]
        )
        displacement[fixed] = target[fixed]
    return base_nodes + displacement.reshape(-1, 3)


def _quality_guarded_nodes(
    mesh: Mesh,
    raw_displacement: np.ndarray,
    base_signed: np.ndarray,
    base_quality: np.ndarray,
    base_region_volumes: Mapping[int, float],
    config: VoxelFEMReconstructionConfig,
) -> tuple[np.ndarray, float, int, bool]:
    if not np.any(raw_displacement):
        return mesh.nodes.copy(), 0.0, 0, False
    effective_minimum_quality = min(
        config.minimum_mean_ratio,
        float(np.min(base_quality)),
    )
    last_reason = "unknown quality failure"
    for backtrack in range(config.backtracking_steps + 1):
        scale = 0.5**backtrack
        candidate_nodes = mesh.nodes + scale * raw_displacement
        signed = _signed_six_volumes(candidate_nodes, mesh.cells)
        if np.any(~np.isfinite(signed)):
            last_reason = "non-finite Jacobian"
            continue
        if np.any(signed * base_signed <= 0.0):
            last_reason = "tetrahedron inversion"
            continue
        volume_ratio = np.abs(signed) / np.abs(base_signed)
        if float(np.min(volume_ratio)) < config.minimum_volume_ratio:
            last_reason = "minimum volume ratio"
            continue
        quality = _mean_ratio(candidate_nodes, mesh.cells)
        if float(np.min(quality)) + 1.0e-14 < effective_minimum_quality:
            last_reason = "absolute mean-ratio quality"
            continue
        retention = np.divide(
            quality,
            base_quality,
            out=np.ones_like(quality),
            where=base_quality > 0.0,
        )
        if float(np.min(retention)) < config.minimum_quality_retention:
            last_reason = "relative mean-ratio quality"
            continue
        candidate_region_volumes = _region_volumes(
            mesh.cell_regions,
            np.abs(signed) / 6.0,
        )
        volume_change = _relative_region_volume_change(
            base_region_volumes,
            candidate_region_volumes,
        )
        if (
            max(volume_change.values(), default=0.0)
            > config.maximum_region_volume_change
        ):
            last_reason = "region volume fidelity"
            continue
        return candidate_nodes, scale, backtrack, False
    if config.quality_failure == "raise":
        raise MeshError(
            "requested smoothing could not satisfy FEM quality constraints: "
            f"{last_reason}"
        )
    return mesh.nodes.copy(), 0.0, config.backtracking_steps + 1, True


def _surface_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _edge_neighbor_average(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(
        edges.reshape(-1),
        minlength=len(points),
    ).astype(np.float64)
    result = np.zeros_like(points)
    for component in range(3):
        result[:, component] = (
            np.bincount(
                edges[:, 0],
                weights=points[edges[:, 1], component],
                minlength=len(points),
            )
            + np.bincount(
                edges[:, 1],
                weights=points[edges[:, 0], component],
                minlength=len(points),
            )
        )
    return np.divide(
        result,
        counts[:, None],
        out=points.copy(),
        where=counts[:, None] > 0.0,
    )


def _clamp_surface_displacement(
    points: np.ndarray,
    base: np.ndarray,
    movable: np.ndarray,
    spacing: tuple[float, float, float],
    maximum: float,
) -> np.ndarray:
    if maximum <= 0.0:
        result = points.copy()
        result[movable] = base[movable]
        return result
    result = points.copy()
    displacement = result[movable] - base[movable]
    normalized = displacement / np.asarray(spacing)
    norms = np.linalg.norm(normalized, axis=1)
    factors = np.minimum(
        1.0,
        maximum / np.maximum(norms, np.finfo(float).tiny),
    )
    result[movable] = base[movable] + displacement * factors[:, None]
    return result


def _interface_node_masks(
    node_count: int,
    interface_faces: Mapping[RegionPair, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    membership_count = np.zeros(node_count, dtype=np.int32)
    for faces in interface_faces.values():
        membership_count[np.unique(faces)] += 1
    return membership_count > 0, membership_count > 1


def _outer_node_mask(voxel_shape: tuple[int, int, int]) -> np.ndarray:
    shape = tuple(value + 1 for value in voxel_shape)
    mask = np.zeros(shape, dtype=bool)
    for axis in range(3):
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        mask[tuple(first)] = True
        mask[tuple(last)] = True
    return mask.reshape(-1)


def _interface_areas_from_faces(
    nodes: np.ndarray,
    interface_faces: Mapping[RegionPair, np.ndarray],
) -> dict[RegionPair, float]:
    result: dict[RegionPair, float] = {}
    for pair, faces in interface_faces.items():
        coordinates = nodes[faces]
        area = 0.5 * np.linalg.norm(
            np.cross(
                coordinates[:, 1] - coordinates[:, 0],
                coordinates[:, 2] - coordinates[:, 0],
            ),
            axis=1,
        )
        result[pair] = float(np.sum(area))
    return result


def _region_volumes(
    cell_regions: np.ndarray,
    volumes: np.ndarray,
) -> dict[int, float]:
    regions, inverse = np.unique(cell_regions, return_inverse=True)
    totals = np.bincount(inverse, weights=volumes)
    return {
        int(region): float(total)
        for region, total in zip(regions, totals, strict=True)
    }


def _relative_region_volume_change(
    before: Mapping[int, float],
    after: Mapping[int, float],
) -> dict[int, float]:
    return {
        region: abs(after[region] - volume) / max(
            abs(volume),
            np.finfo(float).tiny,
        )
        for region, volume in before.items()
    }


def _region_metadata(
    values: np.ndarray,
    supplied_names: Mapping[int, str] | None,
) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
    regions = tuple(sorted(map(int, values)))
    supplied = {} if supplied_names is None else {
        int(region): str(name) for region, name in supplied_names.items()
    }
    extras = sorted(set(supplied) - set(regions))
    if extras:
        raise ValueError(f"region_names contains labels absent from voxels: {extras}")
    names = {
        region: supplied.get(region, f"region_{region}")
        for region in regions
    }
    if len(set(names.values())) != len(names):
        raise ValueError("region_names must be unique")
    if "all_domains" in names.values():
        raise ValueError("region name 'all_domains' is reserved")
    selections = {
        name: (region,)
        for region, name in names.items()
    }
    selections["all_domains"] = regions
    return names, selections


def _integer_labels(value: np.ndarray) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 3 or min(labels.shape, default=0) < 1:
        raise MeshError("phase_labels must be a non-empty 3-D array")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
            raise MeshError("phase_labels must contain finite integer values")
    minimum = int(np.min(labels))
    maximum = int(np.max(labels))
    limits = np.iinfo(np.int32)
    if minimum < limits.min or maximum > limits.max:
        raise MeshError("phase labels exceed signed 32-bit region identifiers")
    return np.ascontiguousarray(labels, dtype=np.int32)


def _normalize_refinement(value: Refinement) -> tuple[int, int, int]:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("refinement must contain one or three positive integers")
    result = tuple(map(int, raw))
    if min(result) < 1 or max(result) > 8:
        raise ValueError("refinement factors must lie in [1, 8]")
    return result


def _normalize_spacing(
    value: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise MeshError("voxel_size_m must contain one or three positive values")
    return tuple(map(float, raw))


def _refine_labels(
    labels: np.ndarray,
    refinement: tuple[int, int, int],
) -> np.ndarray:
    result = labels
    for axis, factor in enumerate(refinement):
        if factor > 1:
            result = np.repeat(result, factor, axis=axis)
    return np.ascontiguousarray(result, dtype=np.int32)


__all__ = [
    "TetMeshQualitySummary",
    "VoxelFEMReconstructionConfig",
    "VoxelFEMReconstructionReport",
    "VoxelFEMReconstructionResult",
    "reconstruct_voxel_fem_mesh",
    "tetrahedron_mean_ratio",
    "tetrahedron_signed_six_volumes",
    "voxel_interface_faces",
    "voxel_to_fem_mesh",
]
