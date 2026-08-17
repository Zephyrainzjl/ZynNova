"""Explicit stochastic particle/electrolyte microstructures for full cells."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core import VoxelMeshResult, voxel_to_tetrahedral_mesh


NEGATIVE_ACTIVE = 1
NEGATIVE_ELECTROLYTE = 2
SEPARATOR_ELECTROLYTE = 3
POSITIVE_ELECTROLYTE = 4
POSITIVE_ACTIVE = 5

DEFAULT_PHASE_NAMES: Mapping[int, str] = {
    NEGATIVE_ACTIVE: "negative_graphite_particles",
    NEGATIVE_ELECTROLYTE: "negative_electrolyte",
    SEPARATOR_ELECTROLYTE: "separator_electrolyte",
    POSITIVE_ELECTROLYTE: "positive_electrolyte",
    POSITIVE_ACTIVE: "positive_NMC_particles",
}


@dataclass(frozen=True, slots=True)
class ExplicitParticleFullCellConfig:
    """Controls an explicit particle-resolved full-cell representative volume."""

    voxel_shape: tuple[int, int, int] = (48, 28, 28)
    voxel_size_m: float | tuple[float, float, float] = 2.0e-6
    layer_voxels: tuple[int, int, int] = (21, 6, 21)
    negative_particle_count: int = 18
    positive_particle_count: int = 16
    negative_active_fraction: float = 0.48
    positive_active_fraction: float = 0.46
    negative_lobes: tuple[int, int] = (2, 4)
    positive_lobes: tuple[int, int] = (4, 8)
    negative_roughness: float = 0.10
    positive_roughness: float = 0.16
    positive_internal_pore_probability: float = 0.55
    positive_crack_probability: float = 0.45
    calibration_iterations: int = 5
    minimum_particle_voxels: int = 12
    seed: int = 20260730

    def __post_init__(self) -> None:
        if len(self.voxel_shape) != 3 or min(self.voxel_shape) < 4:
            raise ValueError("voxel_shape must contain three values of at least four")
        if len(self.layer_voxels) != 3 or min(self.layer_voxels) < 2:
            raise ValueError("each full-cell layer needs at least two x voxels")
        if sum(self.layer_voxels) != self.voxel_shape[0]:
            raise ValueError("layer_voxels must sum to voxel_shape[0]")
        _normalize_spacing(self.voxel_size_m)
        if min(self.negative_particle_count, self.positive_particle_count) < 1:
            raise ValueError("both electrodes need at least one particle")
        for value in (
            self.negative_active_fraction,
            self.positive_active_fraction,
        ):
            if not 0.05 <= value <= 0.75:
                raise ValueError("active fractions must lie in [0.05, 0.75]")
        for lobes in (self.negative_lobes, self.positive_lobes):
            if len(lobes) != 2 or lobes[0] < 1 or lobes[1] < lobes[0]:
                raise ValueError("lobe ranges must be increasing positive pairs")
        if not 0.0 <= self.negative_roughness <= 0.35:
            raise ValueError("negative_roughness must lie in [0, 0.35]")
        if not 0.0 <= self.positive_roughness <= 0.35:
            raise ValueError("positive_roughness must lie in [0, 0.35]")
        for probability in (
            self.positive_internal_pore_probability,
            self.positive_crack_probability,
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("particle feature probabilities must lie in [0, 1]")
        if self.calibration_iterations < 1:
            raise ValueError("calibration_iterations must be positive")
        if self.minimum_particle_voxels < 1:
            raise ValueError("minimum_particle_voxels must be positive")


@dataclass(frozen=True, slots=True)
class ParticleDescriptor:
    """Reproducible geometry metadata for one irregular active particle."""

    particle_id: int
    electrode: str
    center_m: tuple[float, float, float]
    semi_axes_m: tuple[float, float, float]
    lobe_count: int
    roughness_amplitude: float
    internal_pore_count: int
    cracked: bool


@dataclass(frozen=True, slots=True)
class ParticleVoxelStatistics:
    particle_id: int
    phase: int
    voxel_count: int
    volume_m3: float
    surface_area_m2: float
    equivalent_diameter_m: float
    surface_area_ratio: float
    euler_characteristic: int


@dataclass(frozen=True, slots=True)
class ExplicitMicrostructureTetMesh:
    """A Tet4 microstructure plus COMSOL-ready names and region unions."""

    voxel_result: VoxelMeshResult
    region_names: Mapping[int, str]
    domain_selections: Mapping[str, tuple[int, ...]]
    separate_particles: bool

    @property
    def mesh(self):
        return self.voxel_result.mesh


@dataclass(frozen=True, slots=True)
class ExplicitParticleFullCell:
    """Five-phase full-cell voxels with individually labelled active particles."""

    config: ExplicitParticleFullCellConfig
    phase_labels: np.ndarray
    particle_ids: np.ndarray
    particles: tuple[ParticleDescriptor, ...]
    achieved_active_fractions: Mapping[str, float]
    electrolyte_percolation: Mapping[str, bool]
    phase_names: Mapping[int, str] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_NAMES)
    )

    def __post_init__(self) -> None:
        phase_labels = np.ascontiguousarray(self.phase_labels, dtype=np.int32)
        particle_ids = np.ascontiguousarray(self.particle_ids, dtype=np.int32)
        if phase_labels.shape != self.config.voxel_shape:
            raise ValueError("phase_labels shape differs from the microstructure config")
        if particle_ids.shape != phase_labels.shape:
            raise ValueError("particle_ids shape differs from phase_labels")
        known = set(DEFAULT_PHASE_NAMES)
        unknown = sorted(set(map(int, np.unique(phase_labels))) - known)
        if unknown:
            raise ValueError(f"unknown explicit full-cell phase labels {unknown}")
        active = np.isin(phase_labels, (NEGATIVE_ACTIVE, POSITIVE_ACTIVE))
        if np.any((particle_ids > 0) != active):
            raise ValueError("particle IDs must be positive exactly in active voxels")
        object.__setattr__(self, "phase_labels", phase_labels)
        object.__setattr__(self, "particle_ids", particle_ids)
        object.__setattr__(self, "phase_names", dict(self.phase_names))
        object.__setattr__(
            self,
            "achieved_active_fractions",
            dict(self.achieved_active_fractions),
        )
        object.__setattr__(
            self,
            "electrolyte_percolation",
            dict(self.electrolyte_percolation),
        )

    @property
    def voxel_size_m(self) -> tuple[float, float, float]:
        return _normalize_spacing(self.config.voxel_size_m)

    @property
    def physical_lengths_m(self) -> tuple[float, float, float]:
        return tuple(
            self.config.voxel_shape[axis] * self.voxel_size_m[axis]
            for axis in range(3)
        )

    @property
    def phase_volume_fractions(self) -> dict[int, float]:
        values, counts = np.unique(self.phase_labels, return_counts=True)
        return {
            int(value): float(count / self.phase_labels.size)
            for value, count in zip(values, counts, strict=True)
        }

    @property
    def particle_counts(self) -> dict[str, int]:
        negative = sum(item.electrode == "negative" for item in self.particles)
        positive = sum(item.electrode == "positive" for item in self.particles)
        return {"negative": negative, "positive": positive}

    def particle_statistics(self) -> tuple[ParticleVoxelStatistics, ...]:
        spacing = self.voxel_size_m
        voxel_volume = float(np.prod(spacing))
        phase_by_id = {
            item.particle_id: (
                NEGATIVE_ACTIVE
                if item.electrode == "negative"
                else POSITIVE_ACTIVE
            )
            for item in self.particles
        }
        statistics: list[ParticleVoxelStatistics] = []
        for particle_id in sorted(phase_by_id):
            mask = self.particle_ids == particle_id
            voxel_count = int(np.sum(mask))
            volume = voxel_count * voxel_volume
            area = voxel_surface_area(mask, spacing)
            equivalent_diameter = (
                2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                if volume > 0.0
                else 0.0
            )
            sphere_area = (
                4.0 * np.pi * (0.5 * equivalent_diameter) ** 2
                if equivalent_diameter > 0.0
                else 0.0
            )
            statistics.append(
                ParticleVoxelStatistics(
                    particle_id=particle_id,
                    phase=phase_by_id[particle_id],
                    voxel_count=voxel_count,
                    volume_m3=volume,
                    surface_area_m2=area,
                    equivalent_diameter_m=equivalent_diameter,
                    surface_area_ratio=area / max(sphere_area, 1.0e-300),
                    euler_characteristic=cubical_euler_characteristic(mask),
                )
            )
        return tuple(statistics)

    def region_labels(
        self,
        *,
        separate_particles: bool = True,
    ) -> tuple[np.ndarray, dict[int, str], dict[str, tuple[int, ...]]]:
        """Return voxel region codes, labels, and COMSOL aggregate selections."""

        if not separate_particles:
            labels = self.phase_labels.copy()
            region_names = dict(self.phase_names)
            selections = {
                "negative_active_particles": (NEGATIVE_ACTIVE,),
                "positive_active_particles": (POSITIVE_ACTIVE,),
                "all_active_material": (NEGATIVE_ACTIVE, POSITIVE_ACTIVE),
                "all_electrolyte": (
                    NEGATIVE_ELECTROLYTE,
                    SEPARATOR_ELECTROLYTE,
                    POSITIVE_ELECTROLYTE,
                ),
                "negative_electrode_all": (
                    NEGATIVE_ACTIVE,
                    NEGATIVE_ELECTROLYTE,
                ),
                "positive_electrode_all": (
                    POSITIVE_ACTIVE,
                    POSITIVE_ELECTROLYTE,
                ),
                "separator_all": (SEPARATOR_ELECTROLYTE,),
                "full_cell_all_domains": tuple(sorted(DEFAULT_PHASE_NAMES)),
            }
            return labels, region_names, selections

        labels = self.phase_labels.copy()
        region_names = {
            NEGATIVE_ELECTROLYTE: self.phase_names[NEGATIVE_ELECTROLYTE],
            SEPARATOR_ELECTROLYTE: self.phase_names[SEPARATOR_ELECTROLYTE],
            POSITIVE_ELECTROLYTE: self.phase_names[POSITIVE_ELECTROLYTE],
        }
        negative_regions: list[int] = []
        positive_regions: list[int] = []
        for particle in self.particles:
            region = 1000 + particle.particle_id
            labels[self.particle_ids == particle.particle_id] = region
            prefix = "negative" if particle.electrode == "negative" else "positive"
            region_names[region] = f"{prefix}_particle_{particle.particle_id:04d}"
            if particle.electrode == "negative":
                negative_regions.append(region)
            else:
                positive_regions.append(region)
        all_active = tuple(negative_regions + positive_regions)
        all_electrolyte = (
            NEGATIVE_ELECTROLYTE,
            SEPARATOR_ELECTROLYTE,
            POSITIVE_ELECTROLYTE,
        )
        selections = {
            "negative_active_particles": tuple(negative_regions),
            "positive_active_particles": tuple(positive_regions),
            "all_active_material": all_active,
            "all_electrolyte": all_electrolyte,
            "negative_electrode_all": (
                NEGATIVE_ELECTROLYTE,
                *negative_regions,
            ),
            "separator_all": (SEPARATOR_ELECTROLYTE,),
            "positive_electrode_all": (
                POSITIVE_ELECTROLYTE,
                *positive_regions,
            ),
            "full_cell_all_domains": tuple(
                sorted((*all_electrolyte, *all_active))
            ),
        }
        return labels, region_names, selections

    def tetrahedralize(
        self,
        *,
        separate_particles: bool = True,
        origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> ExplicitMicrostructureTetMesh:
        labels, names, selections = self.region_labels(
            separate_particles=separate_particles
        )
        result = voxel_to_tetrahedral_mesh(
            labels,
            voxel_size_m=self.voxel_size_m,
            origin_m=origin_m,
        )
        return ExplicitMicrostructureTetMesh(
            voxel_result=result,
            region_names=names,
            domain_selections=selections,
            separate_particles=separate_particles,
        )


@dataclass(frozen=True, slots=True)
class _ParticleRecipe:
    particle_id: int
    electrode: str
    center_m: np.ndarray
    semi_axes_m: np.ndarray
    rotation: np.ndarray
    lobe_offsets: tuple[np.ndarray, ...]
    lobe_scales: tuple[np.ndarray, ...]
    roughness_amplitude: float
    roughness_frequency: tuple[int, int]
    roughness_phase: tuple[float, float]
    pore_offsets: tuple[np.ndarray, ...]
    pore_radii: tuple[float, ...]
    crack_normal: np.ndarray | None
    crack_tangent: np.ndarray | None
    crack_width_fraction: float


def generate_explicit_particle_full_cell(
    config: ExplicitParticleFullCellConfig | None = None,
) -> ExplicitParticleFullCell:
    """Generate a seeded, nonconvex particle/electrolyte full-cell volume."""

    resolved = config or ExplicitParticleFullCellConfig()
    rng = np.random.default_rng(resolved.seed)
    spacing = _normalize_spacing(resolved.voxel_size_m)
    nx, ny, nz = resolved.voxel_shape
    negative_nx, separator_nx, positive_nx = resolved.layer_voxels

    phase_labels = np.empty((nx, ny, nz), dtype=np.int32)
    phase_labels[:negative_nx] = NEGATIVE_ELECTROLYTE
    phase_labels[
        negative_nx : negative_nx + separator_nx
    ] = SEPARATOR_ELECTROLYTE
    phase_labels[negative_nx + separator_nx :] = POSITIVE_ELECTROLYTE
    particle_ids = np.zeros_like(phase_labels)

    negative_ids, negative_recipes, negative_scale = _generate_electrode_particles(
        shape=(negative_nx, ny, nz),
        spacing=spacing,
        x_origin_m=0.0,
        count=resolved.negative_particle_count,
        target_fraction=resolved.negative_active_fraction,
        lobe_range=resolved.negative_lobes,
        roughness=resolved.negative_roughness,
        electrode="negative",
        particle_id_start=1,
        internal_pore_probability=0.0,
        crack_probability=0.0,
        calibration_iterations=resolved.calibration_iterations,
        minimum_particle_voxels=resolved.minimum_particle_voxels,
        rng=rng,
    )
    positive_ids, positive_recipes, positive_scale = _generate_electrode_particles(
        shape=(positive_nx, ny, nz),
        spacing=spacing,
        x_origin_m=(negative_nx + separator_nx) * spacing[0],
        count=resolved.positive_particle_count,
        target_fraction=resolved.positive_active_fraction,
        lobe_range=resolved.positive_lobes,
        roughness=resolved.positive_roughness,
        electrode="positive",
        particle_id_start=1 + resolved.negative_particle_count,
        internal_pore_probability=resolved.positive_internal_pore_probability,
        crack_probability=resolved.positive_crack_probability,
        calibration_iterations=resolved.calibration_iterations,
        minimum_particle_voxels=resolved.minimum_particle_voxels,
        rng=rng,
    )

    negative_slice = slice(0, negative_nx)
    positive_slice = slice(negative_nx + separator_nx, nx)
    negative_phase_labels = phase_labels[negative_slice]
    positive_phase_labels = phase_labels[positive_slice]
    negative_phase_labels[negative_ids > 0] = NEGATIVE_ACTIVE
    positive_phase_labels[positive_ids > 0] = POSITIVE_ACTIVE
    particle_ids[negative_slice] = negative_ids
    particle_ids[positive_slice] = positive_ids

    negative_fraction = float(np.mean(negative_ids > 0))
    positive_fraction = float(np.mean(positive_ids > 0))
    electrolyte_mask = np.isin(
        phase_labels,
        (
            NEGATIVE_ELECTROLYTE,
            SEPARATOR_ELECTROLYTE,
            POSITIVE_ELECTROLYTE,
        ),
    )
    percolation = {
        "negative_electrolyte_x": phase_percolates(
            phase_labels[negative_slice] == NEGATIVE_ELECTROLYTE,
            axis=0,
        ),
        "positive_electrolyte_x": phase_percolates(
            phase_labels[positive_slice] == POSITIVE_ELECTROLYTE,
            axis=0,
        ),
        "full_cell_electrolyte_x": phase_percolates(
            electrolyte_mask,
            axis=0,
        ),
    }
    particles = tuple(
        _public_descriptor(recipe, negative_scale)
        for recipe in negative_recipes
    ) + tuple(
        _public_descriptor(recipe, positive_scale)
        for recipe in positive_recipes
    )
    return ExplicitParticleFullCell(
        config=resolved,
        phase_labels=phase_labels,
        particle_ids=particle_ids,
        particles=particles,
        achieved_active_fractions={
            "negative": negative_fraction,
            "positive": positive_fraction,
        },
        electrolyte_percolation=percolation,
    )


def voxel_surface_area(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("surface-area masks must be three-dimensional")
    area = 0.0
    face_areas = (
        spacing[1] * spacing[2],
        spacing[0] * spacing[2],
        spacing[0] * spacing[1],
    )
    for axis, face_area in enumerate(face_areas):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        area += float(np.sum(values[tuple(lower)] != values[tuple(upper)])) * face_area
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        area += float(np.sum(values[tuple(first)])) * face_area
        area += float(np.sum(values[tuple(last)])) * face_area
    return area


def cubical_euler_characteristic(mask: np.ndarray) -> int:
    """Euler characteristic of a union of occupied closed voxels."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Euler-characteristic masks must be three-dimensional")
    occupied = np.argwhere(values)
    if len(occupied) == 0:
        return 0
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0) + 1
    cropped = values[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    nx, ny, nz = cropped.shape
    vertices = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    edges_x = np.zeros((nx, ny + 1, nz + 1), dtype=bool)
    edges_y = np.zeros((nx + 1, ny, nz + 1), dtype=bool)
    edges_z = np.zeros((nx + 1, ny + 1, nz), dtype=bool)
    faces_x = np.zeros((nx + 1, ny, nz), dtype=bool)
    faces_y = np.zeros((nx, ny + 1, nz), dtype=bool)
    faces_z = np.zeros((nx, ny, nz + 1), dtype=bool)

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                vertices[
                    dx : dx + nx,
                    dy : dy + ny,
                    dz : dz + nz,
                ] |= cropped
    for dy in (0, 1):
        for dz in (0, 1):
            edges_x[:, dy : dy + ny, dz : dz + nz] |= cropped
    for dx in (0, 1):
        for dz in (0, 1):
            edges_y[dx : dx + nx, :, dz : dz + nz] |= cropped
    for dx in (0, 1):
        for dy in (0, 1):
            edges_z[dx : dx + nx, dy : dy + ny, :] |= cropped
    for dx in (0, 1):
        faces_x[dx : dx + nx, :, :] |= cropped
    for dy in (0, 1):
        faces_y[:, dy : dy + ny, :] |= cropped
    for dz in (0, 1):
        faces_z[:, :, dz : dz + nz] |= cropped

    vertex_count = int(np.sum(vertices))
    edge_count = int(np.sum(edges_x) + np.sum(edges_y) + np.sum(edges_z))
    face_count = int(np.sum(faces_x) + np.sum(faces_y) + np.sum(faces_z))
    cube_count = int(np.sum(cropped))
    return vertex_count - edge_count + face_count - cube_count


def connected_component_count(mask: np.ndarray) -> int:
    _, count = _label_components(np.asarray(mask, dtype=bool))
    return count


def phase_percolates(mask: np.ndarray, *, axis: int) -> bool:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or axis not in {0, 1, 2}:
        raise ValueError("percolation requires a 3-D mask and axis 0, 1, or 2")
    labels, count = _label_components(values)
    if count == 0:
        return False
    first = np.take(labels, 0, axis=axis)
    last = np.take(labels, -1, axis=axis)
    first_labels = set(map(int, np.unique(first))) - {0}
    last_labels = set(map(int, np.unique(last))) - {0}
    return bool(first_labels & last_labels)


def _generate_electrode_particles(
    *,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    x_origin_m: float,
    count: int,
    target_fraction: float,
    lobe_range: tuple[int, int],
    roughness: float,
    electrode: str,
    particle_id_start: int,
    internal_pore_probability: float,
    crack_probability: float,
    calibration_iterations: int,
    minimum_particle_voxels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[_ParticleRecipe, ...], float]:
    lengths = np.asarray(shape) * np.asarray(spacing)
    origin = np.asarray((x_origin_m, 0.0, 0.0))
    centers, grid_dimensions = _jittered_centers(
        count,
        origin,
        origin + lengths,
        rng,
    )
    target_particle_volume = target_fraction * float(np.prod(lengths)) / count
    equivalent_radius = (3.0 * target_particle_volume / (4.0 * np.pi)) ** (
        1.0 / 3.0
    )
    grid_cell = lengths / np.asarray(grid_dimensions)
    recipes: list[_ParticleRecipe] = []
    for local_index, center in enumerate(centers):
        particle_id = particle_id_start + local_index
        size_factor = float(np.exp(rng.normal(-0.5 * 0.16**2, 0.16)))
        if electrode == "negative":
            raw_axes = np.asarray(
                (
                    rng.uniform(0.42, 0.62),
                    rng.uniform(1.12, 1.38),
                    rng.uniform(0.92, 1.18),
                )
            )
            rotation = _graphite_rotation(rng)
            base_factor = 0.82
        else:
            raw_axes = rng.uniform(0.82, 1.20, size=3)
            rotation = _random_rotation(rng)
            base_factor = 0.72
        raw_axes /= float(np.prod(raw_axes) ** (1.0 / 3.0))
        axes = equivalent_radius * size_factor * base_factor * raw_axes
        axes = np.minimum(axes, 0.48 * np.max(grid_cell))

        lobe_count = int(rng.integers(lobe_range[0], lobe_range[1] + 1))
        offsets = [np.zeros(3)]
        scales = [np.ones(3)]
        for _ in range(lobe_count - 1):
            direction = rng.normal(size=3)
            if electrode == "negative":
                direction[0] *= 0.30
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            offsets.append(direction * rng.uniform(0.24, 0.54))
            scales.append(rng.uniform(0.56, 0.88, size=3))

        pore_offsets: list[np.ndarray] = []
        pore_radii: list[float] = []
        if electrode == "positive" and rng.random() < internal_pore_probability:
            pore_count = int(rng.integers(1, 4))
            for _ in range(pore_count):
                direction = rng.normal(size=3)
                direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
                pore_offsets.append(direction * rng.uniform(0.05, 0.38))
                pore_radii.append(float(rng.uniform(0.12, 0.23)))

        crack_normal = None
        crack_tangent = None
        crack_width = 0.0
        if electrode == "positive" and rng.random() < crack_probability:
            crack_normal = rng.normal(size=3)
            crack_normal /= max(float(np.linalg.norm(crack_normal)), 1.0e-12)
            crack_tangent = rng.normal(size=3)
            crack_tangent -= crack_normal * float(
                np.dot(crack_tangent, crack_normal)
            )
            crack_tangent /= max(float(np.linalg.norm(crack_tangent)), 1.0e-12)
            crack_width = float(rng.uniform(0.06, 0.11))

        recipes.append(
            _ParticleRecipe(
                particle_id=particle_id,
                electrode=electrode,
                center_m=center,
                semi_axes_m=axes,
                rotation=rotation,
                lobe_offsets=tuple(offsets),
                lobe_scales=tuple(scales),
                roughness_amplitude=roughness * rng.uniform(0.72, 1.0),
                roughness_frequency=(
                    int(rng.integers(3, 7)),
                    int(rng.integers(2, 6)),
                ),
                roughness_phase=(
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                ),
                pore_offsets=tuple(pore_offsets),
                pore_radii=tuple(pore_radii),
                crack_normal=crack_normal,
                crack_tangent=crack_tangent,
                crack_width_fraction=crack_width,
            )
        )

    scale = 1.0
    labels = np.zeros(shape, dtype=np.int32)
    for _ in range(calibration_iterations):
        labels = _rasterize_recipes(
            shape,
            spacing,
            origin,
            tuple(recipes),
            scale,
        )
        fraction = float(np.mean(labels > 0))
        if fraction <= 0.0:
            scale *= 1.5
        else:
            update = float(
                np.clip(
                    (target_fraction / fraction) ** (1.0 / 3.0),
                    0.82,
                    1.22,
                )
            )
            scale *= update
    labels = _rasterize_recipes(
        shape,
        spacing,
        origin,
        tuple(recipes),
        scale,
    )
    labels = _keep_largest_particle_components(labels)
    counts = {
        int(particle_id): int(np.sum(labels == particle_id))
        for particle_id in range(particle_id_start, particle_id_start + count)
    }
    too_small = {
        particle_id: voxels
        for particle_id, voxels in counts.items()
        if voxels < minimum_particle_voxels
    }
    if too_small:
        raise RuntimeError(
            f"{electrode} particles are under-resolved; increase voxel resolution "
            f"or reduce particle count: {too_small}"
        )
    return labels, tuple(recipes), scale


def _rasterize_recipes(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    recipes: tuple[_ParticleRecipe, ...],
    scale: float,
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    best_score = np.full(shape, -np.inf)
    spacing_array = np.asarray(spacing)
    shape_array = np.asarray(shape)
    for recipe in recipes:
        axes = recipe.semi_axes_m * scale
        extent = np.abs(recipe.rotation) @ (1.85 * axes)
        lower = np.maximum(
            np.floor((recipe.center_m - extent - origin) / spacing_array).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil((recipe.center_m + extent - origin) / spacing_array).astype(int)
            + 1,
            shape_array,
        )
        if np.any(upper <= lower):
            continue
        indices = np.meshgrid(
            *[
                np.arange(lower[axis], upper[axis])
                for axis in range(3)
            ],
            indexing="ij",
        )
        coordinates = np.stack(
            [
                origin[axis] + (indices[axis] + 0.5) * spacing_array[axis]
                for axis in range(3)
            ],
            axis=-1,
        )
        local = (coordinates - recipe.center_m) @ recipe.rotation
        direction = local / np.maximum(axes, 1.0e-30)
        azimuth = np.arctan2(direction[..., 1], direction[..., 0])
        polar = np.arctan2(
            np.sqrt(direction[..., 0] ** 2 + direction[..., 1] ** 2),
            direction[..., 2],
        )
        roughness = recipe.roughness_amplitude * (
            np.sin(
                recipe.roughness_frequency[0] * azimuth
                + recipe.roughness_phase[0]
            )
            * np.sin(
                recipe.roughness_frequency[1] * polar
                + recipe.roughness_phase[1]
            )
            + 0.35
            * np.cos(
                (recipe.roughness_frequency[0] + 1) * polar
                - recipe.roughness_phase[0]
            )
        )
        score = np.full(local.shape[:-1], -np.inf)
        for offset, lobe_scale in zip(
            recipe.lobe_offsets,
            recipe.lobe_scales,
            strict=True,
        ):
            centered = local - offset * axes
            normalized = centered / np.maximum(axes * lobe_scale, 1.0e-30)
            exponent = 2.25 if recipe.electrode == "positive" else 2.6
            radius = np.sum(np.abs(normalized) ** exponent, axis=-1) ** (
                1.0 / exponent
            )
            score = np.maximum(score, 1.0 + roughness - radius)

        for offset, radius_fraction in zip(
            recipe.pore_offsets,
            recipe.pore_radii,
            strict=True,
        ):
            pore_centered = local - offset * axes
            pore_axes = max(float(np.min(axes)) * radius_fraction, 1.0e-30)
            pore = np.sum((pore_centered / pore_axes) ** 2, axis=-1) <= 1.0
            score[pore] = -np.inf
        if recipe.crack_normal is not None and recipe.crack_tangent is not None:
            normal_distance = np.abs(local @ recipe.crack_normal)
            tangent_distance = local @ recipe.crack_tangent
            central_radius = np.sqrt(np.sum(direction**2, axis=-1))
            crack = (
                (normal_distance < recipe.crack_width_fraction * np.min(axes))
                & (central_radius < 0.96)
                & (tangent_distance > -0.05 * np.min(axes))
            )
            score[crack] = -np.inf

        block = tuple(
            slice(int(lower[axis]), int(upper[axis]))
            for axis in range(3)
        )
        current_best = best_score[block]
        update = (score > 0.0) & (score > current_best)
        current_best[update] = score[update]
        labels_block = labels[block]
        labels_block[update] = recipe.particle_id
    return labels


def _keep_largest_particle_components(labels: np.ndarray) -> np.ndarray:
    cleaned = labels.copy()
    for particle_id in sorted(map(int, np.unique(labels))):
        if particle_id == 0:
            continue
        component_labels, count = _label_components(labels == particle_id)
        if count <= 1:
            continue
        component_sizes = np.bincount(component_labels.ravel())
        component_sizes[0] = 0
        keep = int(np.argmax(component_sizes))
        cleaned[(labels == particle_id) & (component_labels != keep)] = 0
    return cleaned


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import label

        structure = np.zeros((3, 3, 3), dtype=np.int8)
        structure[1, 1, 1] = 1
        structure[0, 1, 1] = structure[2, 1, 1] = 1
        structure[1, 0, 1] = structure[1, 2, 1] = 1
        structure[1, 1, 0] = structure[1, 1, 2] = 1
        result, count = label(values, structure=structure)
        return np.asarray(result, dtype=np.int32), int(count)
    except ImportError:
        pass

    labels = np.zeros(values.shape, dtype=np.int32)
    component = 0
    for seed in np.argwhere(values):
        seed_tuple = tuple(map(int, seed))
        if labels[seed_tuple] != 0:
            continue
        component += 1
        labels[seed_tuple] = component
        queue = deque([seed_tuple])
        while queue:
            current = queue.popleft()
            for axis in range(3):
                for step in (-1, 1):
                    neighbor = list(current)
                    neighbor[axis] += step
                    if not 0 <= neighbor[axis] < values.shape[axis]:
                        continue
                    neighbor_tuple = tuple(neighbor)
                    if values[neighbor_tuple] and labels[neighbor_tuple] == 0:
                        labels[neighbor_tuple] = component
                        queue.append(neighbor_tuple)
    return labels, component


def _jittered_centers(
    count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    lengths = upper - lower
    divisions = np.ones(3, dtype=int)
    while int(np.prod(divisions)) < count:
        axis = int(np.argmax(lengths / divisions))
        divisions[axis] += 1
    cell = lengths / divisions
    grid = np.stack(
        np.meshgrid(
            *[np.arange(value) for value in divisions],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    rng.shuffle(grid)
    grid = grid[:count]
    jitter = rng.uniform(-0.20, 0.20, size=grid.shape)
    centers = lower + (grid + 0.5 + jitter) * cell
    return centers, tuple(map(int, divisions))


def _graphite_rotation(rng: np.random.Generator) -> np.ndarray:
    normal = np.asarray((1.0, rng.normal(0.0, 0.18), rng.normal(0.0, 0.18)))
    normal /= np.linalg.norm(normal)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(normal, reference))) > 0.92:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent_a = np.cross(normal, reference)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(normal, tangent_a)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    first = np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
    second = -np.sin(angle) * tangent_a + np.cos(angle) * tangent_b
    return np.column_stack((normal, first, second))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def _public_descriptor(
    recipe: _ParticleRecipe,
    scale: float,
) -> ParticleDescriptor:
    return ParticleDescriptor(
        particle_id=recipe.particle_id,
        electrode=recipe.electrode,
        center_m=tuple(map(float, recipe.center_m)),
        semi_axes_m=tuple(map(float, recipe.semi_axes_m * scale)),
        lobe_count=len(recipe.lobe_offsets),
        roughness_amplitude=recipe.roughness_amplitude,
        internal_pore_count=len(recipe.pore_offsets),
        cracked=recipe.crack_normal is not None,
    )


def _normalize_spacing(
    value: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.repeat(raw, 3)
    if raw.shape != (3,) or np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("voxel_size_m must contain one or three positive values")
    return tuple(map(float, raw))


__all__ = [
    "DEFAULT_PHASE_NAMES",
    "ExplicitMicrostructureTetMesh",
    "ExplicitParticleFullCell",
    "ExplicitParticleFullCellConfig",
    "NEGATIVE_ACTIVE",
    "NEGATIVE_ELECTROLYTE",
    "POSITIVE_ACTIVE",
    "POSITIVE_ELECTROLYTE",
    "ParticleDescriptor",
    "ParticleVoxelStatistics",
    "SEPARATOR_ELECTROLYTE",
    "connected_component_count",
    "cubical_euler_characteristic",
    "generate_explicit_particle_full_cell",
    "phase_percolates",
    "voxel_surface_area",
]
