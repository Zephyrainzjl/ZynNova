"""Stochastic voxel electrode generation with particle/CBD feature parity.

This module is a clean-room ZynNova implementation informed by the public
MCS-CICE ElectrodeGenerationAlgorithm source interface.  It does not import or
vendor that repository.  All public results are expressed as ZynMorph
``MicrostructureVolume`` objects so they can immediately enter characterization
or TetGen/COMSOL workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..volume import MicrostructureVolume
from .io import save_microstructure


@dataclass(frozen=True, slots=True)
class ParticleDistribution:
    geometry: str = "sphere"
    particle_count: int | None = None
    median_diameter_vox: float = 8.0
    lognormal_sigma: float = 0.25
    minimum_diameter_vox: float = 3.0
    maximum_diameter_vox: float = 20.0
    sphere_fraction: float = 0.5
    axis_ratio_ranges: tuple[tuple[float, float], tuple[float, float]] = ((0.65, 1.35), (0.65, 1.35))
    explicit_diameters_vox: tuple[float, ...] = ()
    angle_ranges_degrees: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 360.0), (0.0, 360.0), (0.0, 360.0)
    )
    angle_tolerance_degrees: float | None = None
    base_angles_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.geometry not in {"sphere", "ellipsoid", "mixed"}:
            raise ValueError("geometry must be sphere, ellipsoid, or mixed")
        if self.particle_count is not None and self.particle_count < 1:
            raise ValueError("particle_count must be positive")
        if self.median_diameter_vox <= 0 or self.lognormal_sigma < 0:
            raise ValueError("invalid lognormal PSD parameters")
        if not 0 <= self.sphere_fraction <= 1:
            raise ValueError("sphere_fraction must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class PackingSettings:
    method: str = "rsa"
    boundary_mode: str = "contained"
    overlap_fraction: float = 0.10
    max_attempts_per_particle: int = 2000
    gravity_axis: int = 0
    gravity_direction: int = -1
    extended_padding_vox: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {"rsa", "gravity", "pseudo_gravity", "electrostatic"}:
            raise ValueError("unsupported packing method")
        if self.boundary_mode not in {"contained", "extended", "periodic"}:
            raise ValueError("boundary_mode must be contained, extended, or periodic")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("overlap_fraction must lie in [0,1)")
        if self.gravity_axis not in {0, 1, 2} or self.gravity_direction not in {-1, 1}:
            raise ValueError("invalid gravity configuration")


@dataclass(frozen=True, slots=True)
class CBDSettings:
    method: str = "bridge"
    target_volume_fraction: float = 0.08
    nanoporosity: float = 0.0
    correlation_length_vox: float = 2.5
    bridge_radius_vox: float = 1.5
    interface_decay_vox: float = 3.0
    mixed_bridge_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.method not in {"bridge", "mistry", "random", "blob", "mixed", "none"}:
            raise ValueError("unsupported CBD method")
        if not 0 <= self.target_volume_fraction < 1:
            raise ValueError("target_volume_fraction must lie in [0,1)")
        if not 0 <= self.nanoporosity < 1:
            raise ValueError("nanoporosity must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class ChannelSettings:
    enabled: bool = False
    radius_vox: float = 2.0
    center_zy: tuple[float, float] | None = None
    phase: int = 0


@dataclass(frozen=True, slots=True)
class ElectrodeComposition:
    active_mass_fraction: float = 0.90
    carbon_mass_fraction: float = 0.05
    binder_mass_fraction: float = 0.05
    active_density_g_cm3: float = 2.2
    carbon_density_g_cm3: float = 1.85
    binder_density_g_cm3: float = 1.60
    active_specific_capacity_mAh_g: float | None = None

    def __post_init__(self) -> None:
        fractions = np.asarray(
            [self.active_mass_fraction, self.carbon_mass_fraction, self.binder_mass_fraction],
            dtype=float,
        )
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1.0e-6):
            raise ValueError("composition mass fractions must be non-negative and sum to one")
        if min(self.active_density_g_cm3, self.carbon_density_g_cm3, self.binder_density_g_cm3) <= 0:
            raise ValueError("material densities must be positive")


@dataclass(frozen=True, slots=True)
class ElectrodeSynthesisConfig:
    shape_zyx: tuple[int, int, int] = (64, 64, 64)
    voxel_size_m: float | tuple[float, float, float] = 1.0e-6
    active_volume_fraction: float = 0.65
    active_phase: int = 1
    electrolyte_phase: int = 0
    cbd_phase: int = 2
    seed: int = 0
    particle_distribution: ParticleDistribution = field(default_factory=ParticleDistribution)
    packing: PackingSettings = field(default_factory=PackingSettings)
    cbd: CBDSettings = field(default_factory=CBDSettings)
    channel: ChannelSettings = field(default_factory=ChannelSettings)
    individual_particle_labels: bool = True
    particle_label_offset: int = 1000
    padding_voxels: tuple[int, int, int] = (0, 0, 0)
    crop_after_generation: bool = True
    composition: ElectrodeComposition | None = None

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(int(item) < 4 for item in self.shape_zyx):
            raise ValueError("shape_zyx must contain three values >= 4")
        if not 0 < self.active_volume_fraction < 1:
            raise ValueError("active_volume_fraction must lie in (0,1)")
        if len({self.active_phase, self.electrolyte_phase, self.cbd_phase}) != 3:
            raise ValueError("active/electrolyte/CBD phase IDs must differ")
        if len(self.padding_voxels) != 3 or any(int(item) < 0 for item in self.padding_voxels):
            raise ValueError("padding_voxels must contain three non-negative integers")


@dataclass(frozen=True, slots=True)
class ParticleRecord:
    particle_id: int
    geometry: str
    center_zyx_vox: tuple[float, float, float]
    radii_zyx_vox: tuple[float, float, float]
    rotation_matrix: tuple[tuple[float, float, float], ...]
    nominal_diameter_vox: float
    inserted_voxels: int
    overlap_fraction: float


@dataclass(frozen=True, slots=True)
class PSDValidation:
    sample_count: int
    ks_statistic: float
    p_value: float
    median_diameter_vox: float
    geometric_sigma: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ElectrodeStatistics:
    active_fraction: float
    cbd_geometric_fraction: float
    cbd_solid_fraction: float
    electrolyte_fraction: float
    porosity: float
    effective_porosity: float
    particle_count: int
    mass_loading_mg_cm2: float | None
    areal_capacity_mAh_cm2: float | None


@dataclass(frozen=True, slots=True)
class ElectrodeSynthesisResult:
    volume: MicrostructureVolume
    particle_labels: np.ndarray
    particles: tuple[ParticleRecord, ...]
    psd_validation: PSDValidation
    statistics: ElectrodeStatistics
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def export(self, directory: str | Path, formats: tuple[str, ...] = ("h5", "vtk")) -> Mapping[str, Path]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        for fmt in formats:
            kind = str(fmt).lower().lstrip(".")
            suffix = {"hdf5": ".h5", "h5": ".h5", "vtk": ".vtk", "npy": ".npy", "npz": ".npz"}.get(kind)
            if suffix is None:
                raise ValueError(f"unsupported electrode export format {fmt!r}")
            path = root / f"electrode{suffix}"
            outputs[kind] = save_microstructure(path, self.volume)
        particle_path = root / "particle_labels.npy"
        np.save(particle_path, self.particle_labels, allow_pickle=False)
        outputs["particle_labels"] = particle_path
        return outputs


def _rotation_matrix_xyz(angles_deg: tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = np.deg2rad(np.asarray(angles_deg, dtype=float))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _sample_angles(config: ParticleDistribution, rng: np.random.Generator) -> tuple[float, float, float]:
    if config.angle_tolerance_degrees is not None:
        tol = float(config.angle_tolerance_degrees)
        return tuple(
            float(base + rng.uniform(-tol, tol)) for base in config.base_angles_degrees
        )
    return tuple(float(rng.uniform(low, high)) for low, high in config.angle_ranges_degrees)


def _sample_diameters(config: ParticleDistribution, count: int, rng: np.random.Generator) -> np.ndarray:
    if config.explicit_diameters_vox:
        source = np.asarray(config.explicit_diameters_vox, dtype=float)
        if count <= len(source):
            return source[:count].copy()
        repeats = int(math.ceil(count / len(source)))
        return np.tile(source, repeats)[:count]
    if config.lognormal_sigma == 0:
        values = np.full(count, config.median_diameter_vox, dtype=float)
    else:
        values = rng.lognormal(
            mean=math.log(config.median_diameter_vox),
            sigma=config.lognormal_sigma,
            size=count,
        )
    return np.clip(values, config.minimum_diameter_vox, config.maximum_diameter_vox)


def _particle_geometry(config: ParticleDistribution, diameter: float, rng: np.random.Generator):
    kind = config.geometry
    if kind == "mixed":
        kind = "sphere" if rng.random() < config.sphere_fraction else "ellipsoid"
    if kind == "sphere":
        radii = np.full(3, 0.5 * diameter, dtype=float)
    else:
        r1 = rng.uniform(*config.axis_ratio_ranges[0])
        r2 = rng.uniform(*config.axis_ratio_ranges[1])
        ratios = np.asarray([1.0, r1, r2], dtype=float)
        # Preserve nominal ellipsoid volume of a sphere with the sampled diameter.
        ratios /= np.cbrt(np.prod(ratios))
        radii = 0.5 * diameter * ratios
    rotation = _rotation_matrix_xyz(_sample_angles(config, rng))
    return kind, radii, rotation


def _ellipsoid_indices(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    shape: tuple[int, int, int],
    *,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extent = int(math.ceil(float(np.max(radii)))) + 1
    ranges = [
        np.arange(math.floor(center[d] - extent), math.ceil(center[d] + extent) + 1, dtype=int)
        for d in range(3)
    ]
    zz, yy, xx = np.meshgrid(*ranges, indexing="ij")
    points = np.stack((zz, yy, xx), axis=-1).reshape(-1, 3).astype(float)
    local = (points - center) @ rotation
    inside = np.sum((local / radii) ** 2, axis=1) <= 1.0
    indices = points[inside].astype(np.int64)
    if periodic:
        for axis, size in enumerate(shape):
            indices[:, axis] %= size
    else:
        valid = np.ones(len(indices), dtype=bool)
        for axis, size in enumerate(shape):
            valid &= (indices[:, axis] >= 0) & (indices[:, axis] < size)
        indices = indices[valid]
    if not len(indices):
        return (np.empty(0, dtype=np.int64),) * 3
    indices = np.unique(indices, axis=0)
    return indices[:, 0], indices[:, 1], indices[:, 2]


def _candidate_center(
    shape: tuple[int, int, int], radii: np.ndarray, rng: np.random.Generator, mode: str
) -> np.ndarray:
    if mode == "contained":
        low = radii + 0.5
        high = np.asarray(shape, dtype=float) - radii - 0.5
        if np.any(high <= low):
            raise ValueError("particle is larger than contained generation domain")
        return rng.uniform(low, high)
    if mode == "periodic":
        return rng.uniform(np.zeros(3), np.asarray(shape, dtype=float))
    # Extended permits particle centers outside the final crop.
    return rng.uniform(-radii, np.asarray(shape, dtype=float) + radii)


def _overlap(occupied: np.ndarray, indices) -> float:
    if len(indices[0]) == 0:
        return 1.0
    return float(np.mean(occupied[indices]))


def _gravity_center(
    occupied: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    rng: np.random.Generator,
    config: PackingSettings,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float] | None:
    axis = config.gravity_axis
    shape = occupied.shape
    center = _candidate_center(shape, radii, rng, "periodic" if config.boundary_mode == "periodic" else "contained")
    if config.gravity_direction < 0:
        positions = range(int(shape[axis] - math.ceil(radii[axis]) - 1), int(math.ceil(radii[axis])), -1)
    else:
        positions = range(int(math.ceil(radii[axis])), int(shape[axis] - math.ceil(radii[axis]) - 1))
    best = None
    for value in positions:
        center[axis] = float(value)
        idx = _ellipsoid_indices(center, radii, rotation, shape, periodic=config.boundary_mode == "periodic")
        overlap = _overlap(occupied, idx)
        if overlap <= config.overlap_fraction:
            best = (center.copy(), idx, overlap)
        elif best is not None:
            break
    return best


def _electrostatic_center(
    occupied: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    config: PackingSettings,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~occupied)
    margin = int(math.ceil(np.max(radii))) + 1
    valid = np.ones_like(occupied, dtype=bool)
    if config.boundary_mode == "contained":
        valid[:margin] = valid[-margin:] = False
        valid[:, :margin] = valid[:, -margin:] = False
        valid[:, :, :margin] = valid[:, :, -margin:] = False
    scores = np.where(valid, distance, -np.inf)
    finite = np.flatnonzero(np.isfinite(scores.ravel()))
    if not len(finite):
        return _candidate_center(occupied.shape, radii, rng, config.boundary_mode)
    max_score = np.max(scores)
    candidates = np.flatnonzero(scores.ravel() >= max_score * 0.95)
    choice = int(rng.choice(candidates))
    return np.asarray(np.unravel_index(choice, occupied.shape), dtype=float)


def _estimate_count(shape, target_fraction, distribution: ParticleDistribution) -> int:
    if distribution.particle_count is not None:
        return int(distribution.particle_count)
    volume = float(np.prod(shape) * target_fraction)
    radius = 0.5 * distribution.median_diameter_vox
    particle_volume = 4.0 / 3.0 * math.pi * radius**3
    return max(1, int(math.ceil(volume / max(particle_volume, 1.0))))


def validate_particle_size_distribution(
    diameters_vox: np.ndarray,
    target: ParticleDistribution,
    *,
    significance: float = 0.01,
) -> PSDValidation:
    values = np.asarray(diameters_vox, dtype=float)
    if not len(values):
        return PSDValidation(0, math.nan, math.nan, math.nan, math.nan, False)
    from scipy.stats import kstest, lognorm

    sigma = max(float(target.lognormal_sigma), 1.0e-12)
    distribution = lognorm(s=sigma, scale=target.median_diameter_vox)
    statistic, p_value = kstest(values, distribution.cdf)
    positive = values[values > 0]
    geometric_sigma = float(np.exp(np.std(np.log(positive)))) if len(positive) > 1 else 1.0
    return PSDValidation(
        sample_count=int(len(values)),
        ks_statistic=float(statistic),
        p_value=float(p_value),
        median_diameter_vox=float(np.median(values)),
        geometric_sigma=geometric_sigma,
        passed=bool(p_value >= significance),
    )


def _select_exact(mask: np.ndarray, score: np.ndarray, count: int, *, largest: bool = True) -> np.ndarray:
    available = np.flatnonzero(mask.ravel())
    count = min(max(0, int(count)), len(available))
    result = np.zeros(mask.size, dtype=bool)
    if count == 0:
        return result.reshape(mask.shape)
    values = score.ravel()[available]
    if largest:
        chosen = np.argpartition(values, len(values) - count)[-count:]
    else:
        chosen = np.argpartition(values, count - 1)[:count]
    result[available[chosen]] = True
    return result.reshape(mask.shape)


def _bridge_score(particle_labels: np.ndarray, active: np.ndarray, settings: CBDSettings) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    pore = ~active
    distance = distance_transform_edt(pore)
    # High values near AM surfaces; correlated smoothing naturally fills necks
    # between nearby particles and mimics a bridge-rich carbon/binder network.
    score = np.exp(-distance / max(settings.bridge_radius_vox, 1.0e-3))
    owners = particle_labels.astype(np.float64)
    owner_gradient = sum(np.abs(owners - np.roll(owners, 1, axis=axis)) for axis in range(3))
    score += gaussian_filter((owner_gradient > 0).astype(float), settings.bridge_radius_vox, mode="wrap")
    return np.where(pore, score, -np.inf)


def _cbd_mask(active, particle_labels, settings: CBDSettings, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if settings.method == "none" or settings.target_volume_fraction <= 0:
        return np.zeros_like(active, dtype=bool)
    pore = ~active
    # Geometrical CBD volume is increased so its solid fraction after unresolved
    # nanoporosity equals the requested macroscopic CBD solid volume.
    geometrical_fraction = settings.target_volume_fraction / max(1.0 - settings.nanoporosity, 1.0e-12)
    target = int(round(geometrical_fraction * active.size))
    target = min(target, int(np.count_nonzero(pore)))
    if settings.method == "random":
        score = rng.random(active.shape)
    elif settings.method == "blob":
        score = gaussian_filter(
            rng.normal(size=active.shape),
            sigma=max(settings.correlation_length_vox, 0.5),
            mode="wrap",
        )
    elif settings.method == "mistry":
        distance = distance_transform_edt(pore)
        # Interface-weighted stochastic growth reproduces Mistry-like deposition
        # without requiring a modified PoresPy fork.
        score = np.exp(-distance / max(settings.interface_decay_vox, 1.0e-3))
        score *= 0.75 + 0.25 * gaussian_filter(rng.random(active.shape), 1.0, mode="wrap")
    elif settings.method == "bridge":
        score = _bridge_score(particle_labels, active, settings)
    elif settings.method == "mixed":
        bridge = _bridge_score(particle_labels, active, settings)
        blob = gaussian_filter(
            rng.normal(size=active.shape),
            sigma=max(settings.correlation_length_vox, 0.5),
            mode="wrap",
        )
        bridge = np.nan_to_num(bridge, nan=0.0, neginf=0.0)
        bridge = (bridge - bridge[pore].min()) / max(np.ptp(bridge[pore]), 1.0e-12)
        blob = (blob - blob[pore].min()) / max(np.ptp(blob[pore]), 1.0e-12)
        score = settings.mixed_bridge_fraction * bridge + (1.0 - settings.mixed_bridge_fraction) * blob
    else:
        raise AssertionError(settings.method)
    score = np.where(pore, score, -np.inf)
    return _select_exact(pore, score, target, largest=True)


def _carve_channel(labels: np.ndarray, channel: ChannelSettings) -> None:
    if not channel.enabled:
        return
    nz, ny, _ = labels.shape
    center_z, center_y = channel.center_zy or ((nz - 1) / 2.0, (ny - 1) / 2.0)
    z, y = np.indices((nz, ny), dtype=float)
    circle = (z - center_z) ** 2 + (y - center_y) ** 2 <= channel.radius_vox**2
    labels[circle, :] = int(channel.phase)


def electrode_volume_targets_from_composition(
    composition: ElectrodeComposition,
    *,
    total_porosity: float,
    cbd_nanoporosity: float = 0.0,
) -> dict[str, float]:
    """Convert mass fractions/densities to AM/CBD geometric volume targets."""

    if not 0 <= total_porosity < 1:
        raise ValueError("total_porosity must lie in [0,1)")
    masses = np.asarray(
        [composition.active_mass_fraction, composition.carbon_mass_fraction, composition.binder_mass_fraction],
        dtype=float,
    )
    densities = np.asarray(
        [composition.active_density_g_cm3, composition.carbon_density_g_cm3, composition.binder_density_g_cm3],
        dtype=float,
    )
    solid_volumes = masses / densities
    solid_volumes /= solid_volumes.sum()
    solid_fraction = 1.0 - total_porosity
    active = solid_fraction * solid_volumes[0]
    cbd_solid = solid_fraction * (solid_volumes[1] + solid_volumes[2])
    cbd_geometric = cbd_solid / max(1.0 - cbd_nanoporosity, 1.0e-12)
    return {
        "active_volume_fraction": float(active),
        "cbd_solid_fraction": float(cbd_solid),
        "cbd_geometric_fraction": float(cbd_geometric),
        "electrolyte_fraction": float(max(0.0, 1.0 - active - cbd_geometric)),
    }


def _statistics(labels, config: ElectrodeSynthesisConfig, particle_count: int) -> ElectrodeStatistics:
    active = float(np.mean(labels == config.active_phase))
    cbd_geo = float(np.mean(labels == config.cbd_phase))
    electrolyte = float(np.mean(labels == config.electrolyte_phase))
    cbd_solid = cbd_geo * (1.0 - config.cbd.nanoporosity)
    effective_porosity = electrolyte + cbd_geo * config.cbd.nanoporosity
    mass_loading = None
    capacity = None
    if config.composition is not None:
        spacing = config.voxel_size_m
        if np.isscalar(spacing):
            dz = dy = dx = float(spacing)
        else:
            dz, dy, dx = map(float, spacing)
        thickness_cm = labels.shape[2] * dx * 100.0
        # Average bulk AM volume fraction × density × thickness.
        active_mass_g_cm2 = active * config.composition.active_density_g_cm3 * thickness_cm
        mass_loading = active_mass_g_cm2 * 1000.0
        if config.composition.active_specific_capacity_mAh_g is not None:
            capacity = active_mass_g_cm2 * config.composition.active_specific_capacity_mAh_g
    return ElectrodeStatistics(
        active_fraction=active,
        cbd_geometric_fraction=cbd_geo,
        cbd_solid_fraction=cbd_solid,
        electrolyte_fraction=electrolyte,
        porosity=1.0 - active - cbd_solid,
        effective_porosity=effective_porosity,
        particle_count=int(particle_count),
        mass_loading_mg_cm2=mass_loading,
        areal_capacity_mAh_cm2=capacity,
    )


def crop_structure_to_content(
    array: np.ndarray,
    *,
    background: int = 0,
    margins_vox: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Crop empty outer slabs while preserving optional margins.

    This is the safe ZynNova counterpart of the electrode cropping/padding
    helpers exposed by the audited particle-generator source.  It works for
    arbitrary integer phase labels and never serializes through pickle.
    """

    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError("array must be three-dimensional")
    margins = tuple(map(int, margins_vox))
    if len(margins) != 3 or any(value < 0 for value in margins):
        raise ValueError("margins_vox must contain three non-negative integers")
    occupied = values != int(background)
    if not np.any(occupied):
        return values.copy()
    coordinates = np.argwhere(occupied)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    lower = np.maximum(0, lower - np.asarray(margins))
    upper = np.minimum(np.asarray(values.shape), upper + np.asarray(margins))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    return values[slices].copy()


def pad_structure_to_content(
    array: np.ndarray,
    *,
    background: int = 0,
    margins_vox: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Tight-crop content and then add deterministic symmetric padding."""

    margins = tuple(map(int, margins_vox))
    cropped = crop_structure_to_content(array, background=background)
    return np.pad(
        cropped,
        tuple((value, value) for value in margins),
        mode="constant",
        constant_values=int(background),
    )


def cut_electrode_empty_tail(
    array: np.ndarray,
    *,
    axis: int = 0,
    background: int = 0,
    margin_vox: int = 0,
) -> np.ndarray:
    """Remove trailing empty slabs after the last occupied cross-section."""

    values = np.asarray(array)
    if values.ndim != 3 or axis not in {0, 1, 2}:
        raise ValueError("array must be 3-D and axis must be 0, 1, or 2")
    if margin_vox < 0:
        raise ValueError("margin_vox must be non-negative")
    reduce_axes = tuple(i for i in range(3) if i != axis)
    occupied_sections = np.any(values != int(background), axis=reduce_axes)
    indices = np.flatnonzero(occupied_sections)
    if indices.size == 0:
        return values.copy()
    stop = min(values.shape[axis], int(indices[-1]) + 1 + int(margin_vox))
    slices = [slice(None)] * 3
    slices[axis] = slice(0, stop)
    return values[tuple(slices)].copy()


def generate_particle_electrode(config: ElectrodeSynthesisConfig) -> ElectrodeSynthesisResult:
    """Generate one stochastic electrode with particle/CBD/PSD controls.

    ``padding_voxels`` creates a guard volume for particle packing.  When
    ``crop_after_generation=True`` the guard volume is removed *before* the
    exact active-fraction correction and CBD generation, so the returned target
    box satisfies the requested composition rather than the temporary padded
    box.
    """

    rng = np.random.default_rng(config.seed)
    requested_shape = tuple(map(int, config.shape_zyx))
    padding = tuple(map(int, config.padding_voxels))
    use_padding = any(padding)
    packing_shape = tuple(
        size + 2 * pad for size, pad in zip(requested_shape, padding, strict=True)
    ) if use_padding else requested_shape

    occupied = np.zeros(packing_shape, dtype=bool)
    particle_labels = np.full(packing_shape, -1, dtype=np.int32)
    packing_target_voxels = int(round(config.active_volume_fraction * np.prod(packing_shape)))
    nominal_count = _estimate_count(
        packing_shape, config.active_volume_fraction, config.particle_distribution
    )
    candidate_count = max(nominal_count, config.particle_distribution.particle_count or 0)
    diameters = list(_sample_diameters(config.particle_distribution, candidate_count, rng))
    records: list[ParticleRecord] = []
    accepted_diameters: list[float] = []
    particle_id = 0
    diameter_cursor = 0
    max_total_candidates = (
        candidate_count
        if config.particle_distribution.particle_count is not None
        else max(candidate_count * 6, candidate_count + 64)
    )

    while np.count_nonzero(occupied) < packing_target_voxels and particle_id < max_total_candidates:
        if diameter_cursor >= len(diameters):
            diameters.extend(
                _sample_diameters(
                    config.particle_distribution, max(16, candidate_count // 2), rng
                ).tolist()
            )
        diameter = float(diameters[diameter_cursor])
        diameter_cursor += 1
        kind, radii, rotation = _particle_geometry(config.particle_distribution, diameter, rng)
        accepted = None
        for _attempt in range(config.packing.max_attempts_per_particle):
            if config.packing.method in {"gravity", "pseudo_gravity"}:
                accepted = _gravity_center(occupied, radii, rotation, rng, config.packing)
                if accepted is not None:
                    break
                continue
            if config.packing.method == "electrostatic":
                center = _electrostatic_center(occupied, radii, rng, config.packing)
            else:
                center = _candidate_center(
                    packing_shape, radii, rng, config.packing.boundary_mode
                )
            idx = _ellipsoid_indices(
                center,
                radii,
                rotation,
                packing_shape,
                periodic=config.packing.boundary_mode == "periodic",
            )
            overlap = _overlap(occupied, idx)
            if overlap <= config.packing.overlap_fraction:
                accepted = (center, idx, overlap)
                break
        if accepted is None:
            particle_id += 1
            continue
        center, idx, overlap = accepted
        if len(idx[0]) == 0:
            particle_id += 1
            continue
        before = int(np.count_nonzero(occupied[idx]))
        occupied[idx] = True
        label_value = (
            config.particle_label_offset + particle_id
            if config.individual_particle_labels
            else config.active_phase
        )
        particle_labels[idx] = label_value
        inserted = int(len(idx[0]) - before)
        if inserted > 0:
            records.append(
                ParticleRecord(
                    particle_id=particle_id,
                    geometry=kind,
                    center_zyx_vox=tuple(map(float, center)),
                    radii_zyx_vox=tuple(map(float, radii)),
                    rotation_matrix=tuple(tuple(map(float, row)) for row in rotation),
                    nominal_diameter_vox=diameter,
                    inserted_voxels=inserted,
                    overlap_fraction=float(overlap),
                )
            )
            accepted_diameters.append(diameter)
        particle_id += 1

    # Remove the guard volume.  Particle centers are shifted back into the
    # coordinates of the returned target volume.
    if use_padding and config.crop_after_generation:
        slices = tuple(
            slice(pad, pad + size)
            for pad, size in zip(padding, requested_shape, strict=True)
        )
        occupied = occupied[slices].copy()
        particle_labels = particle_labels[slices].copy()
        shifted_records = []
        shift = np.asarray(padding, dtype=float)
        for record in records:
            center = tuple(
                map(float, np.asarray(record.center_zyx_vox, dtype=float) - shift)
            )
            label_value = (
                config.particle_label_offset + record.particle_id
                if config.individual_particle_labels
                else config.active_phase
            )
            surviving = int(np.count_nonzero(particle_labels == label_value))
            if surviving == 0:
                continue
            shifted_records.append(
                ParticleRecord(
                    particle_id=record.particle_id,
                    geometry=record.geometry,
                    center_zyx_vox=center,
                    radii_zyx_vox=record.radii_zyx_vox,
                    rotation_matrix=record.rotation_matrix,
                    nominal_diameter_vox=record.nominal_diameter_vox,
                    inserted_voxels=surviving,
                    overlap_fraction=record.overlap_fraction,
                )
            )
        records = shifted_records
        shape = requested_shape
    else:
        shape = packing_shape

    target_active_voxels = int(round(config.active_volume_fraction * np.prod(shape)))
    active_count = int(np.count_nonzero(occupied))
    if active_count > target_active_voxels:
        from scipy.ndimage import distance_transform_edt

        distance_inside = distance_transform_edt(occupied)
        keep = _select_exact(occupied, distance_inside, target_active_voxels, largest=True)
        occupied = keep
        particle_labels[~occupied] = -1
    elif active_count < target_active_voxels:
        from scipy.ndimage import distance_transform_edt

        pore = ~occupied
        score = -distance_transform_edt(pore)
        add = _select_exact(pore, score, target_active_voxels - active_count, largest=True)
        if np.any(occupied):
            _, nearest = distance_transform_edt(particle_labels < 0, return_indices=True)
            nearest_labels = particle_labels[tuple(nearest)]
            particle_labels[add] = nearest_labels[add]
        else:
            particle_labels[add] = config.particle_label_offset
        occupied |= add

    # Recompute record voxel counts after exact final-box correction.
    final_records = []
    for record in records:
        label_value = (
            config.particle_label_offset + record.particle_id
            if config.individual_particle_labels
            else config.active_phase
        )
        surviving = int(np.count_nonzero(particle_labels == label_value))
        if surviving:
            final_records.append(
                ParticleRecord(
                    particle_id=record.particle_id,
                    geometry=record.geometry,
                    center_zyx_vox=record.center_zyx_vox,
                    radii_zyx_vox=record.radii_zyx_vox,
                    rotation_matrix=record.rotation_matrix,
                    nominal_diameter_vox=record.nominal_diameter_vox,
                    inserted_voxels=surviving,
                    overlap_fraction=record.overlap_fraction,
                )
            )
    records = final_records

    cbd = _cbd_mask(occupied, particle_labels, config.cbd, rng)
    labels = np.full(shape, config.electrolyte_phase, dtype=np.int32)
    labels[occupied] = config.active_phase
    labels[cbd] = config.cbd_phase
    _carve_channel(labels, config.channel)

    phase_names = {
        config.electrolyte_phase: "electrolyte",
        config.active_phase: "active_material",
        config.cbd_phase: "cbd",
    }
    volume = MicrostructureVolume(
        labels=labels,
        voxel_size_m=config.voxel_size_m,
        phase_names=phase_names,
        metadata={
            "generator": "zynnova-zynmorph-particle-electrode",
            "packing_method": config.packing.method,
            "boundary_mode": config.packing.boundary_mode,
            "cbd_method": config.cbd.method,
            "individual_particle_labels": config.individual_particle_labels,
            "padding_voxels": padding,
            "crop_after_generation": bool(config.crop_after_generation),
            "packing_shape_zyx": packing_shape,
        },
    )
    psd = validate_particle_size_distribution(
        np.asarray([record.nominal_diameter_vox for record in records], dtype=float),
        config.particle_distribution,
    )
    stats = _statistics(labels, config, len(records))
    return ElectrodeSynthesisResult(
        volume=volume,
        particle_labels=particle_labels,
        particles=tuple(records),
        psd_validation=psd,
        statistics=stats,
        metadata={
            "target_active_fraction": config.active_volume_fraction,
            "target_active_voxels": target_active_voxels,
            "accepted_particle_count": len(records),
            "packing_target_active_voxels": packing_target_voxels,
        },
    )


# Explicit aliases for the three packing families exposed by the upstream
# public source interface.  They route through one validated ZynNova engine.
def generate_rsa_electrode(config: ElectrodeSynthesisConfig) -> ElectrodeSynthesisResult:
    from dataclasses import replace

    return generate_particle_electrode(replace(config, packing=replace(config.packing, method="rsa")))


def generate_gravity_electrode(config: ElectrodeSynthesisConfig) -> ElectrodeSynthesisResult:
    from dataclasses import replace

    return generate_particle_electrode(replace(config, packing=replace(config.packing, method="gravity")))


def generate_electrostatic_electrode(config: ElectrodeSynthesisConfig) -> ElectrodeSynthesisResult:
    from dataclasses import replace

    return generate_particle_electrode(replace(config, packing=replace(config.packing, method="electrostatic")))


__all__ = [
    "CBDSettings",
    "ChannelSettings",
    "ElectrodeComposition",
    "ElectrodeStatistics",
    "ElectrodeSynthesisConfig",
    "ElectrodeSynthesisResult",
    "PackingSettings",
    "PSDValidation",
    "ParticleDistribution",
    "ParticleRecord",
    "crop_structure_to_content",
    "cut_electrode_empty_tail",
    "electrode_volume_targets_from_composition",
    "generate_electrostatic_electrode",
    "generate_gravity_electrode",
    "generate_particle_electrode",
    "generate_rsa_electrode",
    "pad_structure_to_content",
    "validate_particle_size_distribution",
]
