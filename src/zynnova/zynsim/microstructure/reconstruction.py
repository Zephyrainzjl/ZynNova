"""Fast 2-D-to-3-D electrode reconstruction from one or three orthogonal views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .imaging import ImageSegmentationConfig, PhaseImageReport, segment_electrode_image
from .morphology import ZynMorphDescriptor, characterize_morphology, retarget_phase_fractions

try:  # optional native acceleration
    from zynnova._native import _zynsim_voxel_native as _native
except Exception:  # pragma: no cover
    _native = None


@dataclass(frozen=True, slots=True)
class OrthogonalImages:
    """One or three segmented/segmentable views.

    Arrays use logical axis order: ``xy[x,y]``, ``xz[x,z]``, and ``yz[y,z]``.
    A single image can be supplied through any one field; its plane determines
    the extrusion direction.
    """

    xy: np.ndarray | str | Path | None = None
    xz: np.ndarray | str | Path | None = None
    yz: np.ndarray | str | Path | None = None

    def count(self) -> int:
        return sum(value is not None for value in (self.xy, self.xz, self.yz))


@dataclass(frozen=True, slots=True)
class ImageToVoxelConfig:
    target_shape: tuple[int, int, int] | None = None
    correlation_length_voxels: tuple[float, float, float] = (8.0, 8.0, 8.0)
    stochastic_strength: float = 0.18
    projection_weight: float = 4.0
    morphology_weight: float = 1.0
    projection_relaxation_iterations: int = 4
    target_phase_fractions: Mapping[int, float] | None = None
    random_seed: int = 0
    chunk_size_x: int = 32
    output_memmap: str | Path | None = None
    characterize: bool = True
    native_volume_limit: int = 128_000_000
    fraction_retarget_method: Literal["auto", "distance", "streaming"] = "auto"
    maximum_distance_retarget_voxels: int = 16_777_216
    maximum_characterization_voxels: int = 16_777_216

    def __post_init__(self) -> None:
        if self.target_shape is not None and (len(self.target_shape) != 3 or min(self.target_shape) < 1):
            raise ValueError("target_shape must contain three positive integers")
        if len(self.correlation_length_voxels) != 3 or min(self.correlation_length_voxels) <= 0.0:
            raise ValueError("correlation lengths must be positive")
        if not 0.0 <= self.stochastic_strength <= 2.0:
            raise ValueError("stochastic_strength must lie in [0,2]")
        if self.projection_weight <= 0.0 or self.morphology_weight < 0.0:
            raise ValueError("projection/morphology weights are invalid")
        if self.projection_relaxation_iterations < 0 or self.chunk_size_x < 1:
            raise ValueError("iteration and chunk counts must be non-negative/positive")
        if self.native_volume_limit < 0:
            raise ValueError("native_volume_limit cannot be negative")
        if self.fraction_retarget_method not in {"auto", "distance", "streaming"}:
            raise ValueError("fraction_retarget_method is invalid")
        if self.maximum_distance_retarget_voxels < 1 or self.maximum_characterization_voxels < 1:
            raise ValueError("retarget/characterization limits must be positive")


@dataclass(frozen=True, slots=True)
class ImageToVoxelResult:
    phase_labels: np.ndarray
    segmented_views: Mapping[str, PhaseImageReport]
    phase_volume_fractions: Mapping[int, float]
    descriptor: ZynMorphDescriptor | None
    projection_agreement: Mapping[str, float]
    reconstruction_method: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_labels", np.asanyarray(self.phase_labels, dtype=np.int32))
        object.__setattr__(self, "segmented_views", dict(self.segmented_views))
        object.__setattr__(self, "phase_volume_fractions", dict(self.phase_volume_fractions))
        object.__setattr__(self, "projection_agreement", dict(self.projection_agreement))
        object.__setattr__(self, "metadata", dict(self.metadata))


def reconstruct_electrode_volume(
    images: OrthogonalImages | np.ndarray | str | Path,
    *,
    segmentation: ImageSegmentationConfig | Mapping[str, ImageSegmentationConfig],
    config: ImageToVoxelConfig | None = None,
) -> ImageToVoxelResult:
    """Reconstruct a multi-phase 3-D voxel field from one or three images.

    Three-view reconstruction maximizes a projection-consistency energy plus a
    correlated morphology prior in x-chunks, so memory scales with one chunk.
    One-view reconstruction uses projection-preserving correlated extrusion.
    """

    resolved = config or ImageToVoxelConfig()
    views = images if isinstance(images, OrthogonalImages) else OrthogonalImages(xy=images)
    if views.count() not in (1, 3):
        raise ValueError("provide exactly one image or all three xy/xz/yz images")
    reports: dict[str, PhaseImageReport] = {}
    for name in ("xy", "xz", "yz"):
        image = getattr(views, name)
        if image is None:
            continue
        seg = segmentation[name] if isinstance(segmentation, Mapping) else segmentation
        reports[name] = segment_electrode_image(image, seg)
    shape = _resolve_shape(reports, resolved.target_shape)
    resized = {name: _resize2(report.labels, _plane_shape(name, shape)) for name, report in reports.items()}
    rng = np.random.default_rng(resolved.random_seed)
    output = _allocate_output(shape, resolved.output_memmap)
    if len(resized) == 1:
        name, plane = next(iter(resized.items()))
        _single_view_extrusion(output, name, plane, resolved, rng)
        method = "projection-preserving-correlated-extrusion"
    else:
        if (
            _native is not None
            and output.size <= resolved.native_volume_limit
            and resolved.output_memmap is None
        ):
            xy, xz, yz = resized["xy"], resized["xz"], resized["yz"]
            phases = sorted(set(map(int, np.unique(xy))) | set(map(int, np.unique(xz))) | set(map(int, np.unique(yz))))
            priors = [
                (float(np.mean(xy == p)) + float(np.mean(xz == p)) + float(np.mean(yz == p))) / 3.0
                for p in phases
            ]
            output[...] = _native.fuse_orthogonal_labels(
                xy, xz, yz, phases,
                [float(np.log(max(value, 1.0e-12)) * resolved.morphology_weight) for value in priors],
                float(resolved.projection_weight),
            )
            if resolved.stochastic_strength > 0.0:
                _correlated_interface_perturbation(output, resolved, rng)
            for _ in range(resolved.projection_relaxation_iterations):
                _relax_projections(output, resized, rng)
            method = "native-three-view-projection-fusion"
        else:
            _three_view_fusion(output, resized, resolved, rng)
            method = "three-view-projection-spectral-fusion"
    if resolved.target_phase_fractions:
        retarget_phase_fractions(
            output,
            resolved.target_phase_fractions,
            preserve_topology_weight=max(resolved.morphology_weight, 1e-6),
            method=resolved.fraction_retarget_method,
            maximum_distance_voxels=resolved.maximum_distance_retarget_voxels,
            chunk_size_x=resolved.chunk_size_x,
            in_place=True,
            random_seed=resolved.random_seed,
        )
    counts = _phase_counts_chunked(output, chunk_size_x=resolved.chunk_size_x)
    fractions = {int(label): float(count / output.size) for label, count in counts.items()}
    agreement = _projection_agreement(output, resized)
    descriptor = (
        characterize_morphology(
            output,
            maximum_analysis_voxels=resolved.maximum_characterization_voxels,
        )
        if resolved.characterize
        else None
    )
    if isinstance(output, np.memmap):
        output.flush()
    return ImageToVoxelResult(
        phase_labels=output,
        segmented_views=reports,
        phase_volume_fractions=fractions,
        descriptor=descriptor,
        projection_agreement=agreement,
        reconstruction_method=method,
        metadata={
            "target_shape": shape,
            "native_backend": bool(_native is not None),
            "random_seed": resolved.random_seed,
        },
    )



def _phase_counts_chunked(volume: np.ndarray, *, chunk_size_x: int) -> dict[int, int]:
    """Count labels without sorting or copying a complete memmapped volume."""

    array = np.asanyarray(volume)
    if (
        _native is not None
        and hasattr(_native, "phase_counts")
        and array.dtype == np.int32
        and array.flags.c_contiguous
    ):
        return {int(key): int(value) for key, value in _native.phase_counts(array).items()}
    counts: dict[int, int] = {}
    for start in range(0, int(array.shape[0]), max(1, int(chunk_size_x))):
        slab = np.asarray(array[start : start + max(1, int(chunk_size_x))])
        unique, local = np.unique(slab, return_counts=True)
        for label, count in zip(unique, local, strict=True):
            key = int(label)
            counts[key] = counts.get(key, 0) + int(count)
    return dict(sorted(counts.items()))

def _resolve_shape(reports: Mapping[str, PhaseImageReport], target: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if target is not None:
        return tuple(map(int, target))
    if len(reports) == 1:
        name, report = next(iter(reports.items()))
        a, b = report.shape
        depth = max(16, int(round(np.sqrt(a * b))))
        return {"xy": (a, b, depth), "xz": (a, depth, b), "yz": (depth, a, b)}[name]
    xy, xz, yz = reports["xy"].shape, reports["xz"].shape, reports["yz"].shape
    x = int(round((xy[0] + xz[0]) / 2))
    y = int(round((xy[1] + yz[0]) / 2))
    z = int(round((xz[1] + yz[1]) / 2))
    return x, y, z


def _plane_shape(name: str, shape: tuple[int, int, int]) -> tuple[int, int]:
    x, y, z = shape
    return {"xy": (x, y), "xz": (x, z), "yz": (y, z)}[name]


def _resize2(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    indices = [np.minimum(np.rint(np.linspace(0, values.shape[i] - 1, shape[i])).astype(int), values.shape[i] - 1) for i in range(2)]
    return np.ascontiguousarray(values[np.ix_(indices[0], indices[1])], dtype=np.int32)


def _allocate_output(shape: tuple[int, int, int], path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.empty(shape, dtype=np.int32)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(target, mode="w+", dtype=np.int32, shape=shape)


def _single_view_extrusion(
    output: np.ndarray,
    name: str,
    plane: np.ndarray,
    config: ImageToVoxelConfig,
    rng: np.random.Generator,
) -> None:
    shape = output.shape
    extrusion_axis = {"xy": 2, "xz": 1, "yz": 0}[name]
    depth = shape[extrusion_axis]
    # Smooth random walks deform the measured morphology while retaining every
    # phase and the exact measured plane statistics in expectation.
    walk = np.zeros((depth, 2), dtype=float)
    increments = rng.normal(scale=config.stochastic_strength, size=(depth, 2))
    for index in range(1, depth):
        walk[index] = 0.92 * walk[index - 1] + increments[index]
    for index in range(depth):
        shifted = np.roll(plane, tuple(np.rint(walk[index]).astype(int)), axis=(0, 1))
        if extrusion_axis == 2:
            output[:, :, index] = shifted
        elif extrusion_axis == 1:
            output[:, index, :] = shifted
        else:
            output[index, :, :] = shifted
    if config.stochastic_strength > 0.0:
        _correlated_interface_perturbation(output, config, rng)


def _three_view_fusion(
    output: np.ndarray,
    views: Mapping[str, np.ndarray],
    config: ImageToVoxelConfig,
    rng: np.random.Generator,
) -> None:
    xy, xz, yz = views["xy"], views["xz"], views["yz"]
    phases = np.asarray(
        sorted(
            set(map(int, np.unique(xy)))
            | set(map(int, np.unique(xz)))
            | set(map(int, np.unique(yz)))
        ),
        dtype=np.int32,
    )
    priors = np.asarray(
        [(np.mean(xy == p) + np.mean(xz == p) + np.mean(yz == p)) / 3.0 for p in phases]
    )
    prior_log = np.log(np.maximum(priors, 1e-12)) * config.morphology_weight
    noise_model = _SeparableCorrelatedNoise(
        output.shape, config.correlation_length_voxels, rng
    )
    # Only one scalar score field and one label field are resident per slab;
    # memory is O(chunk_x * ny * nz), independent of phase count and nx.
    for start in range(0, output.shape[0], config.chunk_size_x):
        stop = min(start + config.chunk_size_x, output.shape[0])
        xy_chunk = xy[start:stop, :, None]
        xz_chunk = xz[start:stop, None, :]
        noise = noise_model.chunk(start, stop)
        best_score = np.full((stop - start, output.shape[1], output.shape[2]), -np.inf, dtype=np.float32)
        best_phase = np.full(best_score.shape, phases[0], dtype=np.int32)
        for index, phase in enumerate(phases):
            score = (
                config.projection_weight
                * (
                    (xy_chunk == phase).astype(np.float32)
                    + (xz_chunk == phase).astype(np.float32)
                    + (yz[None, :, :] == phase).astype(np.float32)
                )
                + np.float32(prior_log[index])
                + np.float32(config.stochastic_strength) * noise
            )
            replace = score > best_score
            best_score[replace] = score[replace]
            best_phase[replace] = phase
        output[start:stop] = best_phase
    for _ in range(config.projection_relaxation_iterations):
        _relax_projections(output, views, rng)


class _SeparableCorrelatedNoise:
    """Low-rank correlated random field with O(nx+ny+nz) storage."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        correlation: tuple[float, float, float],
        rng: np.random.Generator,
    ) -> None:
        self.shape = tuple(map(int, shape))
        vectors: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:  # pragma: no cover - SciPy is in zynsim dependencies
            gaussian_filter1d = None
        for _ in range(3):
            parts = []
            for size, length in zip(self.shape, correlation, strict=True):
                values = rng.standard_normal(size).astype(np.float32)
                if gaussian_filter1d is not None:
                    values = gaussian_filter1d(
                        values,
                        sigma=max(float(length) / 3.0, 0.5),
                        mode="wrap",
                        truncate=3.0,
                    ).astype(np.float32, copy=False)
                values -= np.float32(np.mean(values))
                values /= np.float32(max(float(np.std(values)), 1e-6))
                parts.append(values)
            vectors.append((parts[0], parts[1], parts[2]))
        self.vectors = tuple(vectors)

    def chunk(self, start: int, stop: int) -> np.ndarray:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = self.vectors
        value = (
            x0[start:stop, None, None] * y0[None, :, None]
            + x1[start:stop, None, None] * z1[None, None, :]
            + y2[None, :, None] * z2[None, None, :]
        ).astype(np.float32, copy=False)
        value -= np.float32(np.mean(value))
        value /= np.float32(max(float(np.std(value)), 1e-6))
        return value


def _correlated_interface_perturbation(
    output: np.ndarray,
    config: ImageToVoxelConfig,
    rng: np.random.Generator,
) -> None:
    noise_model = _SeparableCorrelatedNoise(
        tuple(map(int, output.shape)), config.correlation_length_voxels, rng
    )
    nx, ny, nz = map(int, output.shape)
    threshold_quantile = max(0.5, 1.0 - 0.15 * config.stochastic_strength)
    offsets = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    for start in range(0, nx, config.chunk_size_x):
        stop = min(nx, start + config.chunk_size_x)
        noise = noise_model.chunk(start, stop)
        magnitude = np.abs(noise)
        choose = magnitude > float(np.quantile(magnitude, threshold_quantile))
        selected = np.mod(np.floor(magnitude * 997.0).astype(np.int8), len(offsets))
        original = np.asarray(output[start:stop]).copy()
        updated = original.copy()
        for code, (di, dj, dk) in enumerate(offsets[1:], start=1):
            mask = choose & (selected == code)
            if not np.any(mask):
                continue
            xi = np.clip(np.arange(start, stop) + di, 0, nx - 1)
            yi = np.clip(np.arange(ny) + dj, 0, ny - 1)
            zi = np.clip(np.arange(nz) + dk, 0, nz - 1)
            neighbor = np.asarray(output[np.ix_(xi, yi, zi)])
            updated[mask] = neighbor[mask]
        output[start:stop] = updated


def _relax_projections(
    output: np.ndarray,
    views: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    # Vectorized stochastic ray repair: a third of inconsistent rays are
    # updated per sweep without a Python loop over pixels.
    for name, target in views.items():
        projected = _mode_projection(output, name)
        inconsistent = np.argwhere(projected != target)
        count = max(1, len(inconsistent) // 3) if len(inconsistent) else 0
        if count == 0:
            continue
        chosen = rng.choice(len(inconsistent), size=count, replace=False)
        indices = inconsistent[chosen]
        first, second = indices[:, 0], indices[:, 1]
        phases = target[first, second]
        if name == "xy":
            depth = rng.integers(output.shape[2], size=count)
            output[first, second, depth] = phases
        elif name == "xz":
            depth = rng.integers(output.shape[1], size=count)
            output[first, depth, second] = phases
        else:
            depth = rng.integers(output.shape[0], size=count)
            output[depth, first, second] = phases


def _mode_projection(volume: np.ndarray, name: str) -> np.ndarray:
    axis = {"xy": 2, "xz": 1, "yz": 0}[name]
    phases = np.unique(volume)
    counts = np.stack([(volume == phase).sum(axis=axis) for phase in phases], axis=0)
    return phases[np.argmax(counts, axis=0)].astype(np.int32, copy=False)


def _projection_agreement(volume: np.ndarray, views: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(_mode_projection(volume, name) == target)) for name, target in views.items()}


__all__ = [
    "ImageToVoxelConfig",
    "ImageToVoxelResult",
    "OrthogonalImages",
    "reconstruct_electrode_volume",
]
