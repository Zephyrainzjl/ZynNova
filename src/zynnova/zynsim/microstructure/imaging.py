"""Fast image segmentation and phase measurements for electrode micrographs.

The public API accepts NumPy arrays or image paths.  Segmentation can be fully
reproducible through explicit intensity/color thresholds, or automatically
estimated with a deterministic multi-Otsu dynamic program.  All operations
have NumPy/SciPy fallbacks; the optional native backend is used when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:  # optional native acceleration
    from zynnova._native import _zynsim_voxel_native as _native
except Exception:  # pragma: no cover - optional extension
    _native = None


@dataclass(frozen=True, slots=True)
class PhaseThreshold:
    """One phase classification rule.

    ``lower`` and ``upper`` are inclusive grayscale bounds.  For RGB input a
    ``color`` and Euclidean ``tolerance`` can be supplied instead.
    """

    label: int
    lower: float | None = None
    upper: float | None = None
    color: tuple[float, ...] | None = None
    tolerance: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.color is None:
            if self.lower is None or self.upper is None:
                raise ValueError("grayscale thresholds require lower and upper")
            if float(self.lower) > float(self.upper):
                raise ValueError("threshold lower cannot exceed upper")
        else:
            if not self.color:
                raise ValueError("color threshold cannot be empty")
            if self.tolerance is None or float(self.tolerance) < 0.0:
                raise ValueError("color threshold requires non-negative tolerance")


@dataclass(frozen=True, slots=True)
class ImageSegmentationConfig:
    thresholds: tuple[PhaseThreshold, ...] = ()
    automatic_phase_count: int | None = None
    labels: tuple[int, ...] | None = None
    channel: int | LiteralChannel = "luminance"
    invert: bool = False
    unclassified_label: int = 0
    median_radius: int = 0
    majority_iterations: int = 0
    minimum_component_pixels: int = 0
    preserve_dtype_range: bool = False

    def __post_init__(self) -> None:
        if not self.thresholds and self.automatic_phase_count is None:
            raise ValueError("provide thresholds or automatic_phase_count")
        if self.automatic_phase_count is not None:
            if not 2 <= int(self.automatic_phase_count) <= 16:
                raise ValueError("automatic_phase_count must lie in [2, 16]")
            if self.labels is not None and len(self.labels) != int(self.automatic_phase_count):
                raise ValueError("labels length must match automatic_phase_count")
        if self.median_radius < 0 or self.majority_iterations < 0:
            raise ValueError("cleanup iteration counts cannot be negative")
        if self.minimum_component_pixels < 0:
            raise ValueError("minimum_component_pixels cannot be negative")


LiteralChannel = str | int


@dataclass(frozen=True, slots=True)
class PhaseImageReport:
    labels: np.ndarray
    phase_fractions: Mapping[int, float]
    phase_pixels: Mapping[int, int]
    thresholds: tuple[float, ...]
    shape: tuple[int, int]
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", np.ascontiguousarray(self.labels, dtype=np.int32))
        object.__setattr__(self, "phase_fractions", dict(self.phase_fractions))
        object.__setattr__(self, "phase_pixels", dict(self.phase_pixels))
        object.__setattr__(self, "metadata", dict(self.metadata))


def read_image(image: str | Path | np.ndarray) -> np.ndarray:
    """Read an image without importing heavy plotting packages."""

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("image paths require Pillow; install zynnova[zynsim-all]") from exc
        with Image.open(Path(image)) as handle:
            array = np.asarray(handle)
    if array.ndim not in (2, 3):
        raise ValueError("image must be 2-D grayscale or 3-D channel-last")
    if array.ndim == 3 and array.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("channel-last image must have 1, 2, 3, or 4 channels")
    if not np.isfinite(np.asarray(array, dtype=float)).all():
        raise ValueError("image contains non-finite values")
    return np.ascontiguousarray(array)


def segment_electrode_image(
    image: str | Path | np.ndarray,
    config: ImageSegmentationConfig,
) -> PhaseImageReport:
    """Segment one electrode image into integer material phases."""

    raw = read_image(image)
    scalar = _scalar_image(raw, config.channel)
    if config.invert:
        scalar = float(np.max(scalar) + np.min(scalar)) - scalar
    if not config.preserve_dtype_range:
        scalar = _normalize_255(scalar)

    applied_thresholds: tuple[float, ...]
    if config.thresholds:
        grayscale_rules = all(rule.color is None for rule in config.thresholds)
        native_u8_compatible = (
            _native is not None
            and grayscale_rules
            and float(np.min(scalar)) >= 0.0
            and float(np.max(scalar)) <= 255.0
            and all(
                0.0 <= float(rule.lower) <= 255.0 and 0.0 <= float(rule.upper) <= 255.0
                for rule in config.thresholds
            )
        )
        if native_u8_compatible:
            values_u8 = np.rint(scalar).astype(np.uint8, copy=False)
            labels = _native.threshold_labels_u8(
                values_u8,
                [int(np.clip(round(float(rule.lower)), 0, 255)) for rule in config.thresholds],
                [int(np.clip(round(float(rule.upper)), 0, 255)) for rule in config.thresholds],
                [int(rule.label) for rule in config.thresholds],
                int(config.unclassified_label),
            )
        else:
            labels = _explicit_segmentation(raw, scalar, config)
        applied_thresholds = tuple(
            float(value)
            for rule in config.thresholds
            for value in (rule.lower, rule.upper)
            if value is not None
        )
    else:
        phase_count = int(config.automatic_phase_count or 2)
        thresholds = multi_otsu_thresholds(scalar, phase_count)
        output_labels = (
            np.arange(phase_count, dtype=np.int32)
            if config.labels is None
            else np.asarray(config.labels, dtype=np.int32)
        )
        bins = np.searchsorted(np.asarray(thresholds), scalar, side="right")
        labels = output_labels[bins]
        applied_thresholds = tuple(map(float, thresholds))

    labels = _cleanup(labels, config)
    unique, counts = np.unique(labels, return_counts=True)
    fractions = {int(label): float(count / labels.size) for label, count in zip(unique, counts, strict=True)}
    pixels = {int(label): int(count) for label, count in zip(unique, counts, strict=True)}
    source = "array" if isinstance(image, np.ndarray) else str(Path(image))
    return PhaseImageReport(
        labels=labels,
        phase_fractions=fractions,
        phase_pixels=pixels,
        thresholds=applied_thresholds,
        shape=tuple(map(int, labels.shape)),
        source=source,
        metadata={"native_backend": bool(_native is not None)},
    )


def multi_otsu_thresholds(image: np.ndarray, phase_count: int) -> tuple[float, ...]:
    """Return deterministic multi-Otsu thresholds using dynamic programming.

    Complexity is ``O(classes * bins**2)`` with 256 histogram bins, independent
    of image size after histogram construction.
    """

    values = _normalize_255(np.asarray(image, dtype=float))
    classes = int(phase_count)
    if not 2 <= classes <= 16:
        raise ValueError("phase_count must lie in [2, 16]")
    hist = np.bincount(np.rint(values).astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total <= 0:
        raise ValueError("empty image")
    probability = hist / total
    omega = np.concatenate(([0.0], np.cumsum(probability)))
    moment = np.concatenate(([0.0], np.cumsum(probability * np.arange(256))))

    def interval_score(left: int, right: int) -> float:
        weight = omega[right + 1] - omega[left]
        if weight <= 0.0:
            return -np.inf
        mean_numerator = moment[right + 1] - moment[left]
        return mean_numerator * mean_numerator / weight

    dp = np.full((classes + 1, 256), -np.inf, dtype=float)
    split = np.full((classes + 1, 256), -1, dtype=np.int16)
    for right in range(256):
        dp[1, right] = interval_score(0, right)
    for group in range(2, classes + 1):
        for right in range(group - 1, 256):
            best = -np.inf
            best_left = -1
            for left in range(group - 2, right):
                candidate = dp[group - 1, left] + interval_score(left + 1, right)
                if candidate > best:
                    best = candidate
                    best_left = left
            dp[group, right] = best
            split[group, right] = best_left
    cuts: list[int] = []
    right = 255
    for group in range(classes, 1, -1):
        left = int(split[group, right])
        if left < 0:
            # Degenerate histogram: deterministic quantile fallback.
            return tuple(float(np.quantile(values, q)) for q in np.linspace(0, 1, classes + 1)[1:-1])
        cuts.append(left)
        right = left
    cuts.reverse()
    return tuple(float(cut) + 0.5 for cut in cuts)


def _scalar_image(image: np.ndarray, channel: LiteralChannel) -> np.ndarray:
    if image.ndim == 2:
        return np.asarray(image, dtype=float)
    channels = np.asarray(image[..., :3], dtype=float)
    if isinstance(channel, int):
        if not 0 <= channel < image.shape[-1]:
            raise ValueError("channel index is out of range")
        return np.asarray(image[..., channel], dtype=float)
    key = str(channel).lower()
    if key in {"luminance", "gray", "grey"}:
        if channels.shape[-1] == 1:
            return channels[..., 0]
        return 0.2126 * channels[..., 0] + 0.7152 * channels[..., 1] + 0.0722 * channels[..., 2]
    mapping = {"r": 0, "red": 0, "g": 1, "green": 1, "b": 2, "blue": 2}
    if key not in mapping or mapping[key] >= image.shape[-1]:
        raise ValueError(f"unknown channel {channel!r}")
    return np.asarray(image[..., mapping[key]], dtype=float)


def _normalize_255(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi <= lo:
        return np.zeros_like(array)
    return (array - lo) * (255.0 / (hi - lo))


def _explicit_segmentation(
    raw: np.ndarray,
    scalar: np.ndarray,
    config: ImageSegmentationConfig,
) -> np.ndarray:
    labels = np.full(scalar.shape, int(config.unclassified_label), dtype=np.int32)
    assigned = np.zeros(scalar.shape, dtype=bool)
    for rule in config.thresholds:
        if rule.color is None:
            mask = (scalar >= float(rule.lower)) & (scalar <= float(rule.upper))
        else:
            if raw.ndim != 3:
                raise ValueError("color thresholds require a channel image")
            color = np.asarray(rule.color, dtype=float)
            if color.size > raw.shape[-1]:
                raise ValueError("threshold color has more channels than image")
            difference = np.asarray(raw[..., : color.size], dtype=float) - color
            mask = np.linalg.norm(difference, axis=-1) <= float(rule.tolerance)
        # First rule wins in overlap regions, making configuration order explicit.
        mask &= ~assigned
        labels[mask] = int(rule.label)
        assigned |= mask
    return labels


def _cleanup(labels: np.ndarray, config: ImageSegmentationConfig) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int32)
    if config.median_radius:
        try:
            from scipy.ndimage import median_filter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("median cleanup requires scipy") from exc
        size = 2 * int(config.median_radius) + 1
        result = median_filter(result, size=size, mode="nearest")
    for _ in range(int(config.majority_iterations)):
        result = _majority_filter(result)
    if config.minimum_component_pixels:
        result = _remove_small_components(result, int(config.minimum_component_pixels))
    return np.ascontiguousarray(result, dtype=np.int32)


def _majority_filter(labels: np.ndarray) -> np.ndarray:
    padded = np.pad(labels, 1, mode="edge")
    neighborhoods = np.stack(
        [padded[1 + di : 1 + di + labels.shape[0], 1 + dj : 1 + dj + labels.shape[1]]
         for di in (-1, 0, 1) for dj in (-1, 0, 1)],
        axis=0,
    )
    phases = np.unique(labels)
    counts = np.stack([(neighborhoods == phase).sum(axis=0) for phase in phases], axis=0)
    return phases[np.argmax(counts, axis=0)].astype(np.int32, copy=False)


def _remove_small_components(labels: np.ndarray, minimum: int) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover
        raise ImportError("component cleanup requires scipy") from exc
    result = labels.copy()
    structure = np.ones((3, 3), dtype=np.uint8)
    for phase in np.unique(labels):
        components, number = ndimage.label(labels == phase, structure=structure)
        if number == 0:
            continue
        sizes = np.bincount(components.ravel())
        small = np.flatnonzero((sizes < minimum) & (np.arange(len(sizes)) != 0))
        for component in small:
            mask = components == component
            dilated = ndimage.binary_dilation(mask, structure=structure)
            neighbors = result[dilated & ~mask]
            if neighbors.size:
                values, counts = np.unique(neighbors, return_counts=True)
                result[mask] = values[np.argmax(counts)]
    return result


__all__ = [
    "ImageSegmentationConfig",
    "PhaseImageReport",
    "PhaseThreshold",
    "multi_otsu_thresholds",
    "read_image",
    "segment_electrode_image",
]
