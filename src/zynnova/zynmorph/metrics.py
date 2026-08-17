"""Morphology descriptors used for validation and descriptor-conditioned design."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    phase: int
    volume_fraction: float
    connected_fraction: float
    percolates: tuple[bool, bool, bool]
    specific_surface_area_per_m: float


@dataclass(frozen=True, slots=True)
class MicrostructureMetrics:
    phases: Mapping[int, PhaseMetrics]
    interface_area_m2: Mapping[tuple[int, int], float]
    total_volume_m3: float
    voxel_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def flatten(self) -> Mapping[str, float]:
        result: dict[str, float] = {"total_volume_m3": self.total_volume_m3}
        for phase, metric in self.phases.items():
            prefix = f"phase_{phase}"
            result[f"{prefix}.volume_fraction"] = metric.volume_fraction
            result[f"{prefix}.connected_fraction"] = metric.connected_fraction
            result[f"{prefix}.specific_surface_area_per_m"] = (
                metric.specific_surface_area_per_m
            )
            for axis, value in enumerate(metric.percolates):
                result[f"{prefix}.percolates_axis_{axis}"] = float(value)
        for pair, value in self.interface_area_m2.items():
            result[f"interface_{pair[0]}_{pair[1]}.area_m2"] = value
        return result


def analyze_microstructure(volume: MicrostructureVolume) -> MicrostructureMetrics:
    labels = volume.labels
    spacing = volume.voxel_size_m
    assert isinstance(spacing, tuple)
    voxel_volume = float(np.prod(spacing))
    total_volume = labels.size * voxel_volume
    interface_area = _interface_areas(labels, spacing)
    phases: dict[int, PhaseMetrics] = {}
    for phase in np.unique(labels):
        phase_id = int(phase)
        mask = labels == phase_id
        connected_fraction, percolates = _connectivity(mask)
        external_area = sum(
            area
            for pair, area in interface_area.items()
            if phase_id in pair
        )
        phases[phase_id] = PhaseMetrics(
            phase=phase_id,
            volume_fraction=float(np.count_nonzero(mask) / labels.size),
            connected_fraction=connected_fraction,
            percolates=percolates,
            specific_surface_area_per_m=external_area / max(np.count_nonzero(mask) * voxel_volume, 1e-300),
        )
    return MicrostructureMetrics(
        phases=phases,
        interface_area_m2=interface_area,
        total_volume_m3=total_volume,
        voxel_count=labels.size,
        metadata={"shape": volume.shape, "voxel_size_m": spacing},
    )


def descriptor_loss(
    metrics: MicrostructureMetrics,
    targets: Mapping[str, float],
    *,
    relative_floor: float = 1.0e-12,
) -> float:
    flat = metrics.flatten()
    loss = 0.0
    for key, target in targets.items():
        if key not in flat:
            continue
        scale = max(abs(float(target)), relative_floor)
        loss += ((flat[key] - float(target)) / scale) ** 2
    return float(loss)


def _interface_areas(
    labels: np.ndarray,
    spacing: tuple[float, float, float],
) -> Mapping[tuple[int, int], float]:
    areas = (spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1])
    result: dict[tuple[int, int], float] = {}
    for axis in range(3):
        left = np.take(labels, np.arange(labels.shape[axis] - 1), axis=axis)
        right = np.take(labels, np.arange(1, labels.shape[axis]), axis=axis)
        changed = left != right
        left_values = left[changed]
        right_values = right[changed]
        for first, second in zip(left_values, right_values, strict=True):
            pair = tuple(sorted((int(first), int(second))))
            result[pair] = result.get(pair, 0.0) + areas[axis]
    return result


def _connectivity(mask: np.ndarray) -> tuple[float, tuple[bool, bool, bool]]:
    coordinates = np.argwhere(mask)
    count = len(coordinates)
    if count == 0:
        return 0.0, (False, False, False)
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    percolates = [False, False, False]
    shape = mask.shape
    for seed_array in coordinates:
        seed = tuple(int(item) for item in seed_array)
        if visited[seed]:
            continue
        queue: deque[tuple[int, int, int]] = deque([seed])
        visited[seed] = True
        size = 0
        touches_low = [False, False, False]
        touches_high = [False, False, False]
        while queue:
            current = queue.popleft()
            size += 1
            for axis in range(3):
                touches_low[axis] |= current[axis] == 0
                touches_high[axis] |= current[axis] == shape[axis] - 1
            for axis in range(3):
                for delta in (-1, 1):
                    neighbor = list(current)
                    neighbor[axis] += delta
                    if not 0 <= neighbor[axis] < shape[axis]:
                        continue
                    key = tuple(neighbor)
                    if mask[key] and not visited[key]:
                        visited[key] = True
                        queue.append(key)
        largest = max(largest, size)
        for axis in range(3):
            percolates[axis] |= touches_low[axis] and touches_high[axis]
    return largest / count, tuple(percolates)  # type: ignore[return-value]


__all__ = [
    "MicrostructureMetrics",
    "PhaseMetrics",
    "analyze_microstructure",
    "descriptor_loss",
]
