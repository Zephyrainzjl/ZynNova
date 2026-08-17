"""2-D/3-D characterization-driven multi-phase microstructure generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .schema import BatteryPhase, ManufacturingProcessControl, validate_phase_labels


GenerationModel = Callable[..., np.ndarray]


@dataclass(frozen=True, slots=True)
class GeneratedMicrostructure:
    phase_labels: np.ndarray
    source: str
    process: ManufacturingProcessControl
    target_shape: tuple[int, int, int]
    phase_volume_fractions: Mapping[int, float]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_labels", validate_phase_labels(self.phase_labels))


class CharacterizationDrivenGenerator:
    """Generate a 3-D label field from measured slices or a learned generator.

    A user-supplied ``model`` may be a diffusion, flow-matching, GAN, or neural
    field generator.  The deterministic fallback is intentionally transparent:
    it extrudes measured slices, introduces correlated topology variation, then
    applies explicit manufacturing controls without silently inventing phases.
    """

    def __init__(self, model: GenerationModel | None = None) -> None:
        self.model = model

    def from_3d(
        self,
        phase_labels: np.ndarray,
        *,
        process: ManufacturingProcessControl | None = None,
    ) -> GeneratedMicrostructure:
        labels = validate_phase_labels(phase_labels)
        resolved = process or ManufacturingProcessControl()
        processed = apply_manufacturing_controls(labels, resolved)
        return _result(processed, "measured-3d", resolved, {"input": "3d"})

    def from_2d(
        self,
        slices: Sequence[np.ndarray] | np.ndarray,
        target_shape: tuple[int, int, int],
        *,
        process: ManufacturingProcessControl | None = None,
        conditioning: Mapping[str, object] | None = None,
    ) -> GeneratedMicrostructure:
        resolved = process or ManufacturingProcessControl()
        measured = _normalize_slices(slices)
        if self.model is not None:
            generated = self.model(
                measured,
                target_shape=target_shape,
                process=resolved,
                conditioning=dict(conditioning or {}),
            )
            labels = validate_phase_labels(generated)
            source = f"learned:{type(self.model).__name__}"
        else:
            labels = _stochastic_extrusion(measured, target_shape, resolved.random_seed)
            source = "correlated-slice-extrusion"
        labels = apply_manufacturing_controls(labels, resolved)
        return _result(
            labels,
            source,
            resolved,
            {"input_slices": len(measured), **dict(conditioning or {})},
        )


def apply_manufacturing_controls(
    phase_labels: np.ndarray,
    process: ManufacturingProcessControl,
) -> np.ndarray:
    labels = validate_phase_labels(phase_labels).copy()
    rng = np.random.default_rng(process.random_seed)
    if process.calendering_ratio > 0.0:
        labels = _calender(labels, process.calendering_ratio)
    labels = _redistribute_electrode_phases(labels, process, rng)
    if process.particle_coalescence > 0.0:
        labels = _majority_relax(labels, process.particle_coalescence, iterations=3)
    if process.sei_thickness_voxels:
        labels = _grow_interphase(
            labels,
            active=int(BatteryPhase.NEGATIVE_ACTIVE),
            electrolyte=int(BatteryPhase.NEGATIVE_ELECTROLYTE),
            interphase=int(BatteryPhase.NEGATIVE_SEI),
            thickness=process.sei_thickness_voxels,
        )
    if process.cei_thickness_voxels:
        labels = _grow_interphase(
            labels,
            active=int(BatteryPhase.POSITIVE_ACTIVE),
            electrolyte=int(BatteryPhase.POSITIVE_ELECTROLYTE),
            interphase=int(BatteryPhase.POSITIVE_CEI),
            thickness=process.cei_thickness_voxels,
        )
    if process.crack_volume_fraction > 0.0:
        labels = _insert_cracks(
            labels,
            process.crack_volume_fraction,
            process.crack_anisotropy,
            rng,
        )
    return validate_phase_labels(labels)


def _normalize_slices(slices: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(slices, np.ndarray) and slices.ndim == 2:
        values = [slices]
    elif isinstance(slices, np.ndarray) and slices.ndim == 3:
        values = [slices[index] for index in range(slices.shape[0])]
    else:
        values = list(slices)
    if not values:
        raise ValueError("at least one measured slice is required")
    shape = values[0].shape
    if len(shape) != 2 or any(np.asarray(value).shape != shape for value in values):
        raise ValueError("all measured slices must share one two-dimensional shape")
    return [np.ascontiguousarray(value, dtype=np.int32) for value in values]


def _stochastic_extrusion(
    slices: Sequence[np.ndarray],
    target_shape: tuple[int, int, int],
    seed: int,
) -> np.ndarray:
    if len(target_shape) != 3 or min(target_shape) < 1:
        raise ValueError("target_shape must contain three positive integers")
    source = np.stack(slices, axis=0)
    source_indices = np.linspace(0, len(source) - 1, target_shape[0])
    nearest = np.rint(source_indices).astype(int)
    volume = source[nearest]
    volume = _resize_nearest(volume, target_shape)
    rng = np.random.default_rng(seed)
    for z in range(1, target_shape[0]):
        shift_y, shift_x = rng.integers(-1, 2, size=2)
        mask = rng.random(target_shape[1:]) < 0.12
        shifted = np.roll(volume[z - 1], (shift_y, shift_x), axis=(0, 1))
        volume[z][mask] = shifted[mask]
    return validate_phase_labels(volume)


def _resize_nearest(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    indices = [
        np.minimum(
            np.rint(np.linspace(0, volume.shape[axis] - 1, shape[axis])).astype(int),
            volume.shape[axis] - 1,
        )
        for axis in range(3)
    ]
    return volume[np.ix_(indices[0], indices[1], indices[2])]


def _calender(labels: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0.0:
        return labels
    target_x = max(1, int(round(labels.shape[0] * (1.0 - 0.45 * ratio))))
    compressed = _resize_nearest(labels, (target_x, labels.shape[1], labels.shape[2]))
    return _resize_nearest(compressed, labels.shape)


def _redistribute_electrode_phases(
    labels: np.ndarray,
    process: ManufacturingProcessControl,
    rng: np.random.Generator,
) -> np.ndarray:
    result = labels.copy()
    specifications = (
        (
            int(BatteryPhase.POSITIVE_ACTIVE),
            int(BatteryPhase.POSITIVE_ELECTROLYTE),
            int(BatteryPhase.POSITIVE_CBD),
            process.positive_porosity,
            process.positive_cbd_fraction,
        ),
        (
            int(BatteryPhase.NEGATIVE_ACTIVE),
            int(BatteryPhase.NEGATIVE_ELECTROLYTE),
            int(BatteryPhase.NEGATIVE_CBD),
            process.negative_porosity,
            process.negative_cbd_fraction,
        ),
    )
    for active, pore, cbd, porosity, cbd_fraction in specifications:
        domain = (result == active) | (result == pore) | (result == cbd)
        indices = np.flatnonzero(domain)
        if not len(indices):
            continue
        rng.shuffle(indices)
        pore_count = int(round(porosity * len(indices)))
        cbd_count = int(round(cbd_fraction * len(indices)))
        flat = result.reshape(-1)
        flat[indices] = active
        flat[indices[:pore_count]] = pore
        flat[indices[pore_count : pore_count + cbd_count]] = cbd
    return result


def _majority_relax(labels: np.ndarray, strength: float, iterations: int) -> np.ndarray:
    result = labels.copy()
    for _ in range(iterations):
        neighbors = np.stack(
            [
                np.roll(result, shift, axis)
                for axis in range(3)
                for shift in (-1, 1)
            ],
            axis=0,
        )
        candidate = np.empty_like(result)
        for index in np.ndindex(result.shape):
            values, counts = np.unique(neighbors[(slice(None),) + index], return_counts=True)
            candidate[index] = values[int(np.argmax(counts))]
        replace = np.random.default_rng(iterations + result.size).random(result.shape) < 0.25 * strength
        result[replace] = candidate[replace]
    return result


def _grow_interphase(
    labels: np.ndarray,
    *,
    active: int,
    electrolyte: int,
    interphase: int,
    thickness: int,
) -> np.ndarray:
    result = labels.copy()
    frontier = result == active
    for _ in range(thickness):
        adjacent = np.zeros_like(frontier)
        for axis in range(3):
            adjacent |= np.roll(frontier, 1, axis=axis)
            adjacent |= np.roll(frontier, -1, axis=axis)
        layer = adjacent & (result == electrolyte)
        result[layer] = interphase
        frontier = layer
    return result


def _insert_cracks(
    labels: np.ndarray,
    volume_fraction: float,
    anisotropy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = labels.copy()
    active_mask = (result == int(BatteryPhase.POSITIVE_ACTIVE)) | (
        result == int(BatteryPhase.NEGATIVE_ACTIVE)
    )
    count = int(round(volume_fraction * np.sum(active_mask)))
    if count <= 0:
        return result
    noise = rng.normal(size=result.shape)
    for _ in range(3):
        noise = (
            noise
            + (1.0 + anisotropy) * np.roll(noise, 1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, 1, axis=2)
        ) / (4.0 + anisotropy)
    scores = np.where(active_mask, noise, -np.inf)
    flat_indices = np.argpartition(scores.reshape(-1), -count)[-count:]
    result.reshape(-1)[flat_indices] = int(BatteryPhase.CRACK)
    return result


def _result(
    labels: np.ndarray,
    source: str,
    process: ManufacturingProcessControl,
    metadata: Mapping[str, object],
) -> GeneratedMicrostructure:
    unique, counts = np.unique(labels, return_counts=True)
    fractions = {int(label): float(count / labels.size) for label, count in zip(unique, counts, strict=True)}
    return GeneratedMicrostructure(
        phase_labels=labels,
        source=source,
        process=process,
        target_shape=tuple(map(int, labels.shape)),
        phase_volume_fractions=fractions,
        metadata=dict(metadata),
    )


__all__ = [
    "CharacterizationDrivenGenerator",
    "GeneratedMicrostructure",
    "GenerationModel",
    "apply_manufacturing_controls",
]
