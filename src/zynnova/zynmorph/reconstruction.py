"""2-D/orthogonal-slice to 3-D reconstruction with multi-plane consensus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .generation import SpectralConditionalGenerator
from .schema import MicrostructureCondition
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class SliceObservation:
    labels: np.ndarray
    axis: int
    index_fraction: float = 0.5
    weight: float = 1.0

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self.labels, dtype=np.int32)
        if labels.ndim != 2 or min(labels.shape) < 1:
            raise ValueError("slice labels must be a non-empty two-dimensional array")
        if self.axis not in {0, 1, 2}:
            raise ValueError("slice axis must be 0, 1, or 2")
        if not 0.0 <= self.index_fraction <= 1.0:
            raise ValueError("index_fraction must lie in [0,1]")
        if self.weight <= 0.0:
            raise ValueError("slice weight must be positive")
        object.__setattr__(self, "labels", labels)


def reconstruct_from_slices(
    observations: Sequence[SliceObservation],
    condition: MicrostructureCondition,
    *,
    prior_weight: float = 0.35,
) -> MicrostructureVolume:
    """Fuse orthogonal labels with a correlated exact-composition 3-D prior."""

    if not observations:
        raise ValueError("at least one SliceObservation is required")
    if not np.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValueError("prior_weight must be finite and non-negative")
    allowed_phases = set(condition.phases)
    for observation in observations:
        unknown = sorted(
            int(value)
            for value in np.unique(observation.labels)
            if int(value) not in allowed_phases
        )
        if unknown:
            raise ValueError(
                f"slice observation contains phases absent from the condition: {unknown}"
            )
    generated = SpectralConditionalGenerator().generate(condition).volume
    phases = condition.phases
    phase_index = {phase: index for index, phase in enumerate(phases)}
    votes = np.zeros((len(phases), *condition.shape), dtype=np.float32)
    for index, phase in enumerate(phases):
        votes[index] += prior_weight * (generated.labels == phase)
    for observation in observations:
        resized = _resize_nearest_2d(
            observation.labels,
            _plane_shape(condition.shape, observation.axis),
        )
        expanded = np.expand_dims(resized, axis=observation.axis)
        # The resized observation already spans the two in-plane dimensions.
        # Broadcast only along the missing normal axis; tiling by the full
        # three-dimensional shape would multiply the in-plane dimensions too.
        expanded = np.broadcast_to(expanded, condition.shape)
        for phase in np.unique(expanded):
            phase_id = int(phase)
            if phase_id in phase_index:
                votes[phase_index[phase_id]] += observation.weight * (expanded == phase_id)
    labels = np.asarray(phases, dtype=np.int32)[np.argmax(votes, axis=0)]
    labels = _impose_observed_planes(labels, observations)
    return MicrostructureVolume(
        labels=labels,
        voxel_size_m=condition.voxel_size_m,
        metadata={
            "reconstruction": "multi-plane-consensus",
            "observations": len(observations),
            "prior_weight": prior_weight,
        },
    )


def _impose_observed_planes(
    labels: np.ndarray,
    observations: Sequence[SliceObservation],
) -> np.ndarray:
    result = labels.copy()
    for observation in observations:
        index = int(round(observation.index_fraction * (result.shape[observation.axis] - 1)))
        plane = _resize_nearest_2d(observation.labels, _plane_shape(result.shape, observation.axis))
        selection = [slice(None)] * 3
        selection[observation.axis] = index
        result[tuple(selection)] = plane
    return result


def _plane_shape(shape: tuple[int, int, int], axis: int) -> tuple[int, int]:
    return tuple(shape[index] for index in range(3) if index != axis)  # type: ignore[return-value]


def _resize_nearest_2d(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    indices = [
        np.rint(np.linspace(0, values.shape[axis] - 1, shape[axis])).astype(np.int64)
        for axis in range(2)
    ]
    return values[np.ix_(indices[0], indices[1])]


__all__ = ["SliceObservation", "reconstruct_from_slices"]
