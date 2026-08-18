from __future__ import annotations

import numpy as np
import pytest

from zynnova.zynmorph import (
    MicrostructureCondition,
    SliceObservation,
    reconstruct_from_slices,
)


def _condition() -> MicrostructureCondition:
    return MicrostructureCondition(
        shape=(5, 6, 7),
        phase_fractions={1: 0.50, 2: 0.30, 5: 0.20},
        voxel_size_m=(80e-9, 100e-9, 120e-9),
        correlation_lengths_voxels={1: 2.0, 2: 1.5, 5: 1.0},
        seed=123,
    )


def _resize_nearest_2d(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    indices = [
        np.rint(np.linspace(0, values.shape[axis] - 1, shape[axis])).astype(np.int64)
        for axis in range(2)
    ]
    return values[np.ix_(indices[0], indices[1])]


@pytest.mark.parametrize(
    ("axis", "low_resolution"),
    [
        (0, np.asarray([[1, 2, 5], [5, 1, 2]], dtype=np.int32)),
        (1, np.asarray([[1, 5, 2], [2, 1, 5]], dtype=np.int32)),
        (2, np.asarray([[1, 2], [5, 1], [2, 5]], dtype=np.int32)),
    ],
)
def test_reconstruction_broadcasts_low_resolution_plane_along_only_normal_axis(
    axis: int,
    low_resolution: np.ndarray,
) -> None:
    condition = _condition()
    observation = SliceObservation(
        low_resolution,
        axis=axis,
        index_fraction=0.4,
        weight=1.5,
    )

    reconstructed = reconstruct_from_slices([observation], condition, prior_weight=0.35)

    assert reconstructed.shape == condition.shape
    index = int(round(observation.index_fraction * (condition.shape[axis] - 1)))
    plane_shape = tuple(condition.shape[item] for item in range(3) if item != axis)
    expected = _resize_nearest_2d(low_resolution, plane_shape)
    selection = [slice(None)] * 3
    selection[axis] = index
    assert np.array_equal(reconstructed.labels[tuple(selection)], expected)


def test_reconstruction_rejects_invalid_prior_and_unknown_phase() -> None:
    condition = _condition()
    valid = SliceObservation(np.asarray([[1, 2], [5, 1]]), axis=0)
    unknown = SliceObservation(np.asarray([[1, 99], [5, 1]]), axis=0)

    with pytest.raises(ValueError, match="prior_weight"):
        reconstruct_from_slices([valid], condition, prior_weight=-0.1)
    with pytest.raises(ValueError, match="phases absent"):
        reconstruct_from_slices([unknown], condition)
