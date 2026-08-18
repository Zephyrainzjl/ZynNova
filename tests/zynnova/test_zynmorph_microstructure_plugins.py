from __future__ import annotations

import numpy as np
import pytest

from zynnova.zynmorph.microstructure import (
    CharacterizationSettings,
    DESCRIPTORS,
    LOSSES,
    OPTIMIZERS,
    ReconstructionSettings,
    characterize_microstructure,
    compute_loss,
    interpolate_characterizations,
    match_microstructure,
    merge_characterizations,
    reconstruct_microstructure,
)


def _binary_2d():
    x = np.zeros((16, 16), dtype=np.int32)
    x[3:12, 4:10] = 1
    return x


def test_registry_covers_complete_descriptor_loss_optimizer_surface():
    assert {
        "Correlations",
        "CrossCorrelations",
        "FFTCorrelations",
        "FFTCrossCorrelations",
        "GramMatrices",
        "LineCorrelations",
        "LineLinealPathApproximation",
        "LinealPath",
        "LinealPathApproximation",
        "MultiPhaseGramMatrices",
        "OrientationDescriptor",
        "TwoPointCorrelations",
        "Variation",
        "VolumeFractions",
    }.issubset(set(DESCRIPTORS.names()))
    assert {"L1", "L2", "MSE", "RMS", "SSE"}.issubset(set(LOSSES.names()))
    assert {
        "Adadelta", "Adagrad", "Adam", "Adamax", "LBFGSB", "Nadam",
        "RMSprop", "SGD", "SimulatedAnnealing", "TNC", "YTPost",
    }.issubset(set(OPTIMIZERS.names()))


@pytest.mark.parametrize("descriptor", [
    "Correlations", "CrossCorrelations", "FFTCorrelations", "FFTCrossCorrelations",
    "GramMatrices", "LineCorrelations", "LineLinealPathApproximation", "LinealPath",
    "LinealPathApproximation", "MultiPhaseGramMatrices", "OrientationDescriptor",
    "TwoPointCorrelations", "Variation", "VolumeFractions",
])
def test_every_descriptor_characterizes_2d(descriptor):
    result = characterize_microstructure(
        _binary_2d(),
        CharacterizationSettings(
            descriptor_types=(descriptor,),
            limit_to=4,
            use_multigrid_descriptors=False,
        ),
    )
    values = result.descriptors[descriptor].values
    assert values.size > 0
    assert np.all(np.isfinite(values))


def test_3d_directional_slice_characterization():
    volume = np.zeros((8, 10, 12), dtype=np.int32)
    volume[:, 2:8, 3:9] = 1
    result = characterize_microstructure(
        volume,
        CharacterizationSettings(
            descriptor_types=("FFTCorrelations", "Variation"),
            limit_to=2,
            use_multigrid_descriptors=False,
            slice_mode="average",
            isotropic=False,
        ),
    )
    assert result.descriptors["FFTCorrelations"].values.shape[0] == 3
    assert result.descriptors["Variation"].values.shape[0] == 3


def test_descriptor_algebra_merge_and_interpolate():
    a = characterize_microstructure(_binary_2d(), CharacterizationSettings(descriptor_types=("VolumeFractions",), use_multigrid_descriptors=False))
    b_array = np.rot90(_binary_2d())
    b = characterize_microstructure(b_array, CharacterizationSettings(descriptor_types=("VolumeFractions",), use_multigrid_descriptors=False))
    merged = merge_characterizations((a, b), weights=(0.25, 0.75))
    series = interpolate_characterizations(a, b, 5)
    assert len(series) == 5
    assert merged.phase_ids == (0, 1)


def test_all_losses_return_finite_scalars():
    actual = np.asarray([1.0, 2.0, 4.0])
    target = np.asarray([0.5, 1.5, 3.0])
    for name in ("L1", "L2", "MSE", "RMS", "SSE"):
        assert np.isfinite(float(compute_loss(name, actual, target)))


def test_gradient_reconstruction_smoke():
    source = _binary_2d()
    chars = characterize_microstructure(
        source,
        CharacterizationSettings(
            descriptor_types=("VolumeFractions", "Variation"),
            use_multigrid_descriptors=False,
        ),
    )
    result = reconstruct_microstructure(
        chars,
        source.shape,
        settings=ReconstructionSettings(
            descriptor_types=("VolumeFractions", "Variation"),
            descriptor_weights=(20.0, 1.0),
            optimizer_type="Adam",
            max_iter=8,
            convergence_data_steps=1,
            use_multigrid_descriptors=False,
            seed=4,
        ),
    )
    assert result.labels.shape == source.shape
    assert set(np.unique(result.labels)) <= {0, 1}
    assert result.probabilities is not None
    assert np.allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-8)


def test_simulated_annealing_preserves_phase_counts():
    source = _binary_2d()
    chars = characterize_microstructure(
        source,
        CharacterizationSettings(
            descriptor_types=("VolumeFractions",), use_multigrid_descriptors=False
        ),
    )
    result = reconstruct_microstructure(
        chars,
        source.shape,
        settings=ReconstructionSettings(
            descriptor_types=("VolumeFractions",),
            optimizer_type="SimulatedAnnealing",
            max_iter=10,
            convergence_data_steps=2,
            use_multigrid_descriptors=False,
            seed=9,
        ),
    )
    assert np.count_nonzero(result.labels == 1) == np.count_nonzero(source == 1)
