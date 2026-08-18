from __future__ import annotations

import numpy as np

from zynnova.zynmorph.microstructure import (
    CharacterizationSettings,
    ReconstructionSettings,
    ElectrodeSynthesisConfig,
    ParticleDistribution,
    CBDSettings,
    PackingSettings,
    characterize,
    generate_particle_electrode,
    merge_directional,
    reconstruct,
    crop_structure_to_content,
    pad_structure_to_content,
    cut_electrode_empty_tail,
    translate_microstructure,
)


def checkerboard(n: int = 16) -> np.ndarray:
    y, x = np.indices((n, n))
    return ((x // 4 + y // 4) % 2).astype(np.int32)


def test_singlephase_characterization_and_gradient_reconstruction():
    labels = checkerboard(12)
    settings = CharacterizationSettings(
        descriptor_types=("VolumeFractions", "Variation"),
        use_multiphase=False,
        use_multigrid_descriptors=False,
        limit_to=4,
    )
    target = characterize(labels, settings)
    assert target.descriptors["VolumeFractions"].values.size == 1
    assert len(target.metadata["volume_fractions"]) == 2
    result = reconstruct(
        target,
        settings=ReconstructionSettings(
            descriptor_types=("VolumeFractions", "Variation"),
            descriptor_weights=(1.0, 2.0),
            optimizer_type="Adam",
            use_multiphase=False,
            use_multigrid_descriptors=False,
            max_iter=4,
            learning_rate=0.05,
            dtype="float64",
        ),
    )
    assert result.labels.shape == labels.shape
    assert set(np.unique(result.labels)).issubset({0, 1})
    assert np.allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-10)


def test_directional_merge_one_two_three_inputs_preserves_three_directions():
    a = checkerboard(12)
    b = np.rot90(a)
    cfg = CharacterizationSettings(
        descriptor_types=("Correlations", "Variation"),
        use_multigrid_descriptors=False,
        limit_to=3,
    )
    ca = characterize(a, cfg)
    cb = characterize(b, cfg)
    for inputs in ((ca,), (ca, cb), (ca, cb, ca)):
        merged = merge_directional(inputs)
        assert merged.metadata["directional_merge"] is True
        for descriptor in merged.descriptors.values():
            assert descriptor.values.shape[0] == 3


def test_directional_multigrid_coarse_to_fine_reconstruction_smoke():
    a = checkerboard(16)
    b = np.roll(a, 2, axis=0)
    cfg = CharacterizationSettings(
        descriptor_types=("VolumeFractions", "Variation"),
        use_multigrid_descriptors=True,
        multigrid_levels=2,
        limit_to=4,
    )
    target = merge_directional((characterize(a, cfg), characterize(b, cfg)))
    result = reconstruct(
        target,
        desired_shape=(16, 16, 16),
        settings=ReconstructionSettings(
            descriptor_types=("VolumeFractions", "Variation"),
            descriptor_weights=(1.0, 2.0),
            optimizer_type="Adam",
            use_multigrid_descriptors=True,
            use_multigrid_reconstruction=True,
            multigrid_levels=2,
            slice_mode="average",
            isotropic=False,
            max_iter=4,
            learning_rate=0.03,
            dtype="float64",
        ),
    )
    assert result.labels.shape == (16, 16, 16)
    assert result.metadata["multigrid_reconstruction_levels"] >= 2


def test_padding_and_cropping_particle_generation_uses_requested_final_shape():
    config = ElectrodeSynthesisConfig(
        shape_zyx=(16, 18, 20),
        active_volume_fraction=0.35,
        seed=11,
        padding_voxels=(3, 2, 4),
        crop_after_generation=True,
        particle_distribution=ParticleDistribution(
            geometry="mixed",
            median_diameter_vox=5.0,
            minimum_diameter_vox=3.0,
            maximum_diameter_vox=7.0,
            lognormal_sigma=0.15,
        ),
        packing=PackingSettings(method="rsa", max_attempts_per_particle=300),
        cbd=CBDSettings(method="random", target_volume_fraction=0.04),
    )
    result = generate_particle_electrode(config)
    assert result.volume.labels.shape == config.shape_zyx
    assert tuple(result.volume.metadata["packing_shape_zyx"]) == (22, 22, 28)
    # Active target is exact before CBD takes some pore volume; CBD never replaces active.
    assert np.isclose(result.statistics.active_fraction, config.active_volume_fraction, atol=1.0 / np.prod(config.shape_zyx))


def test_crop_pad_and_cut_helpers():
    x = np.zeros((10, 12, 14), dtype=np.int32)
    x[2:7, 3:9, 4:11] = 5
    cropped = crop_structure_to_content(x, background=0, margins_vox=(1, 1, 1))
    assert cropped.shape == (7, 8, 9)
    padded = pad_structure_to_content(x, background=0, margins_vox=(2, 1, 3))
    assert padded.shape == (9, 8, 13)
    tail = np.zeros((12, 6, 6), dtype=np.int32)
    tail[:7] = 1
    cut = cut_electrode_empty_tail(tail, axis=0, margin_vox=1)
    assert cut.shape[0] == 8


def test_periodic_translation_preserves_shape_and_phase_counts():
    labels = np.zeros((7, 9, 11), dtype=np.int32)
    labels[1:4, 2:6, 3:8] = 2
    shifted = translate_microstructure(labels, (2, -3, 4))
    assert shifted.shape == labels.shape
    assert np.array_equal(np.bincount(shifted.ravel()), np.bincount(labels.ravel()))
    assert np.array_equal(translate_microstructure(shifted, (-2, 3, -4)), labels)
