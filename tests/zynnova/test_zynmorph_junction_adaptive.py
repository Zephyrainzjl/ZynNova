from __future__ import annotations

import numpy as np

from zynnova.zynmorph import (
    CBDSettings,
    ChannelSettings,
    ElectrodeComposition,
    ElectrodeSynthesisConfig,
    IrregularMeshPolicy,
    PackingSettings,
    ParticleDistribution,
    generate_particle_electrode,
)
from zynnova.zynmorph.surface import (
    count_nonmanifold_voxel_edges,
    regularize_nonmanifold_junctions,
)


def _mcs_style_electrode():
    composition = ElectrodeComposition(
        active_mass_fraction=0.92,
        carbon_mass_fraction=0.04,
        binder_mass_fraction=0.04,
        active_density_g_cm3=2.26,
        carbon_density_g_cm3=1.85,
        binder_density_g_cm3=1.60,
        active_specific_capacity_mAh_g=372.0,
    )
    config = ElectrodeSynthesisConfig(
        shape_zyx=(24, 28, 30),
        voxel_size_m=0.75e-6,
        active_volume_fraction=0.55,
        seed=20260818,
        padding_voxels=(3, 3, 4),
        crop_after_generation=True,
        particle_distribution=ParticleDistribution(
            geometry="mixed",
            median_diameter_vox=6.0,
            lognormal_sigma=0.22,
            minimum_diameter_vox=3.0,
            maximum_diameter_vox=10.0,
            sphere_fraction=0.40,
            axis_ratio_ranges=((0.60, 1.45), (0.60, 1.45)),
            angle_tolerance_degrees=30.0,
        ),
        packing=PackingSettings(
            method="rsa",
            boundary_mode="extended",
            overlap_fraction=0.12,
            max_attempts_per_particle=500,
        ),
        cbd=CBDSettings(
            method="mixed",
            target_volume_fraction=0.08,
            nanoporosity=0.35,
            correlation_length_vox=2.0,
            mixed_bridge_fraction=0.55,
        ),
        channel=ChannelSettings(enabled=True, radius_vox=1.7, phase=0),
        individual_particle_labels=True,
        composition=composition,
    )
    return generate_particle_electrode(config)


def test_fixed_half_percent_budget_is_insufficient_for_complex_mcs_style_volume():
    electrode = _mcs_style_electrode()
    before = count_nonmanifold_voxel_edges(electrode.volume.labels)
    assert before > 500

    _, report = regularize_nonmanifold_junctions(
        electrode.volume,
        maximum_changed_fraction=0.005,
        adaptive_budget=False,
        hard_maximum_changed_fraction=0.005,
        strict=False,
    )
    assert not report.converged
    assert report.ambiguous_edges_after > 0
    assert report.termination_reason == "change-budget-exhausted"


def test_adaptive_budget_converges_without_large_phase_fraction_drift():
    electrode = _mcs_style_electrode()
    repaired, report = regularize_nonmanifold_junctions(electrode.volume, strict=True)

    assert report.converged
    assert report.ambiguous_edges_after == 0
    assert report.budget_expansions >= 1
    assert report.changed_fraction > report.initial_change_budget_fraction
    assert report.changed_fraction <= report.hard_change_budget_fraction
    assert report.maximum_phase_fraction_drift < 0.01
    assert count_nonmanifold_voxel_edges(repaired.labels) == 0
    assert set(map(int, np.unique(repaired.labels))) == set(
        map(int, np.unique(electrode.volume.labels))
    )


def test_irregular_policy_enables_adaptive_junction_budget_by_default():
    electrode = _mcs_style_electrode()
    config = IrregularMeshPolicy().to_tetgen_config(electrode.volume)

    assert config.junction_adaptive_budget is True
    assert config.junction_maximum_changed_fraction == 0.005
    assert config.junction_hard_maximum_changed_fraction == 0.05
    assert config.junction_budget_growth_factor == 2.0
