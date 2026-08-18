from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zynnova.zynmorph.microstructure import (
    CBDSettings,
    ChannelSettings,
    ElectrodeComposition,
    ElectrodeSynthesisConfig,
    PackingSettings,
    ParticleDistribution,
    electrode_volume_targets_from_composition,
    generate_particle_electrode,
    load_microstructure,
)


def _config(method="rsa", cbd="bridge", geometry="mixed", boundary="contained"):
    return ElectrodeSynthesisConfig(
        shape_zyx=(18, 18, 18),
        active_volume_fraction=0.28,
        seed=13,
        particle_distribution=ParticleDistribution(
            geometry=geometry,
            particle_count=10,
            median_diameter_vox=5.0,
            lognormal_sigma=0.18,
            minimum_diameter_vox=3.5,
            maximum_diameter_vox=7.0,
        ),
        packing=PackingSettings(
            method=method,
            boundary_mode=boundary,
            overlap_fraction=0.20,
            max_attempts_per_particle=80,
        ),
        cbd=CBDSettings(
            method=cbd,
            target_volume_fraction=0.04,
            nanoporosity=0.20,
            correlation_length_vox=1.5,
        ),
    )


@pytest.mark.parametrize("geometry", ["sphere", "ellipsoid", "mixed"])
def test_particle_geometry_modes(geometry):
    result = generate_particle_electrode(_config(geometry=geometry))
    assert np.isclose(result.statistics.active_fraction, 0.28, atol=2 / result.volume.labels.size)
    assert len(result.particles) <= 10
    assert result.particle_labels.shape == result.volume.shape


@pytest.mark.parametrize("method", ["rsa", "gravity", "pseudo_gravity", "electrostatic"])
def test_packing_modes(method):
    result = generate_particle_electrode(_config(method=method))
    assert result.volume.phases == (0, 1, 2)
    assert result.statistics.particle_count > 0


@pytest.mark.parametrize("boundary", ["contained", "extended", "periodic"])
def test_boundary_modes(boundary):
    result = generate_particle_electrode(_config(boundary=boundary))
    assert result.volume.labels.shape == (18, 18, 18)


@pytest.mark.parametrize("method", ["bridge", "mistry", "random", "blob", "mixed", "none"])
def test_all_cbd_modes(method):
    cfg = _config(cbd=method)
    if method == "none":
        cfg = replace(cfg, cbd=replace(cfg.cbd, target_volume_fraction=0.0))
    result = generate_particle_electrode(cfg)
    if method == "none":
        assert result.statistics.cbd_geometric_fraction == 0.0
    else:
        assert result.statistics.cbd_geometric_fraction > 0.0


def test_channel_carving_and_individual_labels():
    cfg = replace(
        _config(),
        channel=ChannelSettings(enabled=True, radius_vox=2.0, phase=0),
        individual_particle_labels=True,
    )
    result = generate_particle_electrode(cfg)
    assert np.count_nonzero(result.particle_labels >= 1000) > 0
    center = result.volume.labels.shape[0] // 2
    assert np.all(result.volume.labels[center, center, :] == 0)


def test_composition_targets_and_mass_loading():
    composition = ElectrodeComposition(active_specific_capacity_mAh_g=372.0)
    targets = electrode_volume_targets_from_composition(
        composition, total_porosity=0.35, cbd_nanoporosity=0.476
    )
    assert 0 < targets["active_volume_fraction"] < 1
    cfg = replace(_config(), composition=composition)
    result = generate_particle_electrode(cfg)
    assert result.statistics.mass_loading_mg_cm2 is not None
    assert result.statistics.areal_capacity_mAh_cm2 is not None


def test_hdf5_and_vtk_export(tmp_path):
    result = generate_particle_electrode(_config())
    outputs = result.export(tmp_path, formats=("h5", "vtk", "npz"))
    assert outputs["h5"].is_file()
    assert outputs["vtk"].is_file()
    assert np.array_equal(load_microstructure(outputs["h5"]), result.volume.labels)
