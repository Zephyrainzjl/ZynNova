from __future__ import annotations

import numpy as np
import pytest

from zynnova.zynmorph import (
    MicrostructureVolume,
    TetGenMeshingConfig,
    export_comsol_mphtxt,
    export_fem_mesh,
    extract_multiphase_plc,
    inspect_comsol_mphtxt,
    lock_plc_interfaces,
    mesh_microstructure,
    smooth_multiphase_plc,
)


def _tracking_label_volume() -> MicrostructureVolume:
    labels = np.zeros((4, 5, 8), dtype=np.int32)
    # Two positive particles with distinct tracking IDs.
    labels[1:3, 1:3, 1:3] = 101
    labels[1:3, 3:5, 2:4] = 102
    # Two negative particles with distinct tracking IDs.
    labels[1:3, 1:3, 5:7] = 201
    labels[1:3, 3:5, 6:8] = 202
    return MicrostructureVolume(
        labels=labels,
        voxel_size_m=1.0e-6,
        phase_names={
            0: "electrolyte",
            101: "positive_particle_01",
            102: "positive_particle_02",
            201: "negative_particle_01",
            202: "negative_particle_02",
        },
    )


def test_material_region_remap_collapses_disconnected_particle_ids() -> None:
    source = _tracking_label_volume()
    remapped = source.remap_regions(
        {101: 1, 102: 1, 201: 3, 202: 3},
        region_names={0: "electrolyte", 1: "positive_active", 3: "negative_active"},
    )

    assert remapped.phases == (0, 1, 3)
    assert remapped.phase_names == {
        0: "electrolyte",
        1: "positive_active",
        3: "negative_active",
    }
    assert np.count_nonzero(remapped.labels == 1) == (
        np.count_nonzero(source.labels == 101) + np.count_nonzero(source.labels == 102)
    )
    assert np.count_nonzero(remapped.labels == 3) == (
        np.count_nonzero(source.labels == 201) + np.count_nonzero(source.labels == 202)
    )


def test_mesh_microstructure_applies_material_region_map_before_meshing(tmp_path) -> None:
    source = _tracking_label_volume()
    fem = mesh_microstructure(
        source,
        method="structured",
        material_region_map={101: 1, 102: 1, 201: 3, 202: 3},
        material_region_names={0: "electrolyte", 1: "positive_active", 3: "negative_active"},
        maximum_tetrahedra=10_000,
    )

    assert set(map(int, np.unique(fem.mesh.cell_regions))) == {0, 1, 3}
    assert fem.mesh.region_names[1] == "positive_active"
    assert fem.mesh.region_names[3] == "negative_active"

    exported = export_fem_mesh(
        fem,
        tmp_path,
        formats=("mphtxt",),
        comsol_options={"include_boundary_triangles": True},
        comsol_domain_selections={
            "all_positive_particles": (1,),
            "all_negative_particles": (3,),
        },
    )
    info = inspect_comsol_mphtxt(exported.exports["mphtxt"])
    selections = {(item.label, item.dimension) for item in info.selections}
    assert ("all_positive_particles", 3) in selections
    assert ("all_negative_particles", 3) in selections
    # The writer creates one COMSOL domain entity per final material ID,
    # irrespective of how many disconnected components have that material.
    assert set(info.geometric_entity_ids["tet"]) == {1, 2, 3}


def test_comsol_historical_triangle_option_is_normalized_and_conflicts_are_rejected(tmp_path) -> None:
    volume = MicrostructureVolume(
        labels=np.asarray([[[0, 1], [0, 1]]], dtype=np.int32),
        voxel_size_m=1.0e-6,
        phase_names={0: "elyte", 1: "active"},
    )
    fem = mesh_microstructure(volume, method="structured", maximum_tetrahedra=100)

    exported = export_fem_mesh(
        fem,
        tmp_path / "ok",
        formats=("mphtxt",),
        comsol_options={
            "include_boundary_triangles": True,
            "include_interface_triangles": True,
        },
    )
    info = inspect_comsol_mphtxt(exported.exports["mphtxt"])
    assert info.element_counts["tri"] > 0

    direct = tmp_path / "direct_alias.mphtxt"
    direct_report = export_comsol_mphtxt(
        direct,
        fem.mesh,
        include_boundary_triangles=True,
        include_interface_triangles=True,
    )
    assert direct_report.triangle_count > 0

    legacy = tmp_path / "legacy_aliases.mphtxt"
    legacy_report = export_comsol_mphtxt(
        legacy,
        fem.mesh,
        include_boundary_triangles=True,
        include_material_interfaces=True,
        include_exterior_boundaries=True,
    )
    assert legacy_report.triangle_count > 0

    with pytest.raises(ValueError, match="conflicting COMSOL options"):
        export_fem_mesh(
            fem,
            tmp_path / "conflict",
            formats=("mphtxt",),
            comsol_options={
                "include_boundary_triangles": True,
                "include_boundaries": False,
            },
        )


def test_locked_material_interfaces_are_unchanged_by_plc_smoothing() -> None:
    labels = np.zeros((5, 6, 9), dtype=np.int32)
    labels[:, :, :3] = 1
    labels[:, :, 3:6] = 0
    labels[:, :, 6:] = 2
    # Add curved internal inclusions without touching the exact layer planes.
    z, y, x = np.indices(labels.shape)
    labels[((z - 2.0) ** 2 + (y - 2.5) ** 2 + (x - 1.3) ** 2) < 2.0] = 4
    labels[((z - 2.0) ** 2 + (y - 2.5) ** 2 + (x - 7.3) ** 2) < 2.0] = 5
    volume = MicrostructureVolume(labels=labels, voxel_size_m=1.0e-6)

    plc = extract_multiphase_plc(volume, strict=True)
    locked = lock_plc_interfaces(plc, ((0, 1), (0, 2)))
    fixed_before = locked.vertices[locked.fixed_vertices].copy()
    smoothed = smooth_multiphase_plc(
        locked,
        iterations=4,
        relaxation=0.30,
        maximum_displacement_m=0.25e-6,
    )

    assert np.array_equal(smoothed.vertices[locked.fixed_vertices], fixed_before)
    assert smoothed.metadata["locked_interface_pairs"] == ((0, 1), (0, 2))
    assert smoothed.metadata["locked_interface_faces"] > 0


def test_tetgen_config_normalizes_fixed_interface_pairs() -> None:
    config = TetGenMeshingConfig(
        fixed_interface_pairs=((3, 1), (1, 3), (4, 0)),
        smoothing_iterations=0,
    )
    assert config.fixed_interface_pairs == ((1, 3), (0, 4))


def test_mesh_unstructured_regions_is_a_tetgen_only_volume_dispatch(monkeypatch) -> None:
    from zynnova.zynmorph import mesh_unstructured_regions

    source = _tracking_label_volume()
    sentinel = object()
    captured = {}

    def fake_mesh(volume, **kwargs):
        captured["volume"] = volume
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("zynnova.zynmorph.meshing.mesh_microstructure", fake_mesh)
    result = mesh_unstructured_regions(
        source,
        tetgen_config={"smoothing_iterations": 0},
        material_region_map={101: 1, 102: 1, 201: 3, 202: 3},
        maximum_tetrahedra=12345,
    )

    assert result is sentinel
    assert captured["method"] == "tetgen"
    assert captured["maximum_tetrahedra"] == 12345
    assert captured["material_region_map"] == {101: 1, 102: 1, 201: 3, 202: 3}
