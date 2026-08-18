from __future__ import annotations

import numpy as np
import pytest

from zynnova.zynmorph import (
    IrregularMeshPolicy,
    MicrostructureVolume,
    equilateral_triangle_area_from_edge,
    infer_irregular_base_edge_length_m,
    regular_tetrahedron_volume_from_edge,
)


def test_irregular_policy_infers_physical_scale_and_builds_region_constraints() -> None:
    volume = MicrostructureVolume(
        labels=np.zeros((3, 4, 5), dtype=np.int32),
        voxel_size_m=(0.8e-6, 1.0e-6, 1.2e-6),
    )
    assert infer_irregular_base_edge_length_m(volume) == pytest.approx(1.6e-6)

    policy = IrregularMeshPolicy(
        region_edge_lengths_m={1: 1.0e-6, 2: 2.0e-6},
        interface_edge_lengths_m={(1, 2): 0.7e-6},
    )
    config = policy.to_tetgen_config(volume)
    assert config.global_maximum_tetra_volume_m3 == pytest.approx(
        regular_tetrahedron_volume_from_edge(1.6e-6)
    )
    assert config.phase_maximum_tetra_volume_m3[1] == pytest.approx(
        regular_tetrahedron_volume_from_edge(1.0e-6)
    )
    assert config.phase_maximum_tetra_volume_m3[2] == pytest.approx(
        regular_tetrahedron_volume_from_edge(2.0e-6)
    )
    assert config.facet_maximum_area_m2[(1, 2)] == pytest.approx(
        equilateral_triangle_area_from_edge(0.7e-6)
    )
    assert config.conforming_delaunay is True


def test_irregular_policy_rejects_nonphysical_sizes() -> None:
    with pytest.raises(ValueError):
        IrregularMeshPolicy(base_edge_length_m=0.0)
    with pytest.raises(ValueError):
        IrregularMeshPolicy(region_edge_lengths_m={1: -1.0})


def test_irregular_policy_forwards_locked_interfaces() -> None:
    volume = MicrostructureVolume(
        labels=np.zeros((3, 4, 5), dtype=np.int32),
        voxel_size_m=1.0e-6,
    )
    policy = IrregularMeshPolicy(
        fixed_interface_pairs=((3, 1), (1, 3), (0, 2)),
        smoothing_iterations=4,
    )
    config = policy.to_tetgen_config(volume)
    assert config.fixed_interface_pairs == ((1, 3), (0, 2))
    assert config.smoothing_iterations == 4


def test_mesh_complex_regions_forwards_material_collapse(monkeypatch) -> None:
    from zynnova.zynmorph import mesh_complex_regions

    volume = MicrostructureVolume(
        labels=np.asarray([[[1001, 1002, 2001, 2002]]], dtype=np.int32),
        voxel_size_m=1.0e-6,
        phase_names={
            1001: "p1",
            1002: "p2",
            2001: "n1",
            2002: "n2",
        },
    )
    sentinel = object()
    captured = {}

    def fake_dispatch(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("zynnova.zynmorph.meshing.mesh_unstructured_regions", fake_dispatch)

    result = mesh_complex_regions(
        volume,
        policy=IrregularMeshPolicy(base_edge_length_m=2.0e-6),
        material_region_map={1001: 1, 1002: 1, 2001: 3, 2002: 3},
        material_region_names={1: "positive_active", 3: "negative_active"},
        require_complete_region_map=True,
    )
    assert result is sentinel
    assert captured["material_region_map"] == {1001: 1, 1002: 1, 2001: 3, 2002: 3}
    assert captured["require_complete_region_map"] is True
    assert captured["tetgen_config"].global_maximum_tetra_volume_m3 > 0.0


def test_mesh_complex_regions_keeps_disconnected_freeform_components_one_material(monkeypatch) -> None:
    import zynnova.zynmorph.freeform as freeform_module
    from zynnova.geometry import TriangleMesh
    from zynnova.zynmorph import FreeformRegion, SurfaceShell, mesh_complex_regions
    from zynnova.zynmorph.tetgen import TetGenNativeStatus

    vertices = np.asarray(
        [[0, 0, 1], [1, 0, -0.5], [-0.5, 0.8, -0.5], [-0.5, -0.8, -0.5]],
        dtype=np.float64,
    ) * 1.0e-6
    faces = np.asarray(
        [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]],
        dtype=np.int64,
    )
    mesh_a = TriangleMesh(vertices=vertices, faces=faces)
    mesh_b = TriangleMesh(
        vertices=vertices + np.asarray([4.0e-6, 0.0, 0.0]),
        faces=faces,
    )
    shells = (
        SurfaceShell(mesh_a, inside_region=7, name="particle_a"),
        SurfaceShell(mesh_b, inside_region=7, name="particle_b"),
    )
    regions = (
        FreeformRegion(7, (0.0, 0.0, 0.0), "positive_active_component_1"),
        FreeformRegion(7, (4.0e-6, 0.0, 0.0), "positive_active_component_2"),
    )

    class FakeNative:
        def tetrahedralize(
            self,
            points,
            facets,
            markers,
            seeds,
            holes,
            constraints,
            zones,
            *args,
        ):
            assert seeds.shape == (2, 5)
            assert np.all(seeds[:, 3] == 7.0)
            return {
                "points": np.asarray(points),
                "tetrahedra": np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
                "region_attributes": np.asarray([7.0, 7.0]),
                "trifaces": np.asarray(facets),
                "triface_markers": np.asarray(markers),
                "switches": "pqaAu",
                "version": "TetGen 1.6.0 fake-contract",
            }

    monkeypatch.setattr(
        freeform_module,
        "tetgen_native_status",
        lambda: TetGenNativeStatus(True, "TetGen 1.6.0", None, None, None, "AGPL"),
    )
    monkeypatch.setattr(freeform_module, "_load_tetgen_native", lambda: FakeNative())

    result = mesh_complex_regions(
        shells,
        regions=regions,
        policy=IrregularMeshPolicy(base_edge_length_m=2.0e-6, smoothing_iterations=0),
    )
    assert np.unique(result.mesh.cell_regions).tolist() == [7]
    assert result.mesh.region_names == {7: "positive_active"}
