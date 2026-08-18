from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zynnova.geometry import TriangleMesh, VolumeMesh
from zynnova.zynmorph import (
    FreeformRegion,
    SurfaceShell,
    assemble_freeform_plc,
    export_comsol_mphtxt,
    load_comsol_tet4_mphtxt,
    orient_closed_surface,
    profile_reference_mesh,
    tetgen_config_from_reference,
)


def _outer_octahedron(scale: float = 1.0e-6) -> TriangleMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -3.0],
        ]
    ) * scale
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
        ], dtype=np.int64,
    )
    # Deliberately make two local orientations inconsistent.
    faces[[1, 5]] = faces[[1, 5]][:, [0, 2, 1]]
    return TriangleMesh(vertices=vertices, faces=faces)


def _inner_tetra(scale: float = 1.0e-6) -> TriangleMesh:
    vertices = np.array(
        [[0, 0, 1], [1, 0, -0.5], [-0.5, 0.8, -0.5], [-0.5, -0.8, -0.5]],
        dtype=float,
    ) * scale
    faces = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]], dtype=np.int64)
    return TriangleMesh(vertices=vertices, faces=faces)


def test_orient_closed_surface_is_scale_safe_and_outward() -> None:
    oriented = orient_closed_surface(_outer_octahedron(scale=1.0e-7))
    assert oriented.n_faces == 8
    assert oriented.metadata["signed_enclosed_volume_m3"] > 0.0
    assert oriented.metadata["oriented_closed_surface"] is True


def test_open_surface_is_rejected() -> None:
    mesh = _outer_octahedron()
    with pytest.raises(Exception, match="closed and 2-manifold"):
        orient_closed_surface(TriangleMesh(mesh.vertices, mesh.faces[:-1]))


def test_nested_arbitrary_shells_form_valid_multidomain_plc() -> None:
    plc = assemble_freeform_plc(
        (
            SurfaceShell(_outer_octahedron(), inside_region=1, name="wavy_outer"),
            SurfaceShell(_inner_tetra(), inside_region=2, outside_region=1, name="inclusion"),
        )
    )
    assert plc.regions == (1, 2)
    assert plc.metadata["shell_count"] == 2
    assert plc.triangles.shape == (12, 3)
    assert set(plc.marker_names.values()) == {"wavy_outer", "inclusion"}


def test_reference_mphtxt_roundtrip_and_profile(tmp_path: Path) -> None:
    nodes = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 1]], dtype=float
    ) * 1.0e-6
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    regions = np.array([1, 2], dtype=np.int32)
    mesh = VolumeMesh(nodes, tets, regions, {1: "left", 2: "right"})
    path = tmp_path / "reference.mphtxt"
    export_comsol_mphtxt(
        path,
        mesh,
        include_default_battery_selections=False,
        include_default_boundary_unions=False,
    )
    loaded = load_comsol_tet4_mphtxt(path)
    assert loaded.n_nodes == 5
    assert loaded.n_cells == 2
    assert loaded.cell_regions.tolist() == [1, 2]
    profile = profile_reference_mesh(loaded)
    assert profile.regions == (1, 2)
    assert profile.edge_length_quantiles_m[3] > 0.0
    assert profile.tetra_volume_quantiles_m3[3] > 0.0
    config = tetgen_config_from_reference(profile, volume_quantile=0.95)
    assert set(config.phase_maximum_tetra_volume_m3) == {1, 2}


def test_freeform_region_validates_seed_and_size() -> None:
    region = FreeformRegion(3, (1e-6, 2e-6, 3e-6), "electrolyte", 1e-18)
    assert region.region == 3
    assert region.maximum_tetra_volume_m3 == pytest.approx(1e-18)
    with pytest.raises(ValueError):
        FreeformRegion(1, (0.0, np.nan, 0.0))


def test_freeform_tetgen_wrapper_uses_native_plc_without_box_assumption(monkeypatch) -> None:
    import zynnova.zynmorph.freeform as freeform_module
    from zynnova.zynmorph import mesh_freeform_tetgen
    from zynnova.zynmorph.tetgen import TetGenNativeStatus

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    ) * 1e-6
    surface = TriangleMesh(
        vertices,
        np.array([[0,2,1], [0,1,3], [0,3,2], [1,2,3]], dtype=np.int64),
    )
    plc = assemble_freeform_plc((SurfaceShell(surface, 1, name="tetra_outer"),))

    class FakeNative:
        def tetrahedralize(self, points, facets, markers, seeds, holes, constraints, zones, *args):
            assert points.shape == (4, 3)
            assert facets.shape == (4, 3)
            assert seeds.shape == (1, 5)
            return {
                "points": np.asarray(points),
                "tetrahedra": np.array([[0, 1, 2, 3]], dtype=np.int64),
                "region_attributes": np.array([1.0]),
                "trifaces": np.asarray(facets),
                "triface_markers": np.asarray(markers),
                "switches": "pqaA",
                "version": "TetGen 1.6.0 fake-contract",
            }

    monkeypatch.setattr(
        freeform_module,
        "tetgen_native_status",
        lambda: TetGenNativeStatus(True, "TetGen 1.6.0", None, None, None, "AGPL"),
    )
    monkeypatch.setattr(freeform_module, "_load_tetgen_native", lambda: FakeNative())
    result = mesh_freeform_tetgen(
        plc,
        (FreeformRegion(1, (0.1e-6, 0.1e-6, 0.1e-6), "solid"),),
    )
    assert result.mesh.n_cells == 1
    assert result.mesh.metadata["rectangular_domain_assumed"] is False
    assert result.quality.fem_ready
    assert result.mesh.region_names == {1: "solid"}


def test_plc_from_volume_mesh_can_inspect_pinched_material_interfaces() -> None:
    from zynnova.zynmorph import plc_from_volume_mesh

    # Two same-material interface sheets touch along one edge: acceptable as
    # legacy-volume diagnostics, but not a strict watertight shell PLC.
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0, -1, 0], [0, 0, -1], [1, 1, 0], [1, -1, 0],
        ],
        dtype=float,
    ) * 1e-6
    tets = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [0, 1, 5, 2],
        ], dtype=np.int64,
    )
    # Alternating materials around edge (0, 1) produce a pinched interface.
    mesh = VolumeMesh(nodes, tets, np.array([1, 2, 1, 2], dtype=np.int32))
    with pytest.raises(Exception, match="topology audit"):
        plc_from_volume_mesh(mesh, strict=True)
    diagnostic = plc_from_volume_mesh(mesh, strict=False)
    audit = diagnostic.metadata["plc_audit"]
    assert not audit.valid
    assert audit.nonmanifold_region_edges > 0


def test_mesh_freeform_like_reference_transfers_size_without_box_geometry(monkeypatch) -> None:
    import zynnova.zynmorph.freeform as freeform_module
    from zynnova.zynmorph import mesh_freeform_like_reference
    from zynnova.zynmorph.tetgen import TetGenNativeStatus

    surface = _inner_tetra(scale=1.0e-6)
    shells = (SurfaceShell(surface, 7, name="arbitrary_outer"),)
    regions = (FreeformRegion(7, (0.0, 0.0, 0.0), "target"),)

    reference_nodes = np.array(
        [[0, 0, 0], [3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float
    ) * 1e-6
    reference = VolumeMesh(
        reference_nodes,
        np.array([[0, 1, 2, 3]], dtype=np.int64),
        np.array([4], dtype=np.int32),
    )

    captured = {}

    class FakeNative:
        def tetrahedralize(self, points, facets, markers, seeds, holes, constraints, zones, *args):
            captured["seeds"] = np.asarray(seeds).copy()
            return {
                "points": np.asarray(points),
                "tetrahedra": np.array([[0, 1, 2, 3]], dtype=np.int64),
                "region_attributes": np.array([7.0]),
                "trifaces": np.asarray(facets),
                "triface_markers": np.asarray(markers),
                "switches": "pqaA",
                "version": "TetGen 1.6.0 fake-contract",
            }

    monkeypatch.setattr(
        freeform_module,
        "tetgen_native_status",
        lambda: TetGenNativeStatus(True, "TetGen 1.6.0", None, None, None, "AGPL"),
    )
    monkeypatch.setattr(freeform_module, "_load_tetgen_native", lambda: FakeNative())

    result = mesh_freeform_like_reference(
        shells,
        regions,
        reference,
        region_map={7: 4},
        volume_quantile=0.95,
        linear_scale=0.5,
    )
    assert result.backend == "tetgen-1.6.0-freeform-plc"
    assert result.mesh.metadata["rectangular_domain_assumed"] is False
    assert captured["seeds"].shape == (1, 5)
    assert captured["seeds"][0, 4] > 0.0
    style = result.metadata["reference_mesh_style"]
    assert style["region_map"] == {7: 4}
    assert style["linear_scale"] == pytest.approx(0.5)
