from __future__ import annotations

import numpy as np
import pytest

from zynnova.core.exceptions import GeometryError
from zynnova.zynmorph import (
    MicrostructureVolume,
    TetGenMeshingConfig,
    audit_multiphase_plc,
    build_region_seeds,
    count_nonmanifold_voxel_edges,
    extract_multiphase_plc,
    mesh_microstructure,
    regularize_nonmanifold_junctions,
    smooth_multiphase_plc,
)
from zynnova.zynmorph.tetgen import (
    TetGenNativeStatus,
    _normalize_tetgen_input,
    mesh_microstructure_tetgen,
)


def _rounded_inclusion() -> MicrostructureVolume:
    labels = np.zeros((8, 9, 10), dtype=np.int32)
    z, y, x = np.indices(labels.shape)
    labels[((z - 3.5) / 2.5) ** 2 + ((y - 4.0) / 3.0) ** 2 + ((x - 4.5) / 3.5) ** 2 <= 1.0] = 1
    labels[(z <= 1) & (x >= 3) & (x <= 6)] = 2
    return MicrostructureVolume(
        labels=labels,
        voxel_size_m=(2.25e-7, 3.00e-7, 4.50e-7),
        origin_m=(1.0e-6, 2.0e-6, 3.0e-6),
        phase_names={0: "pore", 1: "active", 2: "binder"},
    )


def test_global_multiphase_plc_is_closed_once_per_interface() -> None:
    volume = _rounded_inclusion()
    assert count_nonmanifold_voxel_edges(volume.labels) == 0

    plc = extract_multiphase_plc(volume, strict=True)
    audit = audit_multiphase_plc(plc)

    assert audit.valid
    assert audit.degenerate_faces == 0
    assert audit.duplicate_faces == 0
    assert audit.open_region_edges == 0
    assert audit.nonmanifold_region_edges == 0
    assert audit.orientation_conflicts == 0
    assert set(plc.regions) == {0, 1, 2}
    assert np.all(plc.left_regions != plc.right_regions)
    assert any(name.startswith("interface_") for name in plc.marker_names.values())


def test_topology_safe_surface_smoothing_is_bounded_and_preserves_shells() -> None:
    volume = _rounded_inclusion()
    original = extract_multiphase_plc(volume, strict=True)
    maximum = 0.30 * min(volume.voxel_size_m)
    smoothed = smooth_multiphase_plc(
        original,
        iterations=5,
        maximum_displacement_m=maximum,
    )

    displacement = np.linalg.norm(smoothed.vertices - original.vertices, axis=1)
    assert float(displacement.max()) <= maximum * (1.0 + 1.0e-12)
    assert np.array_equal(smoothed.triangles, original.triangles)
    assert np.array_equal(smoothed.facet_markers, original.facet_markers)
    assert np.allclose(smoothed.vertices[original.fixed_vertices], original.vertices[original.fixed_vertices])
    assert audit_multiphase_plc(smoothed).valid
    assert np.count_nonzero(displacement) > 0


def test_checkerboard_edge_contact_is_rejected_then_minimally_regularized() -> None:
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1, 1, 2] = 1
    labels[2, 2, 2] = 1
    volume = MicrostructureVolume(labels=labels, phase_names={0: "pore", 1: "solid"})

    assert count_nonmanifold_voxel_edges(labels) == 1
    with pytest.raises(GeometryError, match="not a closed manifold"):
        extract_multiphase_plc(volume, strict=True)

    repaired, report = regularize_nonmanifold_junctions(
        volume,
        maximum_changed_fraction=0.05,
        minimum_phase_voxels=1,
        strict=True,
    )
    assert report.converged
    assert report.ambiguous_edges_before == 1
    assert report.ambiguous_edges_after == 0
    assert report.changed_voxels == 1
    assert np.count_nonzero(repaired.labels != labels) == 1
    assert audit_multiphase_plc(extract_multiphase_plc(repaired, strict=True)).valid


def test_region_seed_is_created_for_every_six_connected_component() -> None:
    labels = np.zeros((7, 7, 7), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    labels[4:6, 4:6, 4:6] = 1
    volume = MicrostructureVolume(
        labels=labels,
        voxel_size_m=(2.0e-7, 3.0e-7, 4.0e-7),
        origin_m=(1.0e-6, 2.0e-6, 3.0e-6),
    )
    config = TetGenMeshingConfig(
        global_maximum_tetra_volume_m3=4.0e-21,
        phase_maximum_tetra_volume_m3={1: 8.0e-22},
        smoothing_iterations=0,
    )
    seeds = build_region_seeds(volume, config)

    solid = [item for item in seeds if item.phase == 1]
    pore = [item for item in seeds if item.phase == 0]
    assert len(solid) == 2
    assert len(pore) == 1
    assert {item.component_voxels for item in solid} == {8}
    assert all(item.maximum_tetra_volume_m3 == pytest.approx(8.0e-22) for item in solid)
    assert pore[0].maximum_tetra_volume_m3 == pytest.approx(4.0e-21)


def test_si_coordinate_normalization_scales_all_size_controls() -> None:
    points = np.asarray(
        [
            [2.0e-6, 3.0e-6, 4.0e-6],
            [5.0e-6, 3.0e-6, 4.0e-6],
            [2.0e-6, 7.0e-6, 4.0e-6],
            [2.0e-6, 3.0e-6, 9.0e-6],
        ],
        dtype=np.float64,
    )
    seeds = np.asarray([[3.0e-6, 4.0e-6, 5.0e-6, 7.0, 1.25e-18]])
    facets = np.asarray([[9.0, 2.5e-12]])
    zones = np.asarray([[3.0e-6, 4.0e-6, 5.0e-6, 1.0e-6, 8.0e-20]])

    p, s, f, z, transform = _normalize_tetgen_input(points, seeds, facets, zones)
    offset, scale = transform

    assert scale == pytest.approx(5.0e-6)
    assert np.allclose(offset, points.min(axis=0))
    assert np.allclose(p.min(axis=0), 0.0)
    assert float(p.max()) == pytest.approx(1.0)
    assert s[0, 3] == pytest.approx(7.0)
    assert s[0, 4] == pytest.approx(seeds[0, 4] / scale**3)
    assert f[0, 0] == pytest.approx(9.0)
    assert f[0, 1] == pytest.approx(facets[0, 1] / scale**2)
    assert z[0, 3] == pytest.approx(zones[0, 3] / scale)
    assert z[0, 4] == pytest.approx(zones[0, 4] / scale**3)


def test_structured_backend_is_explicit_and_tetgen_has_no_silent_fallback(monkeypatch) -> None:
    volume = _rounded_inclusion()
    structured = mesh_microstructure(volume, method="structured", maximum_tetrahedra=10_000)
    assert structured.backend == "structured-six-tets-per-voxel"
    assert structured.mesh.n_cells == volume.labels.size * 6

    unavailable = TetGenNativeStatus(
        available=False,
        version="TetGen 1.6.0",
        reason="test extension absent",
        module_path=None,
        vendored_source_path=None,
        license="AGPL-3.0-or-later",
    )
    monkeypatch.setattr("zynnova.zynmorph.tetgen.tetgen_native_status", lambda: unavailable)
    with pytest.raises(RuntimeError, match="native extension is unavailable"):
        mesh_microstructure_tetgen(
            volume,
            config=TetGenMeshingConfig(smoothing_iterations=0),
            maximum_tetrahedra=100_000,
        )
