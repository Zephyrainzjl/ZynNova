from __future__ import annotations

from pathlib import Path

import numpy as np

from zynnova.zynmorph import (
    MicrostructureVolume,
    export_voxel_comsol_mphtxt,
    inspect_comsol_mphtxt,
    validate_comsol_hex_connectivity,
    validate_structured_hex_topology,
)
from zynnova.zynsim.core.general_mesh import voxel_to_general_mesh
from zynnova.zynsim.io import write_general_comsol_mphtxt
from zynnova.zynsim.io.comsol_topology import (
    COMSOL_HEX8_FROM_INTERNAL,
    structured_comsol_hex_rows,
)


def _first_connectivity_row(path: Path) -> tuple[int, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = lines.index("# Elements")
    return tuple(map(int, lines[marker + 1].split()))


def _structured_nodes(shape: tuple[int, int, int], spacing: float) -> np.ndarray:
    nx, ny, nz = shape
    x, y, z = np.meshgrid(
        np.arange(nx + 1) * spacing,
        np.arange(ny + 1) * spacing,
        np.arange(nz + 1) * spacing,
        indexing="ij",
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_old_vtk_hex_order_reproduces_reported_comsol_same_side_face() -> None:
    shape = (2, 2, 2)
    spacing = 4.5e-7
    nodes = _structured_nodes(shape, spacing)
    correct = structured_comsol_hex_rows(shape, 0, int(np.prod(shape)))
    # The permutation is self-inverse: applying it to COMSOL rows reconstructs
    # the previous VTK/cyclic rows that were written into MPHTXT unchanged.
    previous_rows = correct[:, COMSOL_HEX8_FROM_INTERNAL]

    previous = validate_comsol_hex_connectivity(nodes, previous_rows)
    repaired = validate_comsol_hex_connectivity(nodes, correct)

    reported_coordinate = (2.25e-7, 4.5e-7, 2.25e-7)
    assert not previous.valid
    assert previous.same_side_shared_faces > 0
    assert any(np.allclose(item, reported_coordinate) for item in previous.sample_failure_coordinates)
    assert repaired.valid
    assert repaired.shared_faces_checked == 12
    assert repaired.same_side_shared_faces == 0


def test_streamed_hex8_writes_comsol_tensor_product_node_order(tmp_path) -> None:
    volume = MicrostructureVolume(
        labels=np.asarray([[[1]], [[2]]], dtype=np.int32),  # z, y, x
        voxel_size_m=(3.0, 5.0, 7.0),
    )
    target = tmp_path / "fixed_hex.mphtxt"
    report = export_voxel_comsol_mphtxt(
        target,
        volume,
        element_type="hex8",
        prefer_native=False,
        include_exterior_boundaries=False,
        include_material_interfaces=False,
    )

    # Adapter transposes z,y,x -> x,y,z, so the streamed shape is (1,1,2).
    assert _first_connectivity_row(target) == tuple(
        structured_comsol_hex_rows((1, 1, 2), 0, 1)[0]
    )
    assert report.topology_validation is not None
    assert report.topology_validation.valid


def test_general_mesh_writer_converts_internal_vtk_hex_and_quad_order(tmp_path) -> None:
    mesh = voxel_to_general_mesh(
        np.ones((1, 1, 1), dtype=np.int32),
        voxel_size_m=(1.0, 2.0, 3.0),
        element_type="hex8",
        include_exterior_boundaries=True,
    )
    target = tmp_path / "general_fixed.mphtxt"
    write_general_comsol_mphtxt(target, mesh)
    assert _first_connectivity_row(target) == tuple(
        structured_comsol_hex_rows((1, 1, 1), 0, 1)[0]
    )


def test_no_domain_entity_index_diagnostic_mode(tmp_path) -> None:
    volume = MicrostructureVolume(
        labels=np.asarray([[[1, 2], [2, 1]]], dtype=np.int32),
        voxel_size_m=4.5e-7,
    )
    target = tmp_path / "without_domain_entities.mphtxt"
    report = export_voxel_comsol_mphtxt(
        target,
        volume,
        element_type="hex8",
        prefer_native=False,
        include_domain_entity_indices=False,
    )
    info = inspect_comsol_mphtxt(target)

    assert report.domain_entity_indices_written is False
    assert info.geometric_entity_ids.get("hex", ()) == ()
    assert not any(selection.dimension == 3 for selection in info.selections)
    assert any(selection.dimension == 2 for selection in info.selections)


def test_structured_topology_validator_is_scale_and_origin_safe() -> None:
    report = validate_structured_hex_topology(
        (5, 4, 3),
        spacing=(4.5e-7, 7.0e-8, 2.0e-6),
        origin=(1.1e-6, -3.0e-7, 8.0e-6),
    )
    assert report.valid
    assert report.minimum_jacobian > 0.0
    assert report.same_side_shared_faces == 0


def test_true_surface_only_diagnostic_contains_no_domain_elements(tmp_path) -> None:
    volume = MicrostructureVolume(
        labels=np.asarray(
            [
                [[1, 1], [1, 2]],
                [[1, 2], [2, 2]],
            ],
            dtype=np.int32,
        ),
        voxel_size_m=(4.5e-7, 2.25e-7, 7.5e-8),
    )
    target = tmp_path / "surface_only.mphtxt"
    report = export_voxel_comsol_mphtxt(
        target,
        volume,
        element_type="hex8",
        prefer_native=True,  # must be bypassed in this diagnostic mode
        include_domain_entity_indices=True,
        include_volume_elements=False,
        include_exterior_boundaries=True,
        include_material_interfaces=True,
    )
    info = inspect_comsol_mphtxt(target)

    assert report.diagnostic_mode == "surface-only"
    assert report.volume_elements_written is False
    assert report.domain_entity_indices_written is False
    assert report.native_backend is False
    assert info.element_counts.get("hex", 0) == 0
    assert info.element_counts.get("tet", 0) == 0
    assert info.element_counts.get("quad", 0) > 0
    assert not any(selection.dimension == 3 for selection in info.selections)
    assert any(selection.dimension == 2 for selection in info.selections)


def test_surface_only_requires_a_surface_block(tmp_path) -> None:
    volume = MicrostructureVolume(
        labels=np.ones((1, 1, 1), dtype=np.int32),
        voxel_size_m=1.0,
    )
    with np.testing.assert_raises_regex(ValueError, "surface-only"):
        export_voxel_comsol_mphtxt(
            tmp_path / "empty_surface.mphtxt",
            volume,
            include_volume_elements=False,
            include_exterior_boundaries=False,
            include_material_interfaces=False,
            prefer_native=False,
        )
