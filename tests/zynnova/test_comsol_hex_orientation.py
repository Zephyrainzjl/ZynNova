from __future__ import annotations

from pathlib import Path

import numpy as np

from zynnova.zynmorph import (
    MicrostructureVolume,
    audit_comsol_hex8_connectivity,
    audit_comsol_hex8_topology,
    export_voxel_comsol_mphtxt,
    inspect_comsol_mphtxt,
)


def _mesh_elements(path: Path, type_number: int = 0) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = lines.index(f"# Type #{type_number}")
    elements_marker = lines.index("# Elements", marker)
    count_line = lines[elements_marker - 1]
    count = int(count_line.split()[0])
    rows = [np.fromstring(line, sep=" ", dtype=np.int64) for line in lines[elements_marker + 1 : elements_marker + 1 + count]]
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.int64)


def _structured_nodes(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> np.ndarray:
    nx, ny, nz = shape
    i, j, k = np.meshgrid(
        np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij"
    )
    return np.column_stack(
        (
            spacing[0] * i.ravel(order="C"),
            spacing[1] * j.ravel(order="C"),
            spacing[2] * k.ravel(order="C"),
        )
    )


def test_comsol_tensor_hex_order_has_positive_jacobians_and_opposite_face_owners() -> None:
    audit = audit_comsol_hex8_topology(
        (2, 2, 2), spacing=(2.25e-7, 4.5e-7, 2.25e-7)
    )
    assert audit.valid
    assert audit.nonpositive_jacobians == 0
    assert audit.same_side_shared_faces == 0
    assert audit.overconnected_faces == 0
    assert audit.expected_shared_faces == 12
    assert audit.shared_faces == 12
    assert audit.missing_shared_faces == 0


def test_legacy_vtk_hex_order_is_rejected_by_the_same_audit() -> None:
    shape = (2, 1, 1)
    points = _structured_nodes(shape, (1.0, 1.0, 1.0))
    # Correct COMSOL tensor local order is 000,100,010,110,001,101,011,111.
    correct = np.asarray(
        [
            [0, 4, 2, 6, 1, 5, 3, 7],
            [4, 8, 6, 10, 5, 9, 7, 11],
        ],
        dtype=np.int64,
    )
    legacy_vtk = correct[:, [0, 1, 3, 2, 4, 5, 7, 6]]

    good = audit_comsol_hex8_connectivity(points, correct)
    bad = audit_comsol_hex8_connectivity(points, legacy_vtk)
    assert good.valid and good.shared_faces == 1
    assert not bad.valid
    assert bad.nonpositive_jacobians == 2
    assert bad.shared_faces == 0


def test_exported_hex8_uses_comsol_order_at_reported_physical_scale(tmp_path: Path) -> None:
    labels = np.zeros((2, 2, 2), dtype=np.int32)  # z, y, x
    labels[:, :, 1] = 1
    volume = MicrostructureVolume(
        labels=labels,
        voxel_size_m=(2.25e-7, 4.5e-7, 2.25e-7),
        phase_names={0: "electrolyte", 1: "active"},
    )
    target = tmp_path / "reported_coordinate_fixed_hex8.mphtxt"
    export_voxel_comsol_mphtxt(
        target,
        volume,
        element_type="hex8",
        prefer_native=False,
        validate_topology=True,
    )
    info = inspect_comsol_mphtxt(target)
    cells = _mesh_elements(target, 0)

    assert info.element_counts["hex"] == 8
    assert np.array_equal(cells[0], [0, 9, 3, 12, 1, 10, 4, 13])
    assert np.array_equal(cells[4], [9, 18, 12, 21, 10, 19, 13, 22])
    assert set(cells[0, [1, 3, 5, 7]]) == set(cells[4, [0, 2, 4, 6]])

    text = target.read_text(encoding="utf-8")
    # The coordinate reported by COMSOL is explicitly represented and is no
    # longer attached to a cyclic/VTK-local Hex8 ordering.
    assert "2.2499999999999999e-07 4.4999999999999998e-07 2.2499999999999999e-07" in text or "2.25e-07 4.5e-07 2.25e-07" in text


def test_domain_only_hex8_diagnostic_still_has_valid_volume_cells(tmp_path: Path) -> None:
    volume = MicrostructureVolume(
        labels=np.asarray([[[0, 1]], [[0, 1]]], dtype=np.int32),
        voxel_size_m=2.25e-7,
        phase_names={0: "pore", 1: "solid"},
    )
    target = tmp_path / "domain_elements_only.mphtxt"
    export_voxel_comsol_mphtxt(
        target,
        volume,
        element_type="hex8",
        include_exterior_boundaries=False,
        include_material_interfaces=False,
        prefer_native=False,
        validate_topology=True,
    )
    info = inspect_comsol_mphtxt(target)
    assert info.element_counts == {"hex": 4}
    assert "1 # number of element types" in target.read_text(encoding="utf-8")
