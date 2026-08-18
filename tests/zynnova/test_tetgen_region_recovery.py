from __future__ import annotations

import numpy as np
import pytest

from zynnova.core.exceptions import GeometryError
from zynnova.zynmorph import TETGEN_REGION_RECOVERY_API, recover_tetgen_material_regions


def test_recovers_tetgen_automatic_topology_regions_to_physical_materials():
    # Four tetrahedra in a face-connected chain. Raw TetGen regions 4 and 5
    # emulate automatically generated facet-bounded topology attributes.
    tets = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 4, 5],
            [0, 4, 5, 6],
        ],
        dtype=np.int64,
    )
    raw = np.asarray([2, 1, 4, 5], dtype=np.int32)
    trifaces = np.asarray(
        [
            [0, 1, 2],  # electrolyte(2) <-> active(1)
            [0, 1, 4],  # active(1) <-> auto compartment(4 => electrolyte)
            [0, 4, 5],  # auto electrolyte(4) <-> auto CBD(5)
        ],
        dtype=np.int64,
    )
    markers = np.asarray([10, 10, 20], dtype=np.int32)

    mapped, report = recover_tetgen_material_regions(
        tets,
        raw,
        trifaces,
        markers,
        {
            10: (1, 2),
            20: (3, 2),
        },
        requested_material_ids=(1, 2, 3),
    )

    assert mapped.tolist() == [2, 1, 2, 3]
    assert report.raw_to_material == {1: 1, 2: 2, 4: 2, 5: 3}
    assert report.automatic_region_ids == (4, 5)
    assert report.recovered_automatic_regions == 2
    assert report.valid


def test_boundary_marker_can_anchor_an_automatic_raw_region():
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    raw = np.asarray([7], dtype=np.int32)
    trifaces = np.asarray([[0, 1, 2]], dtype=np.int64)
    markers = np.asarray([9], dtype=np.int32)

    mapped, report = recover_tetgen_material_regions(
        tets,
        raw,
        trifaces,
        markers,
        {9: (2, None)},
        requested_material_ids=(2,),
    )

    assert mapped.tolist() == [2]
    assert report.raw_to_material == {7: 2}
    assert report.boundary_anchors == 1


def test_unresolved_automatic_region_is_not_guessed_from_its_integer_id():
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    raw = np.asarray([8], dtype=np.int32)

    with pytest.raises(GeometryError, match="no marked subfaces"):
        recover_tetgen_material_regions(
            tets,
            raw,
            np.empty((0, 3), dtype=np.int64),
            np.empty(0, dtype=np.int32),
            {},
            requested_material_ids=(1, 2, 3),
        )


def test_identity_fast_path_keeps_material_ids_unchanged():
    tets = np.asarray(
        [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 4, 5]], dtype=np.int64
    )
    raw = np.asarray([1, 2, 3], dtype=np.int32)

    mapped, report = recover_tetgen_material_regions(
        tets,
        raw,
        np.empty((0, 3), dtype=np.int64),
        np.empty(0, dtype=np.int32),
        {},
        requested_material_ids=(1, 2, 3),
    )

    assert np.array_equal(mapped, raw)
    assert report.automatic_region_ids == ()
    assert report.raw_to_material == {1: 1, 2: 2, 3: 3}



def test_recovers_exact_reported_raw_attributes_1_through_8():
    # Regression for the user-observed native output:
    # requested materials = 1,2,3; raw TetGen attributes = 1..8.
    # Tetrahedra are arranged as a face-connected chain.  Marked PLC
    # interfaces constrain every automatic topology attribute to one of the
    # three physical materials.
    tets = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 4, 5],
            [0, 4, 5, 6],
            [0, 5, 6, 7],
            [0, 6, 7, 8],
            [0, 7, 8, 9],
            [0, 8, 9, 10],
        ],
        dtype=np.int64,
    )
    raw = np.asarray([2, 4, 5, 6, 7, 8, 2, 1], dtype=np.int32)

    # Shared chain faces between consecutive tetrahedra.
    trifaces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 8, 9],
        ],
        dtype=np.int64,
    )

    # Material sequence expected along the chain:
    # 2,1,3,2,1,3,2,1.
    markers = np.asarray([10, 20, 30, 10, 20, 30, 10], dtype=np.int32)
    marker_pairs = {
        10: (1, 2),
        20: (1, 3),
        30: (2, 3),
    }

    mapped, report = recover_tetgen_material_regions(
        tets,
        raw,
        trifaces,
        markers,
        marker_pairs,
        requested_material_ids=(1, 2, 3),
    )

    assert mapped.tolist() == [2, 1, 3, 2, 1, 3, 2, 1]
    assert report.automatic_region_ids == (4, 5, 6, 7, 8)
    assert report.raw_to_material == {
        1: 1,
        2: 2,
        4: 1,
        5: 3,
        6: 2,
        7: 1,
        8: 3,
    }
    assert report.recovered_automatic_regions == 5
    assert set(np.unique(mapped).tolist()) == {1, 2, 3}
    assert report.valid


def test_region_recovery_api_marker():
    assert TETGEN_REGION_RECOVERY_API == "facet-material-v1"
