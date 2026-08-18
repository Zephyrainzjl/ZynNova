from __future__ import annotations

import numpy as np
import pytest

from zynnova.geometry import VolumeMesh
from zynnova.zynmorph import export_comsol_mphtxt


def _curved_like_single_tet() -> VolumeMesh:
    # None of the exterior triangular faces lies on xmin/xmax/ymin/ymax/zmin/zmax.
    # This reproduces the selection topology of a generic curved/free-form mesh:
    # the coordinate helper names exist transiently but each set is empty.
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.17, 0.11],
            [0.13, 1.0, 0.29],
            [0.31, 0.23, 1.0],
        ],
        dtype=np.float64,
    )
    return VolumeMesh(
        nodes=nodes,
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        cell_regions=np.asarray([1], dtype=np.int32),
        region_names={1: "active_material"},
    )


def test_default_coordinate_unions_skip_missing_freeform_sets(tmp_path):
    report = export_comsol_mphtxt(
        tmp_path / "freeform.mphtxt",
        _curved_like_single_tet(),
        include_coordinate_boundaries=True,
        include_default_boundary_unions=True,
        include_boundaries=True,
        verify=True,
    )

    assert "x_terminal_pair" not in report.boundary_selections
    assert "y_periodic_candidate_pair" not in report.boundary_selections
    assert "z_periodic_candidate_pair" not in report.boundary_selections
    assert "transverse_boundaries" not in report.boundary_selections
    assert report.path.is_file()


def test_user_boundary_union_is_still_strict(tmp_path):
    with pytest.raises(ValueError, match="references unknown sets"):
        export_comsol_mphtxt(
            tmp_path / "invalid_user_union.mphtxt",
            _curved_like_single_tet(),
            boundary_selections={"requested_pair": ("xmin", "xmax")},
            include_default_boundary_unions=False,
            verify=False,
        )
