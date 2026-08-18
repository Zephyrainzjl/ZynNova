from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from zynnova.geometry import TriangleMesh
from zynnova.zynmorph.freeform import (
    FreeformRegion,
    SurfaceShell,
    audit_freeform_shell_clearance,
    mesh_freeform_tetgen,
    orient_closed_surface,
)
from zynnova.zynmorph.tetgen import TetGenMeshingConfig, TetGenNativeStatus


def _tetra_surface(offset=(0.0, 0.0, 0.0), scale=1.0, *, invert=False):
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    ) * scale + np.asarray(offset, dtype=float)
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    if invert:
        faces = faces[:, [0, 2, 1]]
    return TriangleMesh(vertices=vertices, faces=faces)


def _component_signed_volume(mesh: TriangleMesh, face_ids: np.ndarray) -> float:
    tri = mesh.vertices[mesh.faces[face_ids]]
    return float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
        / 6.0
    )


def test_orient_closed_surface_orients_each_disconnected_component_outward():
    a = _tetra_surface(scale=1.0, invert=False)
    b = _tetra_surface(offset=(3.0, 0.0, 0.0), scale=0.7, invert=True)
    vertices = np.concatenate([a.vertices, b.vertices])
    faces = np.concatenate([a.faces, b.faces + len(a.vertices)])
    fixed = orient_closed_surface(TriangleMesh(vertices=vertices, faces=faces))
    assert _component_signed_volume(fixed, np.arange(0, 4)) > 0.0
    assert _component_signed_volume(fixed, np.arange(4, 8)) > 0.0
    assert fixed.metadata["connected_surface_components"] == 2


def test_freeform_clearance_audit_flags_underresolved_shell_gap():
    outer = SurfaceShell(_tetra_surface(scale=10.0), inside_region=2, name="outer")
    inner_close = SurfaceShell(
        _tetra_surface(offset=(0.05, 0.05, 0.05), scale=9.8),
        inside_region=1,
        outside_region=2,
        name="inner-close",
    )
    audit = audit_freeform_shell_clearance(
        (outer, inner_close), minimum_clearance_factor=0.35
    )
    assert not audit.valid
    assert audit.minimum_clearance_ratio < 0.35


def test_split_subface_retries_with_boundary_preservation(monkeypatch):
    shell = SurfaceShell(_tetra_surface(), inside_region=1, name="domain")
    calls = []

    class FakeNative:
        def tetrahedralize(self, *args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("Internal TetGen error within `split_subface`.")
            points = np.asarray(args[0], dtype=float)
            return {
                "points": points,
                "tetrahedra": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                "region_attributes": np.asarray([1.0]),
                "trifaces": np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
                "triface_markers": np.ones(4, dtype=np.int32),
                "switches": "pzAafq1.45/8Y",
                "version": "TetGen 1.6.0",
            }

    monkeypatch.setattr(
        "zynnova.zynmorph.freeform.tetgen_native_status",
        lambda: TetGenNativeStatus(True, "TetGen 1.6.0", None, Path("fake"), None, "AGPL-3.0-or-later"),
    )
    monkeypatch.setattr("zynnova.zynmorph.freeform._load_tetgen_native", lambda: FakeNative())

    result = mesh_freeform_tetgen(
        (shell,),
        (FreeformRegion(1, (0.1, 0.1, 0.1), "domain"),),
        config=TetGenMeshingConfig(
            normalize_coordinates=False,
            freeform_strict_clearance=True,
            retry_preserve_boundary_on_facet_error=True,
            conforming_delaunay=True,
        ),
    )
    assert len(calls) == 2
    # New native ABI appends preserve_boundary_facets after quiet.
    assert calls[0][-1] is False
    assert calls[1][-1] is True
    assert calls[1][-3] is False  # conforming_delaunay disabled on recovery
    assert np.asarray(calls[1][5]).shape == (0, 2)
    assert result.metadata["tetgen_boundary_recovery_mode"] == "preserve-boundary-facets"


def test_cpp_binding_exposes_tetgen_y_recovery_switch():
    source = Path(__file__).parents[2] / "cpp" / "bindings" / "zynmorph_tetgen_module.cpp"
    text = source.read_text(encoding="utf-8")
    assert 'switches << "Y"' in text
    assert 'py::arg("preserve_boundary_facets") = false' in text
    assert 'tetgen_binding_abi' in text


def test_native_loader_rejects_old_binding_abi():
    from zynnova.zynmorph.tetgen import _validate_tetgen_native_module

    class OldNative:
        tetgen_version = "TetGen 1.6.0"
        tetgen_binding_abi = 1

        @staticmethod
        def tetrahedralize(*args):
            return None

    with pytest.raises(ImportError, match="ABI is too old"):
        _validate_tetgen_native_module(OldNative())
