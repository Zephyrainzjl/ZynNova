"""Regression tests for the public ZynMorph COMSOL API."""

from __future__ import annotations


def test_top_level_import_and_hex_audit_public_symbols() -> None:
    import zynnova
    from zynnova.zynmorph import (
        HexTopologyAudit,
        audit_comsol_hex8_connectivity,
        audit_comsol_hex8_topology,
    )
    from zynnova.zynsim.io import HexTopologyAudit as IOHexTopologyAudit

    assert hasattr(zynnova, "zynmorph")
    assert HexTopologyAudit is IOHexTopologyAudit
    assert callable(audit_comsol_hex8_connectivity)
    assert callable(audit_comsol_hex8_topology)


def test_reported_scale_hex_topology_is_valid() -> None:
    from zynnova.zynmorph import audit_comsol_hex8_topology

    report = audit_comsol_hex8_topology(
        (2, 2, 2),
        spacing=(2.25e-7, 4.5e-7, 2.25e-7),
        origin=(0.0, 0.0, 0.0),
    )
    assert report.valid
    assert report.nonpositive_jacobians == 0
    assert report.same_side_shared_faces == 0
    assert report.missing_shared_faces == 0
