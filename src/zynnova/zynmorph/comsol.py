"""COMSOL MPHTXT export adapters for ZynMorph microstructures.

This module intentionally does not implement a second, approximate COMSOL
writer.  It adapts ZynMorph's ``VolumeMesh`` and ``MicrostructureVolume``
containers to the two native-text writers retained from the original ZynSim
implementation:

* :func:`zynnova.zynsim.io.write_comsol_mphtxt` for an already materialized,
  partitioned Tet4 mesh with named selections and explicit interfaces;
* :func:`zynnova.zynsim.io.write_large_voxel_comsol_mphtxt` for bounded-memory
  Hex8 or six-Tet4-per-voxel export directly from a label volume.

Consequently, the MPHTXT header, class version, zero-based connectivity,
geometric-entity remapping, connected boundary partitioning, selection
objects, atomic file replacement, and streaming rules are exactly those of
that validated implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import VolumeMesh
from ..zynsim.core import Mesh
from ..zynsim.io import (
    COMSOLMPHTXTInfo,
    COMSOLMeshExportReport,
    GeneralCOMSOLExportReport,
    LargeVoxelMeshPlan,
    inspect_mphtxt,
    plan_large_voxel_mesh,
    write_comsol_mphtxt,
    write_large_voxel_comsol_mphtxt,
)
from .schema import BatteryPhase
from .volume import MicrostructureVolume


def default_battery_domain_selections(
    regions: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    """Build non-empty battery-domain unions using original region labels.

    The exact ZynSim MPHTXT writer accepts arbitrary named unions of source
    region identifiers and remaps them transactionally to COMSOL's positive,
    contiguous geometric entity numbers.  This helper only constructs useful
    ZynMorph battery unions; the writer remains responsible for the actual
    remapping and validation.
    """

    active = {int(value) for value in regions}
    groups: dict[str, tuple[int, ...]] = {
        "all_domains": tuple(sorted(active)),
        "all_electrolyte": (
            int(BatteryPhase.SEPARATOR_ELECTROLYTE),
            int(BatteryPhase.POSITIVE_ELECTROLYTE),
            int(BatteryPhase.NEGATIVE_ELECTROLYTE),
        ),
        "all_active_material": (
            int(BatteryPhase.POSITIVE_ACTIVE),
            int(BatteryPhase.NEGATIVE_ACTIVE),
        ),
        "all_cbd": (
            int(BatteryPhase.POSITIVE_CBD),
            int(BatteryPhase.NEGATIVE_CBD),
        ),
        "all_interphases": (
            int(BatteryPhase.POSITIVE_CEI),
            int(BatteryPhase.NEGATIVE_SEI),
        ),
        "all_cracks_and_voids": (int(BatteryPhase.CRACK),),
        "all_current_collectors": (
            int(BatteryPhase.NEGATIVE_CURRENT_COLLECTOR),
            int(BatteryPhase.POSITIVE_CURRENT_COLLECTOR),
        ),
        "positive_electrode": (
            int(BatteryPhase.POSITIVE_ACTIVE),
            int(BatteryPhase.POSITIVE_ELECTROLYTE),
            int(BatteryPhase.POSITIVE_CBD),
            int(BatteryPhase.POSITIVE_CEI),
        ),
        "negative_electrode": (
            int(BatteryPhase.NEGATIVE_ACTIVE),
            int(BatteryPhase.NEGATIVE_ELECTROLYTE),
            int(BatteryPhase.NEGATIVE_CBD),
            int(BatteryPhase.NEGATIVE_SEI),
        ),
        "separator": (int(BatteryPhase.SEPARATOR_ELECTROLYTE),),
    }
    result: dict[str, tuple[int, ...]] = {}
    for name, values in groups.items():
        selected = tuple(value for value in values if value in active)
        if selected:
            result[name] = selected
    return result


def default_coordinate_boundary_unions() -> dict[str, tuple[str, ...]]:
    """Return named unions referencing the six coordinate-face selections."""

    return {
        "x_terminal_pair": ("xmin", "xmax"),
        "y_periodic_candidate_pair": ("ymin", "ymax"),
        "z_periodic_candidate_pair": ("zmin", "zmax"),
        "transverse_boundaries": ("ymin", "ymax", "zmin", "zmax"),
    }


def to_comsol_tet_mesh(
    mesh: VolumeMesh,
    *,
    named_boundary_faces: Mapping[str, np.ndarray] | None = None,
    include_coordinate_boundaries: bool = True,
    coordinate_tolerance: float | None = None,
) -> Mesh:
    """Adapt a ZynMorph ``VolumeMesh`` to the exact ZynSim Tet4 writer type.

    Connectivity and region identifiers are preserved without renumbering.
    COMSOL entity renumbering is deliberately deferred to
    :func:`write_comsol_mphtxt`, matching the original implementation.
    """

    converted = Mesh(
        nodes=mesh.nodes,
        cells=mesh.tetrahedra,
        cell_regions=mesh.cell_regions,
        boundary_faces=dict(named_boundary_faces or {}),
        metadata={**mesh.metadata, "adapter": "zynnova.zynmorph.comsol"},
    )
    if include_coordinate_boundaries:
        converted = converted.with_coordinate_boundaries(
            tolerance=coordinate_tolerance,
            overwrite=False,
        )
    return converted


def export_comsol_mphtxt(
    path: str | Path,
    mesh: VolumeMesh,
    *,
    mesh_tag: str = "mesh1",
    domain_names: Mapping[int, str] | None = None,
    domain_selections: Mapping[str, Iterable[int]] | None = None,
    named_boundary_faces: Mapping[str, np.ndarray] | None = None,
    boundary_selections: Mapping[str, Iterable[str]] | None = None,
    include_coordinate_boundaries: bool = True,
    include_default_battery_selections: bool = True,
    include_default_boundary_unions: bool = True,
    include_boundaries: bool = True,
    include_internal_interfaces: bool = True,
    include_exterior: bool = True,
    create_domain_selections: bool = True,
    create_boundary_selections: bool = True,
    create_interface_selections: bool = True,
    create_exterior_selection: bool = True,
    float_precision: int = 17,
    line_ending: str = "\n",
    verify: bool = True,
) -> COMSOLMeshExportReport:
    """Export a partitioned Tet4 mesh through the original ZynSim writer.

    The resulting MPHTXT uses COMSOL native text format ``0 1`` and external
    ``Mesh`` class version 4.  Domain and boundary selections are stored as
    native ``Selection`` objects rather than sidecar-only metadata.
    """

    active_regions = tuple(sorted(map(int, np.unique(mesh.cell_regions))))
    supplied_names = mesh.region_names if domain_names is None else domain_names
    active_names = {
        int(region): str(supplied_names.get(int(region), f"domain_{int(region)}"))
        for region in active_regions
    }

    merged_domain_selections: dict[str, tuple[int, ...]] = {}
    if include_default_battery_selections:
        merged_domain_selections.update(default_battery_domain_selections(active_regions))
    for label, values in (domain_selections or {}).items():
        merged_domain_selections[str(label)] = tuple(map(int, values))

    converted = to_comsol_tet_mesh(
        mesh,
        named_boundary_faces=named_boundary_faces,
        include_coordinate_boundaries=include_coordinate_boundaries,
    )

    merged_boundary_selections: dict[str, tuple[str, ...]] = {}
    if (
        include_default_boundary_unions
        and include_coordinate_boundaries
        and include_boundaries
        and create_boundary_selections
    ):
        merged_boundary_selections.update(default_coordinate_boundary_unions())
    for label, names in (boundary_selections or {}).items():
        merged_boundary_selections[str(label)] = tuple(map(str, names))

    report = write_comsol_mphtxt(
        path,
        converted,
        mesh_tag=mesh_tag,
        domain_names=active_names,
        domain_selections=merged_domain_selections or None,
        include_boundaries=include_boundaries,
        boundary_selections=merged_boundary_selections or None,
        include_internal_interfaces=include_internal_interfaces,
        include_exterior=include_exterior,
        create_domain_selections=create_domain_selections,
        create_boundary_selections=create_boundary_selections,
        create_interface_selections=create_interface_selections,
        create_exterior_selection=create_exterior_selection,
        float_precision=float_precision,
        line_ending=line_ending,
    )
    if verify:
        _verify_tet4_export(report)
    return report



def plan_voxel_comsol_mphtxt(
    volume: MicrostructureVolume,
    *,
    element_type: str = "hex8",
    include_exterior_boundaries: bool = True,
    include_material_interfaces: bool = True,
) -> LargeVoxelMeshPlan:
    """Estimate counts and text size with the original bounded-memory planner."""

    labels_xyz = np.ascontiguousarray(volume.labels.transpose(2, 1, 0))
    return plan_large_voxel_mesh(
        labels_xyz,
        element_type=element_type,
        include_exterior_boundaries=include_exterior_boundaries,
        include_material_interfaces=include_material_interfaces,
    )

def export_voxel_comsol_mphtxt(
    path: str | Path,
    volume: MicrostructureVolume,
    *,
    element_type: str = "hex8",
    include_exterior_boundaries: bool = True,
    include_material_interfaces: bool = True,
    mesh_tag: str = "mesh1",
    float_precision: int = 17,
    chunk_size: int = 262_144,
    prefer_native: bool = True,
    verify: bool = True,
) -> GeneralCOMSOLExportReport:
    """Stream a label volume to MPHTXT using the original out-of-core writer.

    ZynMorph stores arrays in ``(z, y, x)`` order while the original ZynSim
    voxel writer consumes ``(x, y, z)``.  The adapter performs an explicit
    transpose and reverses spacing/origin tuples so the physical COMSOL axes
    remain ``x, y, z``.  No interpolation or relabelling is performed.
    """

    labels_xyz = np.ascontiguousarray(volume.labels.transpose(2, 1, 0))
    dz, dy, dx = (float(value) for value in volume.voxel_size_m)
    oz, oy, ox = (float(value) for value in volume.origin_m)
    active_regions = tuple(sorted(map(int, np.unique(volume.labels))))
    phase_names = {
        region: str(volume.phase_names.get(region, f"phase_{region}"))
        for region in active_regions
    }
    report = write_large_voxel_comsol_mphtxt(
        path,
        labels_xyz,
        voxel_size_m=(dx, dy, dz),
        origin_m=(ox, oy, oz),
        element_type=element_type,
        phase_names=phase_names,
        include_exterior_boundaries=include_exterior_boundaries,
        include_material_interfaces=include_material_interfaces,
        mesh_tag=mesh_tag,
        float_precision=float_precision,
        chunk_size=chunk_size,
        prefer_native=prefer_native,
    )
    if verify:
        _verify_voxel_export(report, volume=volume, element_type=element_type)
    return report


def inspect_comsol_mphtxt(path: str | Path) -> COMSOLMPHTXTInfo:
    """Inspect an exported Mesh-v4 MPHTXT with the original lightweight reader."""

    return inspect_mphtxt(path)


def _verify_tet4_export(report: COMSOLMeshExportReport) -> COMSOLMPHTXTInfo:
    info = inspect_mphtxt(report.path)
    if info.format_version != (0, 1) or info.mesh_class_version != 4:
        raise RuntimeError("MPHTXT verification failed: unexpected COMSOL format/class version")
    if info.space_dimension != 3 or info.start_vertex_index != 0:
        raise RuntimeError("MPHTXT verification failed: expected 3-D zero-based mesh")
    if info.vertex_count != report.vertex_count:
        raise RuntimeError("MPHTXT verification failed: vertex count mismatch")
    if info.element_counts.get("tet") != report.tetrahedron_count:
        raise RuntimeError("MPHTXT verification failed: tetrahedron count mismatch")
    if info.element_counts.get("tri", 0) != report.triangle_count:
        raise RuntimeError("MPHTXT verification failed: triangle count mismatch")
    return info


def _verify_voxel_export(
    report: GeneralCOMSOLExportReport,
    *,
    volume: MicrostructureVolume,
    element_type: str,
) -> COMSOLMPHTXTInfo:
    info = inspect_mphtxt(report.path)
    if info.format_version != (0, 1) or info.mesh_class_version != 4:
        raise RuntimeError("MPHTXT verification failed: unexpected COMSOL format/class version")
    if info.space_dimension != 3 or info.start_vertex_index != 0:
        raise RuntimeError("MPHTXT verification failed: expected 3-D zero-based mesh")
    nz, ny, nx = volume.shape
    expected_vertices = (nx + 1) * (ny + 1) * (nz + 1)
    if info.vertex_count != expected_vertices or report.vertex_count != expected_vertices:
        raise RuntimeError("MPHTXT verification failed: voxel vertex count mismatch")
    normalized = str(element_type).lower()
    is_tet = normalized in {"tet", "tet4", "tetrahedron"}
    element_name = "tet" if is_tet else "hex"
    multiplier = 6 if is_tet else 1
    if info.element_counts.get(element_name) != volume.labels.size * multiplier:
        raise RuntimeError("MPHTXT verification failed: voxel element count mismatch")
    return info


__all__ = [
    "COMSOLMPHTXTInfo",
    "COMSOLMeshExportReport",
    "GeneralCOMSOLExportReport",
    "LargeVoxelMeshPlan",
    "default_battery_domain_selections",
    "default_coordinate_boundary_unions",
    "export_comsol_mphtxt",
    "export_voxel_comsol_mphtxt",
    "inspect_comsol_mphtxt",
    "plan_voxel_comsol_mphtxt",
    "to_comsol_tet_mesh",
]
