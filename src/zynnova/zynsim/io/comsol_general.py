"""Mixed-element and out-of-core COMSOL MPHTXT writers."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

import numpy as np

from ..core.general_mesh import GeneralMesh
from .comsol_topology import (
    COMSOL_HEX8_TO_TET4,
    HexTopologyValidationReport,
    structured_comsol_hex_rows,
    to_comsol_connectivity,
    validate_comsol_hex_connectivity,
    validate_structured_hex_topology,
)

try:  # direct C++ writer may be available in built wheels
    from zynnova._native import _zynsim_voxel_native as _native
except Exception:  # pragma: no cover
    _native = None


@dataclass(frozen=True, slots=True)
class LargeVoxelMeshPlan:
    voxel_shape: tuple[int, int, int]
    element_type: str
    vertex_count: int
    volume_element_count: int
    exterior_face_count: int
    interface_face_count: int
    estimated_binary_bytes: int
    estimated_mphtxt_bytes: int
    region_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GeneralCOMSOLExportReport:
    path: Path
    vertex_count: int
    element_counts: Mapping[str, int]
    selections: Mapping[str, tuple[int, tuple[int, ...]]]
    out_of_core: bool
    native_backend: bool
    domain_entity_indices_written: bool = True
    volume_elements_written: bool = True
    topology_validation: HexTopologyValidationReport | None = None

    @property
    def diagnostic_mode(self) -> str:
        if not self.volume_elements_written:
            return "surface-only"
        if not self.domain_entity_indices_written:
            return "volume-without-domain-entities"
        return "full-volume"


@dataclass(frozen=True, slots=True)
class HexTopologyAudit:
    """Orientation/topology evidence for first-order COMSOL Hex8 cells."""

    cells_checked: int
    faces_checked: int
    shared_faces: int
    boundary_faces: int
    nonpositive_jacobians: int
    nonplanar_faces: int
    same_side_shared_faces: int
    overconnected_faces: int
    valid: bool
    expected_shared_faces: int | None = None
    missing_shared_faces: int = 0


def plan_large_voxel_mesh(
    phase_labels: np.ndarray,
    *,
    element_type: str = "hex8",
    include_exterior_boundaries: bool = True,
    include_material_interfaces: bool = True,
) -> LargeVoxelMeshPlan:
    labels = _labels(phase_labels)
    kind = _kind(element_type)
    nx, ny, nz = map(int, labels.shape)
    vertices = (nx + 1) * (ny + 1) * (nz + 1)
    multiplier = 1 if kind == "hex8" else 6
    volume = labels.size * multiplier
    boundary_multiplier = 1 if kind == "hex8" else 2
    exterior = (
        boundary_multiplier * 2 * (nx * ny + nx * nz + ny * nz)
        if include_exterior_boundaries else 0
    )
    interface = (
        boundary_multiplier * _interface_count(labels)
        if include_material_interfaces else 0
    )
    nodes_per = 8 if kind == "hex8" else 4
    binary = vertices * 3 * 8 + volume * nodes_per * 8 + volume * 4
    text = int(vertices * 72 + volume * (nodes_per * 14 + 10) + (exterior + interface) * 70)
    return LargeVoxelMeshPlan(
        voxel_shape=(nx, ny, nz),
        element_type=kind,
        vertex_count=vertices,
        volume_element_count=volume,
        exterior_face_count=exterior,
        interface_face_count=interface,
        estimated_binary_bytes=binary,
        estimated_mphtxt_bytes=text,
        region_labels=tuple(_phase_counts_chunked(labels)),
    )


def write_general_comsol_mphtxt(
    path: str | Path,
    mesh: GeneralMesh,
    *,
    mesh_tag: str = "mesh1",
    float_precision: int = 17,
    include_domain_entity_indices: bool = True,
    include_volume_elements: bool = True,
) -> GeneralCOMSOLExportReport:
    """Write a validated mixed first-order mesh to COMSOL MPHTXT.

    User material labels are allowed to be zero, negative, sparse, or repeated
    across topological dimensions.  Before writing they are deterministically
    remapped to positive, contiguous COMSOL geometric entity numbers *within
    each dimension*.  Named selections are remapped in the same transaction.
    """

    target = _target(path)
    blocks = tuple(
        block for block in mesh.blocks if include_volume_elements or block.dimension != 3
    )
    if not blocks:
        raise ValueError(
            "surface-only COMSOL export requires at least one non-volume element block"
        )
    entity_maps = _entity_maps(mesh)
    selections = _remap_selections(mesh.selections, entity_maps)
    write_domain_entities = bool(include_domain_entity_indices and include_volume_elements)
    if not write_domain_entities:
        selections = {name: value for name, value in selections.items() if value[0] != 3}
    selection_records = [(label, dim, entities) for label, (dim, entities) in selections.items()]
    tags = [mesh_tag, *[f"{mesh_tag}_sel_{i:06d}" for i in range(1, len(selection_records) + 1)]]
    temporary = _temporary(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            _header(handle, tags)
            _line(handle, "# --------- Object 0 ----------")
            _line(handle, "0 0 1")
            _string(handle, "Mesh", "class")
            _line(handle, "4 # version")
            _line(handle, "3 # sdim")
            _line(handle)
            _line(handle, f"{mesh.n_nodes} # number of mesh vertices")
            _line(handle, "0 # start vertex index")
            _line(handle, "# Mesh vertex coordinates")
            _write_float_rows(handle, mesh.nodes, float_precision)
            _line(handle)
            _line(handle, f"{len(blocks)} # number of element types")
            for block_index, block in enumerate(blocks):
                _line(handle, f"# Type #{block_index}")
                _string(handle, block.comsol_type_name, "type name")
                _line(handle, f"{block.connectivity.shape[1]} # number of vertices per element")
                _line(handle, f"{block.n_elements} # number of elements")
                _line(handle, "# Elements")
                _write_int_rows(
                    handle,
                    to_comsol_connectivity(block.element_type, block.connectivity),
                )
                write_entities = write_domain_entities or block.dimension != 3
                entity_count = block.n_elements if write_entities else 0
                _line(handle, f"{entity_count} # number of geometric entity indices")
                _line(handle, "# Geometric entity indices")
                if write_entities:
                    mapping = entity_maps[block.dimension]
                    remapped = np.fromiter(
                        (mapping[int(value)] for value in block.entity_ids),
                        dtype=np.int32,
                        count=block.n_elements,
                    )
                    _write_int_vector(handle, remapped)
                _line(handle)
            _write_selection_objects(handle, mesh_tag, tags, selection_records)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    counts: Counter[str] = Counter()
    for block in blocks:
        counts[block.element_type] += block.n_elements
    return GeneralCOMSOLExportReport(
        path=target,
        vertex_count=mesh.n_nodes,
        element_counts=dict(counts),
        selections=selections,
        out_of_core=False,
        native_backend=False,
        domain_entity_indices_written=write_domain_entities,
        volume_elements_written=bool(include_volume_elements),
        topology_validation=None,
    )


def write_large_voxel_comsol_mphtxt(
    path: str | Path,
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | Sequence[float] = 1.0,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    element_type: str = "hex8",
    phase_names: Mapping[int, str] | None = None,
    include_exterior_boundaries: bool = True,
    include_material_interfaces: bool = True,
    mesh_tag: str = "mesh1",
    float_precision: int = 17,
    chunk_size: int = 262_144,
    prefer_native: bool = True,
    include_domain_entity_indices: bool = True,
    include_volume_elements: bool = True,
    validate_topology: bool = True,
    topology_validation_limit: int = 100_000,
) -> GeneralCOMSOLExportReport:
    """Write a huge voxel mesh without materializing nodes/connectivity.

    The function performs deterministic sequential passes over the label field.
    Peak additional memory is ``O(chunk_size)`` rather than ``O(n_elements)``.
    Both conforming Hex8 and six-Tet4-per-voxel volume discretizations are
    supported.  Exterior and phase-interface Quad4 sets are used with Hex8; conforming Tri3 sets are used with Tet4.
    """

    labels = _labels(phase_labels)
    target = _target(path)
    spacing = _triple(voxel_size_m)
    origin = _origin3(origin_m)
    kind = _kind(element_type)
    if not include_volume_elements and not (
        include_exterior_boundaries or include_material_interfaces
    ):
        raise ValueError(
            "surface-only COMSOL export requires exterior boundaries and/or material interfaces"
        )
    write_domain_entities = bool(include_domain_entity_indices and include_volume_elements)
    if (
        include_volume_elements
        and prefer_native
        and _native is not None
        and hasattr(_native, "write_voxel_mphtxt")
    ):
        report = _native.write_voxel_mphtxt(
            labels,
            str(target),
            spacing,
            origin,
            kind,
            bool(include_exterior_boundaries),
            bool(include_material_interfaces),
            str(mesh_tag),
            int(float_precision),
            bool(write_domain_entities),
        )
        return GeneralCOMSOLExportReport(
            path=target,
            vertex_count=int(report["vertex_count"]),
            element_counts=dict(report["element_counts"]),
            selections={},
            out_of_core=True,
            native_backend=True,
            domain_entity_indices_written=write_domain_entities,
            volume_elements_written=True,
            topology_validation=None,
        )

    topology_report = None
    if kind == "hex8" and include_volume_elements and validate_topology:
        topology_report = validate_structured_hex_topology(
            labels.shape,
            spacing=spacing,
            origin=origin,
            maximum_shared_faces=topology_validation_limit,
        )
        if not topology_report.valid:
            raise ValueError(
                "COMSOL Hex8 topology validation failed: "
                f"same_side_shared_faces={topology_report.same_side_shared_faces}, "
                f"minimum_jacobian={topology_report.minimum_jacobian}, "
                f"failure_coordinate={topology_report.failure_coordinate}"
            )

    plan = plan_large_voxel_mesh(
        labels,
        element_type=kind,
        include_exterior_boundaries=include_exterior_boundaries,
        include_material_interfaces=include_material_interfaces,
    )
    region_map = {phase: index for index, phase in enumerate(plan.region_labels, start=1)}
    names = {phase: _safe_name((phase_names or {}).get(phase, f"phase_{phase}")) for phase in plan.region_labels}
    boundary_entities, interface_entities = _boundary_entity_maps(labels, include_exterior_boundaries, include_material_interfaces)
    selections: dict[str, tuple[int, tuple[int, ...]]] = (
        {names[phase]: (3, (region_map[phase],)) for phase in plan.region_labels}
        if write_domain_entities
        else {}
    )
    selections.update({name: (2, (entity,)) for name, entity in boundary_entities.items()})
    selections.update({f"interface_{names[a]}_{names[b]}": (2, (entity,)) for (a, b), entity in interface_entities.items()})
    selection_records = [(label, dim, entities) for label, (dim, entities) in selections.items()]
    tags = [mesh_tag, *[f"{mesh_tag}_sel_{i:06d}" for i in range(1, len(selection_records) + 1)]]
    temporary = _temporary(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            _header(handle, tags)
            _line(handle, "# --------- Object 0 ----------")
            _line(handle, "0 0 1")
            _string(handle, "Mesh", "class")
            _line(handle, "4 # version")
            _line(handle, "3 # sdim")
            _line(handle)
            _line(handle, f"{plan.vertex_count} # number of mesh vertices")
            _line(handle, "0 # start vertex index")
            _line(handle, "# Mesh vertex coordinates")
            _stream_nodes(handle, labels.shape, spacing, origin, float_precision, chunk_size)
            boundary_count = plan.exterior_face_count + plan.interface_face_count
            type_count = int(include_volume_elements) + int(boundary_count > 0)
            _line(handle)
            _line(handle, f"{type_count} # number of element types")
            type_index = 0
            if include_volume_elements:
                _line(handle, f"# Type #{type_index}")
                comsol_kind, nodes_per = ("hex", 8) if kind == "hex8" else ("tet", 4)
                _string(handle, comsol_kind, "type name")
                _line(handle, f"{nodes_per} # number of vertices per element")
                _line(handle, f"{plan.volume_element_count} # number of elements")
                _line(handle, "# Elements")
                _stream_volume_elements(handle, labels.shape, kind, chunk_size)
                volume_entity_count = plan.volume_element_count if write_domain_entities else 0
                _line(handle, f"{volume_entity_count} # number of geometric entity indices")
                _line(handle, "# Geometric entity indices")
                if write_domain_entities:
                    _stream_volume_entities(handle, labels, region_map, kind, chunk_size)
                type_index += 1
            if boundary_count:
                if include_volume_elements:
                    _line(handle)
                _line(handle, f"# Type #{type_index}")
                boundary_type, boundary_nodes = ("quad", 4) if kind == "hex8" else ("tri", 3)
                _string(handle, boundary_type, "type name")
                _line(handle, f"{boundary_nodes} # number of vertices per element")
                _line(handle, f"{boundary_count} # number of elements")
                _line(handle, "# Elements")
                _stream_boundary_quads(handle, labels, include_exterior_boundaries, include_material_interfaces, values=False, boundary_entities=boundary_entities, interface_entities=interface_entities, chunk_size=chunk_size, triangulate=(kind == "tet4"))
                _line(handle, f"{boundary_count} # number of geometric entity indices")
                _line(handle, "# Geometric entity indices")
                _stream_boundary_quads(handle, labels, include_exterior_boundaries, include_material_interfaces, values=True, boundary_entities=boundary_entities, interface_entities=interface_entities, chunk_size=chunk_size, triangulate=(kind == "tet4"))
            _write_selection_objects(handle, mesh_tag, tags, selection_records)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    counts: dict[str, int] = {}
    if include_volume_elements:
        counts[kind] = plan.volume_element_count
    if boundary_count:
        counts["quad4" if kind == "hex8" else "tri3"] = boundary_count
    return GeneralCOMSOLExportReport(
        path=target,
        vertex_count=plan.vertex_count,
        element_counts=counts,
        selections=selections,
        out_of_core=True,
        native_backend=False,
        domain_entity_indices_written=write_domain_entities,
        volume_elements_written=bool(include_volume_elements),
        topology_validation=topology_report,
    )


def _stream_nodes(handle: TextIO, shape: tuple[int, int, int], spacing: tuple[float, float, float], origin: tuple[float, float, float], precision: int, chunk: int) -> None:
    nx, ny, nz = map(int, shape)
    yz = (ny + 1) * (nz + 1)
    total = (nx + 1) * yz
    fmt = f"%.{precision}g"
    for start in range(0, total, chunk):
        ids = np.arange(start, min(total, start + chunk), dtype=np.int64)
        i, remainder = np.divmod(ids, yz)
        j, k = np.divmod(remainder, nz + 1)
        coordinates = np.column_stack((origin[0] + spacing[0] * i, origin[1] + spacing[1] * j, origin[2] + spacing[2] * k))
        np.savetxt(handle, coordinates, fmt=fmt)


def _hex_rows(shape: tuple[int, int, int], start: int, stop: int) -> np.ndarray:
    if (
        _native is not None
        and hasattr(_native, "hex_connectivity_range")
        and hasattr(_native, "hex_connectivity_convention")
        and _native.hex_connectivity_convention() == "comsol-v4-tensor-1"
    ):
        return np.ascontiguousarray(
            _native.hex_connectivity_range(tuple(map(int, shape)), int(start), int(stop)),
            dtype=np.int64,
        )
    # Old installed native extensions emitted VTK/cyclic order.  Do not call
    # them silently: the pure-Python implementation below is the authoritative
    # COMSOL Mesh-v4 tensor-product order.
    return structured_comsol_hex_rows(shape, start, stop)


def _stream_volume_elements(handle: TextIO, shape: tuple[int, int, int], kind: str, chunk: int) -> None:
    total = int(np.prod(shape))
    pattern = COMSOL_HEX8_TO_TET4
    for start in range(0, total, chunk):
        hexes = _hex_rows(shape, start, min(total, start + chunk))
        rows = hexes if kind == "hex8" else hexes[:, pattern].reshape(-1, 4)
        np.savetxt(handle, rows, fmt="%d")


def _stream_volume_entities(handle: TextIO, labels: np.ndarray, region_map: Mapping[int, int], kind: str, chunk: int) -> None:
    flat = labels.ravel(order="C")
    multiplier = 1 if kind == "hex8" else 6
    phases = np.asarray(sorted(map(int, region_map)), dtype=np.int64)
    mapped = np.asarray([int(region_map[int(phase)]) for phase in phases], dtype=np.int32)
    for start in range(0, flat.size, chunk):
        values = np.asarray(flat[start : start + chunk], dtype=np.int64)
        positions = np.searchsorted(phases, values)
        if np.any(positions >= len(phases)) or np.any(phases[np.minimum(positions, len(phases) - 1)] != values):
            raise ValueError("label field contains a phase absent from region_map")
        entities = mapped[positions]
        if multiplier > 1:
            entities = np.repeat(entities, multiplier)
        np.savetxt(handle, entities, fmt="%d")


def _iter_interface_chunks(
    labels: np.ndarray,
    axis: int,
    *,
    x_chunk: int = 32,
):
    """Yield interface comparisons in bounded x-slabs.

    Chunking along x keeps the helper compatible with NumPy memmaps and avoids
    allocating an array proportional to the complete volume.
    """

    nx = int(labels.shape[0])
    if axis == 0:
        for start in range(0, max(nx - 1, 0), x_chunk):
            stop = min(nx - 1, start + x_chunk)
            yield start, labels[start:stop, :, :], labels[start + 1 : stop + 1, :, :]
    elif axis == 1:
        for start in range(0, nx, x_chunk):
            stop = min(nx, start + x_chunk)
            yield start, labels[start:stop, :-1, :], labels[start:stop, 1:, :]
    elif axis == 2:
        for start in range(0, nx, x_chunk):
            stop = min(nx, start + x_chunk)
            yield start, labels[start:stop, :, :-1], labels[start:stop, :, 1:]
    else:  # pragma: no cover - internal contract
        raise ValueError("axis must be 0, 1, or 2")


def _boundary_entity_maps(labels: np.ndarray, exterior: bool, interfaces: bool) -> tuple[dict[str, int], dict[tuple[int, int], int]]:
    entity = 1
    boundary: dict[str, int] = {}
    if exterior:
        for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            boundary[name] = entity
            entity += 1
    pairs: set[tuple[int, int]] = set()
    if interfaces:
        for axis in range(3):
            for _, left, right in _iter_interface_chunks(labels, axis):
                mask = left != right
                if not np.any(mask):
                    continue
                a = left[mask].astype(np.int64, copy=False)
                b = right[mask].astype(np.int64, copy=False)
                lo = np.minimum(a, b)
                hi = np.maximum(a, b)
                pairs.update(zip(map(int, lo), map(int, hi), strict=True))
    interfaces_map = {pair: entity + index for index, pair in enumerate(sorted(pairs))}
    return boundary, interfaces_map


def _stream_boundary_quads(
    handle: TextIO,
    labels: np.ndarray,
    exterior: bool,
    interfaces: bool,
    *,
    values: bool,
    boundary_entities: Mapping[str, int],
    interface_entities: Mapping[tuple[int, int], int],
    chunk_size: int = 262_144,
    triangulate: bool = False,
) -> None:
    """Stream COMSOL tensor Quad4 faces with physically consistent normals.

    COMSOL Quad4 local order is ``00, 10, 01, 11``.  It is not the cyclic
    VTK order ``00, 10, 11, 01``.  The parameter directions below are chosen
    so ``u x v`` points outwards on the exterior and from the lower-coordinate
    voxel towards the higher-coordinate voxel on internal interfaces.
    """

    nx, ny, nz = map(int, labels.shape)

    def node(i: np.ndarray | int, j: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
        return np.asarray(i) * ((ny + 1) * (nz + 1)) + np.asarray(j) * (nz + 1) + np.asarray(k)

    def emit(rows: np.ndarray, entity: int | np.ndarray) -> None:
        output_rows = np.asarray(rows, dtype=np.int64)
        if triangulate:
            # Tensor quad diagonal 00--11.
            output_rows = np.concatenate(
                (output_rows[:, [0, 1, 3]], output_rows[:, [0, 3, 2]]),
                axis=0,
            )
        if values:
            data = (
                np.full(len(rows), int(entity), dtype=np.int64)
                if np.isscalar(entity)
                else np.asarray(entity, dtype=np.int64)
            )
            if triangulate:
                data = np.concatenate((data, data))
            np.savetxt(handle, data, fmt="%d")
        else:
            np.savetxt(handle, output_rows, fmt="%d")

    def ranges(total: int):
        for first in range(0, total, max(1, int(chunk_size))):
            yield first, min(total, first + max(1, int(chunk_size)))

    if exterior:
        # xmin: u=+z, v=+y -> -x.  xmax: u=+y, v=+z -> +x.
        for i, name in ((0, "xmin"), (nx, "xmax")):
            for first, stop in ranges(ny * nz):
                flat = np.arange(first, stop, dtype=np.int64)
                y, z = flat // nz, flat % nz
                if name == "xmin":
                    rows = np.column_stack((
                        node(i,y,z), node(i,y,z+1),
                        node(i,y+1,z), node(i,y+1,z+1),
                    ))
                else:
                    rows = np.column_stack((
                        node(i,y,z), node(i,y+1,z),
                        node(i,y,z+1), node(i,y+1,z+1),
                    ))
                emit(rows, boundary_entities[name])

        # ymin: u=+x, v=+z -> -y.  ymax: u=+z, v=+x -> +y.
        for j, name in ((0, "ymin"), (ny, "ymax")):
            for first, stop in ranges(nx * nz):
                flat = np.arange(first, stop, dtype=np.int64)
                x, z = flat // nz, flat % nz
                if name == "ymin":
                    rows = np.column_stack((
                        node(x,j,z), node(x+1,j,z),
                        node(x,j,z+1), node(x+1,j,z+1),
                    ))
                else:
                    rows = np.column_stack((
                        node(x,j,z), node(x,j,z+1),
                        node(x+1,j,z), node(x+1,j,z+1),
                    ))
                emit(rows, boundary_entities[name])

        # zmin: u=+y, v=+x -> -z.  zmax: u=+x, v=+y -> +z.
        for k, name in ((0, "zmin"), (nz, "zmax")):
            for first, stop in ranges(nx * ny):
                flat = np.arange(first, stop, dtype=np.int64)
                x, y = flat // ny, flat % ny
                if name == "zmin":
                    rows = np.column_stack((
                        node(x,y,k), node(x,y+1,k),
                        node(x+1,y,k), node(x+1,y+1,k),
                    ))
                else:
                    rows = np.column_stack((
                        node(x,y,k), node(x+1,y,k),
                        node(x,y+1,k), node(x+1,y+1,k),
                    ))
                emit(rows, boundary_entities[name])

    if interfaces:
        x_chunk = max(1, int(chunk_size) // max(1, ny * nz))
        for axis in range(3):
            for x_offset, left, right in _iter_interface_chunks(labels, axis, x_chunk=x_chunk):
                indices = np.argwhere(left != right)
                if not len(indices):
                    continue
                phase_a = left[tuple(indices.T)].astype(int, copy=False)
                phase_b = right[tuple(indices.T)].astype(int, copy=False)
                entities = np.fromiter(
                    (interface_entities[(min(a, b), max(a, b))] for a, b in zip(phase_a, phase_b, strict=True)),
                    dtype=np.int64,
                    count=len(indices),
                )
                indices[:, 0] += x_offset
                if axis == 0:
                    # +x, u=+y, v=+z.
                    i, j, k = indices[:,0] + 1, indices[:,1], indices[:,2]
                    rows = np.column_stack((
                        node(i,j,k), node(i,j+1,k),
                        node(i,j,k+1), node(i,j+1,k+1),
                    ))
                elif axis == 1:
                    # +y, u=+z, v=+x.
                    i, j, k = indices[:,0], indices[:,1] + 1, indices[:,2]
                    rows = np.column_stack((
                        node(i,j,k), node(i,j,k+1),
                        node(i+1,j,k), node(i+1,j,k+1),
                    ))
                else:
                    # +z, u=+x, v=+y.
                    i, j, k = indices[:,0], indices[:,1], indices[:,2] + 1
                    rows = np.column_stack((
                        node(i,j,k), node(i+1,j,k),
                        node(i,j+1,k), node(i+1,j+1,k),
                    ))
                emit(rows, entities)



def audit_comsol_hex8_connectivity(
    nodes: np.ndarray,
    cells: np.ndarray,
    *,
    representative: bool = False,
    tolerance: float | None = None,
) -> HexTopologyAudit:
    """Check COMSOL Hex8 Jacobians and ownership of every shared tensor face.

    The shared-face test mirrors the COMSOL import diagnostic: the two cell
    centres must lie on opposite sides of the shared face plane.  Planarity,
    face multiplicity, and the center Jacobian are checked independently so a
    bad local numbering cannot hide behind an unordered face-key match.
    """

    points = np.ascontiguousarray(nodes, dtype=np.float64)
    hexes = np.ascontiguousarray(cells, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if hexes.ndim != 2 or hexes.shape[1] != 8:
        raise ValueError("cells must have shape (M, 8)")
    if hexes.size and (hexes.min() < 0 or hexes.max() >= len(points)):
        raise ValueError("cells contain an out-of-range node index")
    if representative and len(hexes) > 4096:
        ids = np.unique(np.linspace(0, len(hexes) - 1, 4096, dtype=np.int64))
        hexes = hexes[ids]

    local_faces = np.asarray(
        [
            [0, 2, 4, 6],  # x-min tensor face
            [1, 3, 5, 7],  # x-max
            [0, 1, 4, 5],  # y-min
            [2, 3, 6, 7],  # y-max
            [0, 1, 2, 3],  # z-min
            [4, 5, 6, 7],  # z-max
        ],
        dtype=np.int64,
    )
    # Derivatives of trilinear Hex8 shape functions at the element center for
    # tensor local coordinates (xi, eta, zeta) in {-1,+1}^3.
    signs = np.asarray(
        [
            [-1,-1,-1], [1,-1,-1], [-1,1,-1], [1,1,-1],
            [-1,-1,1], [1,-1,1], [-1,1,1], [1,1,1],
        ],
        dtype=np.float64,
    )
    scale = np.ptp(points, axis=0) if len(points) else np.ones(3)
    characteristic = float(max(np.max(scale), 1.0))
    eps = (
        max(np.finfo(float).eps * 2048.0 * characteristic, np.finfo(float).tiny)
        if tolerance is None
        else float(tolerance)
    )
    if eps < 0.0:
        raise ValueError("tolerance cannot be negative")

    nonpositive = 0
    nonplanar = 0
    face_owners: dict[
        tuple[int, int, int, int],
        list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    for cell in hexes:
        xyz = points[cell]
        jacobian = (signs.T @ xyz) / 8.0
        determinant = float(np.linalg.det(jacobian))
        if not determinant > eps**3:
            nonpositive += 1
        centre = xyz.mean(axis=0)
        for local in local_faces:
            face_ids = cell[local]
            face = points[face_ids]
            # Tensor Quad4: 00,10,01,11.  The first three define the plane.
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            norm = float(np.linalg.norm(normal))
            if norm <= eps**2:
                nonplanar += 1
                unit = np.zeros(3)
            else:
                unit = normal / norm
                if abs(float(np.dot(face[3] - face[0], unit))) > eps:
                    nonplanar += 1
            key = tuple(sorted(map(int, face_ids)))
            face_owners.setdefault(key, []).append((centre, face.mean(axis=0), unit))

    shared = boundary = same_side = overconnected = 0
    for owners in face_owners.values():
        if len(owners) == 1:
            boundary += 1
        elif len(owners) == 2:
            shared += 1
            (left_centre, face_centre, face_normal), (right_centre, _same_face, _normal) = owners
            normal_length = float(np.linalg.norm(face_normal))
            if normal_length <= 0.0:
                same_side += 1
            else:
                signed_left = float(np.dot(left_centre - face_centre, face_normal))
                signed_right = float(np.dot(right_centre - face_centre, face_normal))
                # Distances are measured along a unit face normal.  Comparing
                # against eps**2 is then scale-consistent even for 100 nm voxels.
                if signed_left * signed_right >= -eps**2:
                    same_side += 1
        else:
            overconnected += 1

    valid = nonpositive == nonplanar == same_side == overconnected == 0
    return HexTopologyAudit(
        cells_checked=int(len(hexes)),
        faces_checked=int(sum(len(value) for value in face_owners.values())),
        shared_faces=shared,
        boundary_faces=boundary,
        nonpositive_jacobians=nonpositive,
        nonplanar_faces=nonplanar,
        same_side_shared_faces=same_side,
        overconnected_faces=overconnected,
        valid=valid,
        expected_shared_faces=None,
        missing_shared_faces=0,
    )


def audit_comsol_hex8_topology(
    shape: Sequence[int],
    *,
    spacing: float | Sequence[float] = 1.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    representative: bool = False,
) -> HexTopologyAudit:
    """Construct/audit the same structured topology used by the MPHTXT writer."""

    resolved_shape = tuple(map(int, shape))
    if len(resolved_shape) != 3 or min(resolved_shape) < 1:
        raise ValueError("shape must contain three positive integers")
    resolved_spacing = _triple(spacing)
    resolved_origin = _origin3(origin)
    if representative:
        # A 2x2x2 template exercises all three shared-face directions while
        # remaining independent of production volume size.
        test_shape = tuple(min(size, 2) for size in resolved_shape)
    else:
        test_shape = resolved_shape
    nodes = _structured_nodes_array(test_shape, resolved_spacing, resolved_origin)
    cells = _hex_rows(test_shape, 0, int(np.prod(test_shape)))
    audit = audit_comsol_hex8_connectivity(nodes, cells)
    tx, ty, tz = test_shape
    expected_shared = (tx - 1) * ty * tz + tx * (ty - 1) * tz + tx * ty * (tz - 1)
    missing = max(0, int(expected_shared - audit.shared_faces))
    return replace(
        audit,
        expected_shared_faces=int(expected_shared),
        missing_shared_faces=missing,
        valid=bool(audit.valid and missing == 0),
    )


def _structured_nodes_array(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    nx, ny, nz = shape
    i, j, k = np.meshgrid(
        np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1),
        indexing="ij",
    )
    return np.column_stack((
        origin[0] + spacing[0] * i.ravel(order="C"),
        origin[1] + spacing[1] * j.ravel(order="C"),
        origin[2] + spacing[2] * k.ravel(order="C"),
    ))


def _phase_counts_chunked(labels: np.ndarray, *, chunk_size_x: int = 32) -> dict[int, int]:
    """Return deterministic phase counts with bounded extra memory."""

    if (
        _native is not None
        and hasattr(_native, "phase_counts")
        and labels.dtype == np.int32
        and labels.flags.c_contiguous
    ):
        return dict(sorted((int(k), int(v)) for k, v in _native.phase_counts(labels).items()))
    counts: Counter[int] = Counter()
    for start in range(0, int(labels.shape[0]), max(1, int(chunk_size_x))):
        values, local = np.unique(
            np.asarray(labels[start : start + max(1, int(chunk_size_x))]),
            return_counts=True,
        )
        counts.update({int(value): int(count) for value, count in zip(values, local, strict=True)})
    return dict(sorted(counts.items()))

def _interface_count(labels: np.ndarray) -> int:
    if _native is not None and hasattr(_native, "interface_counts") and labels.flags.c_contiguous:
        return int(sum(_native.interface_counts(labels)))
    count = 0
    for axis in range(3):
        for _, left, right in _iter_interface_chunks(labels, axis):
            count += int(np.count_nonzero(left != right))
    return count


def _entity_maps(mesh: GeneralMesh) -> dict[int, dict[int, int]]:
    maps: dict[int, dict[int, int]] = {}
    for dimension in range(4):
        values = sorted(
            {
                int(value)
                for block in mesh.blocks
                if block.dimension == dimension
                for value in block.entity_ids
            }
        )
        maps[dimension] = {value: index for index, value in enumerate(values, start=1)}
    return maps


def _remap_selections(
    selections: Mapping[str, tuple[int, Sequence[int]]],
    entity_maps: Mapping[int, Mapping[int, int]],
) -> dict[str, tuple[int, tuple[int, ...]]]:
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for name, (dimension, entities) in selections.items():
        dimension = int(dimension)
        mapping = entity_maps.get(dimension, {})
        missing = [int(entity) for entity in entities if int(entity) not in mapping]
        if missing:
            raise ValueError(
                f"selection {name!r} references absent dimension-{dimension} entities {missing}"
            )
        result[str(name)] = (
            dimension,
            tuple(sorted({mapping[int(entity)] for entity in entities})),
        )
    return result

def _header(handle: TextIO, tags: Sequence[str]) -> None:
    _line(handle, "# COMSOL native text mesh generated by ZynNova ZynSim")
    _line(handle, "# Major & minor version")
    _line(handle, "0 1")
    _line(handle)
    _line(handle, f"{len(tags)} # number of tags")
    _line(handle, "# Tags")
    for tag in tags:
        _string(handle, tag)
    _line(handle)
    _line(handle, f"{len(tags)} # number of types")
    _line(handle, "# Types")
    for _ in tags:
        _string(handle, "obj")
    _line(handle)


def _write_selection_objects(handle: TextIO, mesh_tag: str, tags: Sequence[str], records: Sequence[tuple[str, int, Iterable[int]]]) -> None:
    for object_index, (label, dimension, entities) in enumerate(records, start=1):
        values = tuple(map(int, entities))
        _line(handle)
        _line(handle, f"# --------- Object {object_index} ----------")
        _line(handle, "0 0 1")
        _string(handle, "Selection", "class")
        _line(handle, "0 # version")
        _string(handle, str(label), "label")
        _string(handle, mesh_tag, "geometry/mesh tag")
        _line(handle, f"{int(dimension)} # dimension")
        _line(handle, f"{len(values)} # number of entities")
        _line(handle, "# Entities")
        for entity in values:
            _line(handle, entity)


def _write_float_rows(handle: TextIO, values: np.ndarray, precision: int) -> None:
    np.savetxt(handle, np.asarray(values, dtype=float), fmt=f"%.{precision}g")


def _write_int_rows(handle: TextIO, values: np.ndarray) -> None:
    np.savetxt(handle, np.asarray(values, dtype=np.int64), fmt="%d")


def _write_int_vector(handle: TextIO, values: np.ndarray) -> None:
    np.savetxt(handle, np.asarray(values, dtype=np.int64), fmt="%d")


def _line(handle: TextIO, value: object = "") -> None:
    handle.write(f"{value}\n")


def _string(handle: TextIO, value: str, comment: str | None = None) -> None:
    suffix = "" if comment is None else f" # {comment}"
    _line(handle, f"{len(value)} {value}{suffix}")


def _labels(value: np.ndarray) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 3 or labels.size == 0:
        raise ValueError("phase_labels must be a non-empty 3-D array")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.allclose(labels, np.rint(labels)):
            raise ValueError("phase labels must be integers")
    return np.asanyarray(labels, dtype=np.int32)


def _kind(value: str) -> str:
    key = str(value).lower()
    aliases = {"hex": "hex8", "hexahedron": "hex8", "tet": "tet4", "tetrahedron": "tet4"}
    key = aliases.get(key, key)
    if key not in {"hex8", "tet4"}:
        raise ValueError("element_type must be hex8 or tet4")
    return key



def _origin3(value: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(map(float, value))
    if len(result) != 3 or not np.isfinite(result).all():
        raise ValueError("origin_m must contain three finite values")
    return result

def _triple(value: float | Sequence[float]) -> tuple[float, float, float]:
    result = (float(value),) * 3 if np.isscalar(value) else tuple(map(float, value))
    if len(result) != 3 or min(result) <= 0.0 or not np.isfinite(result).all():
        raise ValueError("expected three positive finite values")
    return result  # type: ignore[return-value]


def _target(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".mphtxt":
        raise ValueError("COMSOL mesh path must end in .mphtxt")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _temporary(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, text=True)
    os.close(descriptor)
    return Path(name)


def _safe_name(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in text) or "selection"


__all__ = [
    "GeneralCOMSOLExportReport",
    "HexTopologyAudit",
    "HexTopologyValidationReport",
    "LargeVoxelMeshPlan",
    "audit_comsol_hex8_connectivity",
    "audit_comsol_hex8_topology",
    "plan_large_voxel_mesh",
    "validate_comsol_hex_connectivity",
    "validate_structured_hex_topology",
    "write_general_comsol_mphtxt",
    "write_large_voxel_comsol_mphtxt",
]
