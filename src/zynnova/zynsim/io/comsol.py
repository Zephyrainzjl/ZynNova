"""COMSOL native text mesh export for large, partitioned Tet4 models.

The writer follows the public COMSOL native text format (``0 1``) and the
minimal Mesh class version 4 layout used by externally generated meshes.  A
file can contain only tetrahedra, in which case COMSOL reconstructs lower
dimensional entities, or it can additionally contain explicitly partitioned
triangle entities and named selections.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from ..core import Mesh


@dataclass(frozen=True, slots=True)
class COMSOLSelectionInfo:
    """A named entity selection stored in an MPHTXT file."""

    label: str
    dimension: int
    entity_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class COMSOLMPHTXTInfo:
    """A memory-light structural summary returned by :func:`inspect_mphtxt`."""

    path: Path
    format_version: tuple[int, int]
    mesh_tag: str
    mesh_class_version: int
    space_dimension: int
    vertex_count: int
    start_vertex_index: int
    element_counts: Mapping[str, int]
    geometric_entity_ids: Mapping[str, tuple[int, ...]]
    selections: tuple[COMSOLSelectionInfo, ...]


@dataclass(frozen=True, slots=True)
class COMSOLMeshExportReport:
    """Metadata needed to reproduce and audit one COMSOL mesh export."""

    path: Path
    mesh_tag: str
    vertex_count: int
    tetrahedron_count: int
    triangle_count: int
    region_entity_map: Mapping[int, int]
    domain_selections: Mapping[str, tuple[int, ...]]
    boundary_selections: Mapping[str, tuple[int, ...]]
    boundary_entity_count: int


@dataclass(slots=True)
class _BoundaryPartition:
    triangles: np.ndarray
    entity_indices: np.ndarray
    selection_entities: dict[str, tuple[int, ...]]
    entity_count: int


def write_comsol_mphtxt(
    path: str | Path,
    mesh: Mesh,
    *,
    mesh_tag: str = "mesh1",
    domain_names: Mapping[int, str] | None = None,
    domain_selections: Mapping[str, Iterable[int]] | None = None,
    include_boundaries: bool = True,
    boundary_faces: Mapping[str, np.ndarray] | None = None,
    boundary_selections: Mapping[str, Iterable[str]] | None = None,
    include_internal_interfaces: bool = True,
    include_exterior: bool = True,
    create_domain_selections: bool = True,
    create_boundary_selections: bool = True,
    create_interface_selections: bool = True,
    create_exterior_selection: bool = True,
    float_precision: int = 17,
    line_ending: str = "\n",
) -> COMSOLMeshExportReport:
    """Write a partitioned Tet4 mesh in COMSOL's native ``.mphtxt`` format.

    Parameters
    ----------
    path:
        Destination. The write is atomic: a temporary sibling is replaced only
        after the complete text file has been produced.
    mesh:
        A validated :class:`~zynnova.zynsim.core.Mesh`. Arbitrarily shaped,
        disconnected, porous, or voxel-derived geometries are supported as
        long as their Tet4 topology is conforming.
    domain_names:
        Optional labels keyed by the original values in ``mesh.cell_regions``.
        COMSOL volume entity numbers always start at one; the returned report
        records the original-to-COMSOL mapping.
    domain_selections:
        Additional named unions of original region identifiers.
    include_boundaries:
        Write explicit ``tri`` elements for the exterior, material interfaces,
        and named internal faces. Set this to ``False`` for the compact,
        tetrahedron-only layout used by many external mesh generators; COMSOL
        will reconstruct missing lower-dimensional entities during import.
    boundary_faces:
        Additional named triangle sets. They are merged with
        ``mesh.boundary_faces``. Overlapping sets are supported and become
        overlapping COMSOL selections without duplicating triangle elements.
    boundary_selections:
        Optional named unions whose values reference boundary-set names.
    include_internal_interfaces:
        Explicitly preserve faces separating different cell regions.
    include_exterior:
        Explicitly preserve all exterior faces, including unnamed faces.

    Notes
    -----
    The writer streams coordinates, connectivity, and entity indices instead
    of constructing one large string. Memory growth is therefore dominated by
    the input mesh. Explicit boundary extraction requires additional
    ``O(number_of_tetrahedra)`` topology storage; for very large volume-only
    models use ``include_boundaries=False``.
    """

    target = Path(path)
    if target.suffix.lower() != ".mphtxt":
        raise ValueError("COMSOL native text mesh paths must end in .mphtxt")
    _validate_tag(mesh_tag)
    if not 8 <= int(float_precision) <= 17:
        raise ValueError("float_precision must lie between 8 and 17")
    if line_ending not in {"\n", "\r\n"}:
        raise ValueError("line_ending must be '\\n' or '\\r\\n'")

    original_regions = tuple(sorted(map(int, np.unique(mesh.cell_regions))))
    if not original_regions:
        raise ValueError("a COMSOL domain mesh needs at least one cell region")
    region_entity_map = {
        original_region: entity_index
        for entity_index, original_region in enumerate(original_regions, start=1)
    }
    domain_entity_indices = np.asarray(
        [region_entity_map[int(region)] for region in mesh.cell_regions],
        dtype=np.int64,
    )
    normalized_domain_names = _normalize_domain_names(original_regions, domain_names)
    normalized_domain_selections = _domain_selection_entities(
        original_regions,
        region_entity_map,
        normalized_domain_names,
        domain_selections,
        create_defaults=create_domain_selections,
    )

    if include_boundaries:
        named_faces = _merge_named_boundary_faces(mesh, boundary_faces)
        partition = _partition_boundary_faces(
            mesh,
            domain_entity_indices,
            named_faces,
            normalized_domain_names,
            include_internal_interfaces=include_internal_interfaces,
            include_exterior=include_exterior,
            create_named_selections=create_boundary_selections,
            create_interface_selections=create_interface_selections,
            create_exterior_selection=create_exterior_selection,
        )
        normalized_boundary_selections = _union_boundary_selections(
            partition.selection_entities,
            boundary_selections,
        )
    else:
        if boundary_faces or boundary_selections:
            raise ValueError(
                "boundary_faces and boundary_selections require include_boundaries=True"
            )
        partition = _BoundaryPartition(
            triangles=np.empty((0, 3), dtype=np.int64),
            entity_indices=np.empty(0, dtype=np.int64),
            selection_entities={},
            entity_count=0,
        )
        normalized_boundary_selections = {}

    selection_records: list[COMSOLSelectionInfo] = []
    selection_records.extend(
        COMSOLSelectionInfo(label, 3, entities)
        for label, entities in normalized_domain_selections.items()
    )
    selection_records.extend(
        COMSOLSelectionInfo(label, 2, entities)
        for label, entities in normalized_boundary_selections.items()
    )
    selection_tags = [
        f"{mesh_tag}_sel_{index:06d}"
        for index in range(1, len(selection_records) + 1)
    ]
    tags = [mesh_tag, *selection_tags]

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            _write_mphtxt(
                handle,
                mesh,
                tags,
                selection_records,
                domain_entity_indices,
                partition,
                float_precision=int(float_precision),
                newline=line_ending,
            )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return COMSOLMeshExportReport(
        path=target,
        mesh_tag=mesh_tag,
        vertex_count=mesh.n_nodes,
        tetrahedron_count=mesh.n_cells,
        triangle_count=int(len(partition.triangles)),
        region_entity_map=region_entity_map,
        domain_selections=normalized_domain_selections,
        boundary_selections=normalized_boundary_selections,
        boundary_entity_count=partition.entity_count,
    )


def write_mphtxt(
    path: str | Path,
    mesh: Mesh,
    **kwargs: object,
) -> COMSOLMeshExportReport:
    """Concise alias for :func:`write_comsol_mphtxt`."""

    return write_comsol_mphtxt(path, mesh, **kwargs)


def inspect_mphtxt(path: str | Path) -> COMSOLMPHTXTInfo:
    """Inspect a minimal Mesh-v4 MPHTXT file without retaining its large arrays.

    This reader intentionally targets the external-mesh layout emitted by this
    module and by the supplied ``PntCorGraph``-style reference file. Complete
    COMSOL Mesh-v8 files include an embedded geometric model and are outside
    this lightweight validation reader's scope.
    """

    source = Path(path)
    reader = _MPHTXTReader(source)
    format_version = tuple(reader.read_ints(2))
    if format_version != (0, 1):
        raise ValueError(f"unsupported COMSOL native format version {format_version}")
    tag_count = reader.read_int()
    tags = tuple(reader.read_string() for _ in range(tag_count))
    type_count = reader.read_int()
    object_types = tuple(reader.read_string() for _ in range(type_count))
    if tag_count != type_count or any(name != "obj" for name in object_types):
        raise ValueError("MPHTXT tags and object types are inconsistent")

    mesh_tag: str | None = None
    mesh_class_version: int | None = None
    space_dimension: int | None = None
    vertex_count: int | None = None
    start_vertex_index: int | None = None
    element_counts: dict[str, int] = defaultdict(int)
    entity_ids: dict[str, set[int]] = defaultdict(set)
    selections: list[COMSOLSelectionInfo] = []

    for object_index, tag in enumerate(tags):
        object_header = reader.read_ints(3)
        if object_header != [0, 0, 1]:
            raise ValueError(
                f"object {object_index} has invalid header {object_header}"
            )
        class_name = reader.read_string()
        if class_name == "Mesh":
            if mesh_tag is not None:
                raise ValueError("MPHTXT contains more than one Mesh object")
            mesh_tag = tag
            mesh_class_version = reader.read_int()
            if mesh_class_version != 4:
                raise ValueError(
                    "inspect_mphtxt supports external Mesh class version 4; "
                    f"found version {mesh_class_version}"
                )
            space_dimension = reader.read_int()
            vertex_count = reader.read_int()
            start_vertex_index = reader.read_int()
            coordinate_min = float("inf")
            coordinate_max = -float("inf")
            for _ in range(vertex_count):
                coordinates = reader.read_floats(space_dimension)
                coordinate_min = min(coordinate_min, min(coordinates))
                coordinate_max = max(coordinate_max, max(coordinates))
            if vertex_count and not np.isfinite([coordinate_min, coordinate_max]).all():
                raise ValueError("MPHTXT contains non-finite coordinates")

            element_type_count = reader.read_int()
            for _ in range(element_type_count):
                element_name = reader.read_string()
                vertices_per_element = reader.read_int()
                element_count = reader.read_int()
                minimum_index = start_vertex_index + vertex_count
                maximum_index = start_vertex_index - 1
                for _ in range(element_count):
                    connectivity = reader.read_ints(vertices_per_element)
                    minimum_index = min(minimum_index, min(connectivity))
                    maximum_index = max(maximum_index, max(connectivity))
                if element_count and (
                    minimum_index < start_vertex_index
                    or maximum_index >= start_vertex_index + vertex_count
                ):
                    raise ValueError(
                        f"{element_name} connectivity contains an invalid vertex index"
                    )
                entity_count = reader.read_int()
                if entity_count not in {0, element_count}:
                    raise ValueError(
                        f"{element_name} geometric entity count differs from "
                        "its element count"
                    )
                for _ in range(entity_count):
                    entity_ids[element_name].add(reader.read_int())
                element_counts[element_name] += element_count
        elif class_name == "Selection":
            selection_version = reader.read_int()
            if selection_version != 0:
                raise ValueError(
                    f"unsupported Selection class version {selection_version}"
                )
            label = reader.read_string()
            referenced_mesh = reader.read_string()
            dimension = reader.read_int()
            selection_count = reader.read_int()
            selected_entities = tuple(
                reader.read_int() for _ in range(selection_count)
            )
            if mesh_tag is not None and referenced_mesh != mesh_tag:
                raise ValueError(
                    f"selection {label!r} references {referenced_mesh!r}, "
                    f"not {mesh_tag!r}"
                )
            selections.append(
                COMSOLSelectionInfo(label, dimension, selected_entities)
            )
        else:
            raise ValueError(f"unsupported MPHTXT object class {class_name!r}")

    if mesh_tag is None:
        raise ValueError("MPHTXT does not contain a Mesh object")
    if reader.has_more_data():
        raise ValueError(
            f"unexpected trailing MPHTXT content near line {reader.line_number}"
        )
    assert mesh_class_version is not None
    assert space_dimension is not None
    assert vertex_count is not None
    assert start_vertex_index is not None
    return COMSOLMPHTXTInfo(
        path=source,
        format_version=(int(format_version[0]), int(format_version[1])),
        mesh_tag=mesh_tag,
        mesh_class_version=mesh_class_version,
        space_dimension=space_dimension,
        vertex_count=vertex_count,
        start_vertex_index=start_vertex_index,
        element_counts=dict(element_counts),
        geometric_entity_ids={
            name: tuple(sorted(values)) for name, values in entity_ids.items()
        },
        selections=tuple(selections),
    )


def _normalize_domain_names(
    regions: tuple[int, ...],
    names: Mapping[int, str] | None,
) -> dict[int, str]:
    supplied = {int(key): _validate_label(value) for key, value in (names or {}).items()}
    unknown = sorted(set(supplied) - set(regions))
    if unknown:
        raise ValueError(f"domain_names contains unknown cell regions {unknown}")
    return {
        region: supplied.get(region, f"domain_{region}")
        for region in regions
    }


def _domain_selection_entities(
    regions: tuple[int, ...],
    entity_map: Mapping[int, int],
    domain_names: Mapping[int, str],
    selections: Mapping[str, Iterable[int]] | None,
    *,
    create_defaults: bool,
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    if create_defaults:
        for region in regions:
            result[domain_names[region]] = (entity_map[region],)
    for raw_label, raw_regions in (selections or {}).items():
        label = _validate_label(raw_label)
        selected_regions = tuple(sorted(set(map(int, raw_regions))))
        if not selected_regions:
            raise ValueError(f"domain selection {label!r} is empty")
        unknown = sorted(set(selected_regions) - set(regions))
        if unknown:
            raise ValueError(
                f"domain selection {label!r} contains unknown regions {unknown}"
            )
        entities = tuple(entity_map[region] for region in selected_regions)
        result[label] = entities
    return result


def _merge_named_boundary_faces(
    mesh: Mesh,
    additional: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = defaultdict(list)
    for source in (mesh.boundary_faces, additional or {}):
        for raw_name, raw_faces in source.items():
            name = _validate_label(raw_name)
            faces = np.asarray(raw_faces, dtype=np.int64)
            if faces.size == 0:
                continue
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(
                    f"boundary face set {name!r} must have shape (n_faces, 3)"
                )
            if np.min(faces) < 0 or np.max(faces) >= mesh.n_nodes:
                raise ValueError(
                    f"boundary face set {name!r} contains an invalid node index"
                )
            merged[name].append(faces)
    return {
        name: np.unique(
            np.sort(np.concatenate(blocks, axis=0), axis=1),
            axis=0,
        )
        for name, blocks in merged.items()
    }


def _partition_boundary_faces(
    mesh: Mesh,
    domain_entities: np.ndarray,
    named_faces: Mapping[str, np.ndarray],
    domain_names: Mapping[int, str],
    *,
    include_internal_interfaces: bool,
    include_exterior: bool,
    create_named_selections: bool,
    create_interface_selections: bool,
    create_exterior_selection: bool,
) -> _BoundaryPartition:
    cells = mesh.cells
    cell_indices = np.arange(mesh.n_cells, dtype=np.int64)
    faces = np.concatenate(
        (
            cells[:, (1, 2, 3)],
            cells[:, (0, 3, 2)],
            cells[:, (0, 1, 3)],
            cells[:, (0, 2, 1)],
        ),
        axis=0,
    )
    owners = np.tile(cell_indices, 4)
    opposites = np.concatenate(
        (cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3])
    )
    keys = np.sort(faces, axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_faces = faces[order]
    sorted_owners = owners[order]
    sorted_opposites = opposites[order]

    reverse_names: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for name, selection_faces in named_faces.items():
        for face in selection_faces:
            reverse_names[tuple(map(int, face))].add(name)
    unmatched_named_faces = set(reverse_names)

    if len(sorted_keys) == 0:
        starts = np.empty(0, dtype=np.int64)
        stops = np.empty(0, dtype=np.int64)
    else:
        changes = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        starts = np.r_[0, np.flatnonzero(changes) + 1]
        stops = np.r_[starts[1:], len(sorted_keys)]

    grouped_faces: dict[
        tuple[bool, int, int, tuple[str, ...]],
        list[np.ndarray],
    ] = defaultdict(list)
    for start, stop in zip(starts, stops, strict=True):
        count = int(stop - start)
        key = tuple(map(int, sorted_keys[start]))
        unmatched_named_faces.discard(key)
        if count > 2:
            raise ValueError(f"non-manifold tetrahedral face {key} has {count} owners")
        group_names = tuple(sorted(reverse_names.get(key, ())))
        owner_rows = np.arange(start, stop)
        owner_domains = domain_entities[sorted_owners[owner_rows]]
        exterior = count == 1
        interface = count == 2 and owner_domains[0] != owner_domains[1]
        explicitly_named = bool(group_names)
        if not (
            (exterior and include_exterior)
            or (interface and include_internal_interfaces)
            or explicitly_named
        ):
            continue

        if exterior:
            selected_row = start
            adjacent = (0, int(owner_domains[0]))
        else:
            local_choice = min(
                range(count),
                key=lambda offset: (
                    int(owner_domains[offset]),
                    int(sorted_owners[start + offset]),
                ),
            )
            selected_row = start + local_choice
            adjacent = tuple(sorted(map(int, owner_domains)))
        face = sorted_faces[selected_row].copy()
        opposite = int(sorted_opposites[selected_row])
        p0, p1, p2 = mesh.nodes[face]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.dot(normal, mesh.nodes[opposite] - p0) > 0.0:
            face = face[[0, 2, 1]]
        signature = (exterior, adjacent[0], adjacent[1], group_names)
        grouped_faces[signature].append(face)

    if unmatched_named_faces:
        preview = sorted(unmatched_named_faces)[:8]
        raise ValueError(
            "named boundary triangles are not faces of the tetrahedral mesh: "
            f"{preview}"
        )

    entity_triangles: list[np.ndarray] = []
    entity_indices: list[np.ndarray] = []
    selection_entities: dict[str, set[int]] = defaultdict(set)
    entity_id = 0
    entity_to_region = {
        int(entity): int(region)
        for region, entity in zip(
            np.unique(mesh.cell_regions),
            np.unique(domain_entities),
            strict=True,
        )
    }
    for signature in sorted(grouped_faces):
        exterior, first_domain, second_domain, group_names = signature
        block = np.asarray(grouped_faces[signature], dtype=np.int64)
        for component in _connected_face_components(block):
            component_faces = block[component]
            entity_triangles.append(component_faces)
            entity_indices.append(
                np.full(len(component_faces), entity_id, dtype=np.int64)
            )
            if create_named_selections:
                for name in group_names:
                    selection_entities[name].add(entity_id)
            if exterior and create_exterior_selection:
                selection_entities["all_exterior"].add(entity_id)
            if (
                not exterior
                and first_domain != second_domain
                and create_interface_selections
            ):
                first_region = entity_to_region[first_domain]
                second_region = entity_to_region[second_domain]
                label = (
                    f"interface_{domain_names[first_region]}_"
                    f"{domain_names[second_region]}"
                )
                selection_entities[label].add(entity_id)
            entity_id += 1

    triangles = (
        np.concatenate(entity_triangles, axis=0)
        if entity_triangles
        else np.empty((0, 3), dtype=np.int64)
    )
    indices = (
        np.concatenate(entity_indices)
        if entity_indices
        else np.empty(0, dtype=np.int64)
    )
    return _BoundaryPartition(
        triangles=np.ascontiguousarray(triangles),
        entity_indices=np.ascontiguousarray(indices),
        selection_entities={
            label: tuple(sorted(entities))
            for label, entities in selection_entities.items()
            if entities
        },
        entity_count=entity_id,
    )


def _connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    if len(faces) == 0:
        return []
    parent = np.arange(len(faces), dtype=np.int64)
    rank = np.zeros(len(faces), dtype=np.int8)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    edge_owner: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        a, b, c = map(int, face)
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            previous = edge_owner.setdefault(key, face_index)
            union(face_index, previous)

    groups: dict[int, list[int]] = defaultdict(list)
    for face_index in range(len(faces)):
        groups[find(face_index)].append(face_index)
    return [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(groups.items(), key=lambda item: min(item[1]))
    ]


def _union_boundary_selections(
    base: Mapping[str, tuple[int, ...]],
    unions: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[int, ...]]:
    result = dict(base)
    for raw_label, raw_names in (unions or {}).items():
        label = _validate_label(raw_label)
        names = tuple(map(str, raw_names))
        if not names:
            raise ValueError(f"boundary selection {label!r} is empty")
        missing = sorted(set(names) - set(base))
        if missing:
            raise ValueError(
                f"boundary selection {label!r} references unknown sets {missing}"
            )
        entities = sorted(
            {
                entity
                for name in names
                for entity in base[name]
            }
        )
        result[label] = tuple(entities)
    return result


def _write_mphtxt(
    handle: TextIO,
    mesh: Mesh,
    tags: list[str],
    selections: list[COMSOLSelectionInfo],
    domain_entities: np.ndarray,
    boundaries: _BoundaryPartition,
    *,
    float_precision: int,
    newline: str,
) -> None:
    def line(value: object = "") -> None:
        handle.write(f"{value}{newline}")

    def string(value: str, comment: str | None = None) -> None:
        suffix = "" if comment is None else f" # {comment}"
        line(f"{len(value)} {value}{suffix}")

    line("# COMSOL native text mesh generated by ZynNova ZynSim")
    line("# Major & minor version")
    line("0 1")
    line()
    line(f"{len(tags)} # number of tags")
    line("# Tags")
    for tag in tags:
        string(tag)
    line()
    line(f"{len(tags)} # number of types")
    line("# Types")
    for _ in tags:
        string("obj")
    line()

    line("# --------- Object 0 ----------")
    line("0 0 1")
    string("Mesh", "class")
    line("4 # version")
    line("3 # sdim")
    line()
    line(f"{mesh.n_nodes} # number of mesh vertices")
    line("0 # start vertex index")
    line("# Mesh vertex coordinates")
    coordinate_format = f".{float_precision}g"
    for coordinates in mesh.nodes:
        line(" ".join(format(float(value), coordinate_format) for value in coordinates))
    line()
    element_type_count = 1 + int(len(boundaries.triangles) > 0)
    line(f"{element_type_count} # number of element types")
    line("# Type #0")
    string("tet", "type name")
    line("4 # number of vertices per element")
    line(f"{mesh.n_cells} # number of elements")
    line("# Elements")
    for cell in mesh.cells:
        line(" ".join(map(str, map(int, cell))))
    line(f"{mesh.n_cells} # number of geometric entity indices")
    line("# Geometric entity indices")
    for entity in domain_entities:
        line(int(entity))

    if len(boundaries.triangles):
        line()
        line("# Type #1")
        string("tri", "type name")
        line("3 # number of vertices per element")
        line(f"{len(boundaries.triangles)} # number of elements")
        line("# Elements")
        for face in boundaries.triangles:
            line(" ".join(map(str, map(int, face))))
        line(
            f"{len(boundaries.entity_indices)} "
            "# number of geometric entity indices"
        )
        line("# Geometric entity indices")
        for entity in boundaries.entity_indices:
            line(int(entity))

    for object_index, selection in enumerate(selections, start=1):
        line()
        line(f"# --------- Object {object_index} ----------")
        line("0 0 1")
        string("Selection", "class")
        line("0 # version")
        string(selection.label, "label")
        string(tags[0], "geometry/mesh tag")
        line(f"{selection.dimension} # dimension")
        line(f"{len(selection.entity_ids)} # number of entities")
        line("# Entities")
        for entity in selection.entity_ids:
            line(entity)


def _validate_tag(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(value)):
        raise ValueError(
            "mesh_tag must start with an ASCII letter and contain only "
            "ASCII letters, digits, and underscores"
        )


def _validate_label(value: object) -> str:
    label = str(value)
    if not label or any(character in label for character in "\r\n\x00"):
        raise ValueError("COMSOL selection labels must be nonempty single-line strings")
    return label


class _MPHTXTReader:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("r", encoding="utf-8-sig", newline=None)
        self._buffer: str | None = None
        self.line_number = 0

    def _next_raw_data_line(self) -> str:
        if self._buffer is not None:
            line = self._buffer
            self._buffer = None
            return line
        for raw_line in self._handle:
            self.line_number += 1
            line = raw_line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            return line
        raise ValueError("unexpected end of MPHTXT file")

    def read_string(self) -> str:
        line = self._next_raw_data_line()
        match = re.match(r"^\s*(\d+)\s(.*)$", line)
        if match is None:
            raise ValueError(
                f"expected serialized string at line {self.line_number}"
            )
        length = int(match.group(1))
        remainder = match.group(2)
        value = remainder[:length]
        trailing = remainder[length:].strip()
        if len(value) != length or (trailing and not trailing.startswith("#")):
            raise ValueError(
                f"invalid serialized string length at line {self.line_number}"
            )
        return value

    def _read_numeric_tokens(self) -> list[str]:
        line = self._next_raw_data_line()
        content = line.split("#", 1)[0].strip()
        if not content:
            raise ValueError(f"missing numeric data at line {self.line_number}")
        return content.split()

    def read_int(self) -> int:
        values = self._read_numeric_tokens()
        if len(values) != 1:
            raise ValueError(f"expected one integer at line {self.line_number}")
        return int(values[0])

    def read_ints(self, count: int) -> list[int]:
        values = self._read_numeric_tokens()
        if len(values) != count:
            raise ValueError(
                f"expected {count} integers at line {self.line_number}, "
                f"found {len(values)}"
            )
        return list(map(int, values))

    def read_floats(self, count: int) -> list[float]:
        values = self._read_numeric_tokens()
        if len(values) != count:
            raise ValueError(
                f"expected {count} floats at line {self.line_number}, "
                f"found {len(values)}"
            )
        return list(map(float, values))

    def has_more_data(self) -> bool:
        try:
            self._buffer = self._next_raw_data_line()
        except ValueError as exc:
            if str(exc) == "unexpected end of MPHTXT file":
                self._handle.close()
                return False
            raise
        return True


__all__ = [
    "COMSOLMPHTXTInfo",
    "COMSOLMeshExportReport",
    "COMSOLSelectionInfo",
    "inspect_mphtxt",
    "write_comsol_mphtxt",
    "write_mphtxt",
]
