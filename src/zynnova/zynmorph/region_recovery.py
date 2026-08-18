"""Recover physical material IDs from TetGen facet-bounded region attributes.

TetGen region attributes identify *topological* facet-bounded compartments.  A
single physical material can occupy several disconnected compartments.  Recent
TetGen versions may assign additional non-zero attributes to compartments that
were not explicitly seeded.  This module collapses those raw topology IDs back
to the material IDs declared by the PLC, using the marked PLC interfaces rather
than numeric heuristics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.exceptions import GeometryError


TETGEN_REGION_RECOVERY_API = "facet-material-v1"


@dataclass(frozen=True, slots=True)
class TetGenRegionRecoveryReport:
    """Audit trail for topology-region -> physical-material recovery."""

    requested_material_ids: tuple[int, ...]
    raw_region_ids: tuple[int, ...]
    automatic_region_ids: tuple[int, ...]
    raw_to_material: Mapping[int, int]
    raw_region_cell_counts: Mapping[int, int]
    material_cell_counts: Mapping[int, int]
    boundary_anchors: int
    interface_constraints: int
    recovered_automatic_regions: int
    recovered_cells: int
    unresolved_region_ids: tuple[int, ...]
    valid: bool


def _structured_face_keys(faces: np.ndarray) -> np.ndarray:
    ordered = np.ascontiguousarray(np.sort(faces, axis=1), dtype=np.int64)
    key_dtype = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])
    return ordered.view(key_dtype).reshape(-1)


def _subface_adjacent_tetrahedra(
    tetrahedra: np.ndarray,
    trifaces: np.ndarray,
    *,
    chunk_tetrahedra: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one/two adjacent tetrahedron indices for each TetGen triface.

    The implementation matches only the output subfaces, avoiding a Python dict
    for every face of a potentially multi-million-cell tetrahedral mesh.
    """

    tets = np.ascontiguousarray(tetrahedra, dtype=np.int64)
    faces = np.ascontiguousarray(trifaces, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise GeometryError("tetrahedra must have shape (N, 4)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise GeometryError("TetGen trifaces must have shape (M, 3)")

    adjacent_0 = np.full(len(faces), -1, dtype=np.int64)
    adjacent_1 = np.full(len(faces), -1, dtype=np.int64)
    if not len(faces):
        return adjacent_0, adjacent_1

    triface_keys = _structured_face_keys(faces)
    order = np.argsort(triface_keys, kind="stable")
    sorted_keys = triface_keys[order]

    local_faces = (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (0, 2, 1),
    )

    for start in range(0, len(tets), int(chunk_tetrahedra)):
        stop = min(len(tets), start + int(chunk_tetrahedra))
        block = tets[start:stop]
        block_ids = np.arange(start, stop, dtype=np.int64)

        candidate_faces = np.concatenate(
            [block[:, local] for local in local_faces], axis=0
        )
        candidate_tets = np.concatenate([block_ids] * 4)
        candidate_keys = _structured_face_keys(candidate_faces)

        positions = np.searchsorted(sorted_keys, candidate_keys)
        within = positions < len(sorted_keys)
        if not np.any(within):
            continue

        candidate_indices = np.flatnonzero(within)
        positions_within = positions[within]
        exact = sorted_keys[positions_within] == candidate_keys[within]
        if not np.any(exact):
            continue

        candidate_indices = candidate_indices[exact]
        matched_sorted_positions = positions[candidate_indices]
        matched_face_indices = order[matched_sorted_positions]
        matched_tet_indices = candidate_tets[candidate_indices]

        # At most two tetrahedra can own a manifold triangle.  The loop is over
        # matched PLC faces only (normally O(surface faces)), not all tet faces.
        for face_index, tet_index in zip(
            matched_face_indices.tolist(),
            matched_tet_indices.tolist(),
            strict=True,
        ):
            current = int(adjacent_0[face_index])
            if current < 0:
                adjacent_0[face_index] = int(tet_index)
            elif current != int(tet_index):
                second = int(adjacent_1[face_index])
                if second < 0:
                    adjacent_1[face_index] = int(tet_index)
                elif second != int(tet_index):
                    raise GeometryError(
                        "a TetGen output subface is adjacent to more than two tetrahedra"
                    )

    return adjacent_0, adjacent_1


def _assign_material(
    mapping: dict[int, int],
    raw_region: int,
    material: int,
    *,
    reason: str,
) -> bool:
    previous = mapping.get(int(raw_region))
    if previous is None:
        mapping[int(raw_region)] = int(material)
        return True
    if previous != int(material):
        raise GeometryError(
            "conflicting TetGen material recovery constraints for raw region "
            f"{raw_region}: {previous} vs {material} ({reason})"
        )
    return False


def recover_tetgen_material_regions(
    tetrahedra: np.ndarray,
    raw_region_attributes: np.ndarray,
    trifaces: np.ndarray,
    triface_markers: np.ndarray,
    marker_region_pairs: Mapping[int, tuple[int | None, int | None]],
    *,
    requested_material_ids: Sequence[int],
    outside_region: int = -1,
    void_region_ids: Sequence[int] = (),
) -> tuple[np.ndarray, TetGenRegionRecoveryReport]:
    """Collapse TetGen topology attributes to the PLC's physical material IDs.

    TetGen may generate extra attributes for facet-bounded compartments without
    an explicit region seed.  Numeric IDs such as 4, 5, ... are topology labels,
    not new materials.  We recover their material identity from the marked PLC
    interface graph:

    * user-requested TetGen attributes are anchors;
    * an exterior/void boundary anchors its sole adjacent raw compartment to the
      material on the meshed side;
    * an internal marked facet imposes the unordered material pair declared by
      ``marker_region_pairs`` on its two adjacent raw compartments;
    * constraints propagate until every raw compartment has a physical material.

    No rule relies on raw attribute magnitude/order.
    """

    tets = np.ascontiguousarray(tetrahedra, dtype=np.int64)
    raw = np.ascontiguousarray(raw_region_attributes, dtype=np.int32)
    faces = np.ascontiguousarray(trifaces, dtype=np.int64)
    markers = np.ascontiguousarray(triface_markers, dtype=np.int32)

    if raw.shape != (len(tets),):
        raise GeometryError("raw_region_attributes must contain one value per tetrahedron")
    if markers.shape != (len(faces),):
        raise GeometryError("triface_markers must contain one value per triface")

    requested = tuple(sorted({int(value) for value in requested_material_ids}))
    requested_set = set(requested)
    if not requested:
        raise GeometryError("requested_material_ids cannot be empty")

    voids = {int(value) for value in void_region_ids}
    outside = int(outside_region)
    raw_ids, raw_counts = np.unique(raw, return_counts=True)
    raw_ids_tuple = tuple(map(int, raw_ids.tolist()))
    automatic = tuple(sorted(set(raw_ids_tuple) - requested_set))

    # Fast path: no TetGen-generated topology attributes to collapse.
    if not automatic:
        counts = {int(k): int(v) for k, v in zip(raw_ids, raw_counts, strict=True)}
        report = TetGenRegionRecoveryReport(
            requested_material_ids=requested,
            raw_region_ids=raw_ids_tuple,
            automatic_region_ids=(),
            raw_to_material={value: value for value in raw_ids_tuple},
            raw_region_cell_counts=counts,
            material_cell_counts=counts,
            boundary_anchors=0,
            interface_constraints=0,
            recovered_automatic_regions=0,
            recovered_cells=0,
            unresolved_region_ids=(),
            valid=set(raw_ids_tuple) == requested_set,
        )
        if not report.valid:
            raise GeometryError(
                "TetGen output is missing requested material regions: "
                f"requested={list(requested)}, output={list(raw_ids_tuple)}"
            )
        return raw, report

    if not len(faces):
        raise GeometryError(
            "TetGen produced extra topology-region attributes but returned no marked "
            "subfaces from which their physical materials can be recovered"
        )

    adjacent_0, adjacent_1 = _subface_adjacent_tetrahedra(tets, faces)
    missing = int(np.count_nonzero(adjacent_0 < 0))
    if missing:
        raise GeometryError(
            f"could not match {missing} TetGen output subfaces to tetrahedron faces"
        )

    mapping: dict[int, int] = {
        raw_id: raw_id for raw_id in raw_ids_tuple if raw_id in requested_set
    }
    boundary_anchors = 0
    constraints: list[tuple[int, int, int, int, int]] = []

    normalized_pairs = {
        int(marker): (
            None if pair[0] is None else int(pair[0]),
            None if pair[1] is None else int(pair[1]),
        )
        for marker, pair in marker_region_pairs.items()
    }

    for face_index, marker_value in enumerate(markers.tolist()):
        pair = normalized_pairs.get(int(marker_value))
        if pair is None:
            continue

        tet_a = int(adjacent_0[face_index])
        tet_b = int(adjacent_1[face_index])
        raw_a = int(raw[tet_a])

        left, right = pair
        pair_materials = [
            value
            for value in (left, right)
            if value is not None and value != outside and value not in voids
        ]

        if tet_b < 0:
            # Exterior boundaries and material/void boundaries have exactly one
            # meshed side.  That side must be the sole non-outside/non-void
            # material declared by the PLC marker.
            unique_materials = sorted(set(pair_materials))
            if len(unique_materials) == 1:
                if _assign_material(
                    mapping,
                    raw_a,
                    unique_materials[0],
                    reason=f"boundary marker {marker_value}",
                ):
                    boundary_anchors += 1
            continue

        raw_b = int(raw[tet_b])
        if raw_a == raw_b:
            # A marked physical interface cannot sit inside one raw TetGen region
            # when it declares two distinct meshed materials.
            meshed_pair = sorted(set(pair_materials))
            if len(meshed_pair) == 2:
                raise GeometryError(
                    "TetGen did not separate a marked material interface into two "
                    f"topology regions: marker={marker_value}, raw_region={raw_a}, "
                    f"materials={meshed_pair}"
                )
            continue

        if len(set(pair_materials)) == 2:
            m1, m2 = sorted(set(pair_materials))
            constraints.append((raw_a, raw_b, m1, m2, int(marker_value)))

    # Constraint propagation from seeded/user attributes and outer/void anchors.
    changed = True
    while changed:
        changed = False
        for raw_a, raw_b, m1, m2, marker in constraints:
            a = mapping.get(raw_a)
            b = mapping.get(raw_b)

            if a is not None and b is not None:
                if {a, b} != {m1, m2}:
                    raise GeometryError(
                        "TetGen raw-region adjacency conflicts with the PLC material "
                        f"pair on marker {marker}: raw=({raw_a}->{a}, {raw_b}->{b}), "
                        f"expected materials={m1, m2}"
                    )
                continue

            if a is not None:
                if a == m1:
                    changed |= _assign_material(
                        mapping, raw_b, m2, reason=f"interface marker {marker}"
                    )
                elif a == m2:
                    changed |= _assign_material(
                        mapping, raw_b, m1, reason=f"interface marker {marker}"
                    )
                else:
                    raise GeometryError(
                        f"raw region {raw_a} maps to material {a}, which is not on "
                        f"PLC marker {marker} material pair {(m1, m2)}"
                    )
                continue

            if b is not None:
                if b == m1:
                    changed |= _assign_material(
                        mapping, raw_a, m2, reason=f"interface marker {marker}"
                    )
                elif b == m2:
                    changed |= _assign_material(
                        mapping, raw_a, m1, reason=f"interface marker {marker}"
                    )
                else:
                    raise GeometryError(
                        f"raw region {raw_b} maps to material {b}, which is not on "
                        f"PLC marker {marker} material pair {(m1, m2)}"
                    )

    unresolved = tuple(sorted(set(raw_ids_tuple) - set(mapping)))
    if unresolved:
        # Give actionable topology information rather than guessing from the
        # integer values assigned by TetGen.
        involved: dict[int, list[tuple[int, int, int]]] = {value: [] for value in unresolved}
        for raw_a, raw_b, m1, m2, marker in constraints:
            if raw_a in involved:
                involved[raw_a].append((raw_b, m1, m2))
            if raw_b in involved:
                involved[raw_b].append((raw_a, m1, m2))
        raise GeometryError(
            "TetGen generated facet-bounded topology regions that could not be "
            "mapped to physical materials from PLC markers. Add/reposition region "
            "seeds or fix missing interface markers. "
            f"unresolved={list(unresolved)}, adjacency={involved}"
        )

    ordered_raw_ids = np.asarray(sorted(mapping), dtype=np.int32)
    ordered_material_ids = np.asarray(
        [mapping[int(value)] for value in ordered_raw_ids], dtype=np.int32
    )
    lookup_positions = np.searchsorted(ordered_raw_ids, raw)
    if (
        np.any(lookup_positions >= len(ordered_raw_ids))
        or not np.array_equal(ordered_raw_ids[lookup_positions], raw)
    ):
        raise GeometryError("internal TetGen raw-region lookup failure")
    mapped = np.ascontiguousarray(ordered_material_ids[lookup_positions], dtype=np.int32)
    material_ids, material_counts = np.unique(mapped, return_counts=True)
    output_materials = set(map(int, material_ids.tolist()))
    if output_materials != requested_set:
        raise GeometryError(
            "TetGen topology-region recovery did not reproduce exactly the requested "
            f"materials: requested={sorted(requested_set)}, recovered={sorted(output_materials)}"
        )

    raw_count_map = {
        int(k): int(v) for k, v in zip(raw_ids, raw_counts, strict=True)
    }
    material_count_map = {
        int(k): int(v) for k, v in zip(material_ids, material_counts, strict=True)
    }
    recovered_cells = int(
        sum(raw_count_map[value] for value in automatic)
    )
    report = TetGenRegionRecoveryReport(
        requested_material_ids=requested,
        raw_region_ids=raw_ids_tuple,
        automatic_region_ids=automatic,
        raw_to_material=dict(sorted(mapping.items())),
        raw_region_cell_counts=raw_count_map,
        material_cell_counts=material_count_map,
        boundary_anchors=int(boundary_anchors),
        interface_constraints=len(constraints),
        recovered_automatic_regions=len(automatic),
        recovered_cells=recovered_cells,
        unresolved_region_ids=(),
        valid=True,
    )
    return np.ascontiguousarray(mapped, dtype=np.int32), report


__all__ = [
    "TETGEN_REGION_RECOVERY_API",
    "TetGenRegionRecoveryReport",
    "recover_tetgen_material_regions",
]
