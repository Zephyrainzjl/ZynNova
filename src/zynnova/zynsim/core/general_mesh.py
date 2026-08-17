"""Validated mixed finite-element meshes for COMSOL and external solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..exceptions import MeshError
from .mesh import Mesh


_ELEMENT_NODES: dict[str, int] = {
    "point1": 1,
    "line2": 2,
    "tri3": 3,
    "quad4": 4,
    "tet4": 4,
    "pyramid5": 5,
    "wedge6": 6,
    "hex8": 8,
}
_ELEMENT_DIMENSION: dict[str, int] = {
    "point1": 0,
    "line2": 1,
    "tri3": 2,
    "quad4": 2,
    "tet4": 3,
    "pyramid5": 3,
    "wedge6": 3,
    "hex8": 3,
}
_COMSOL_NAMES = {
    "point1": "vtx",
    "line2": "edg",
    "tri3": "tri",
    "quad4": "quad",
    "tet4": "tet",
    "pyramid5": "pyr",
    "wedge6": "prism",
    "hex8": "hex",
}


@dataclass(slots=True)
class ElementBlock:
    element_type: str
    connectivity: np.ndarray
    entity_ids: np.ndarray | None = None
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.element_type).lower()
        if key not in _ELEMENT_NODES:
            raise MeshError(f"unsupported element type {self.element_type!r}")
        self.element_type = key
        cells = np.ascontiguousarray(self.connectivity, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != _ELEMENT_NODES[key]:
            raise MeshError(
                f"{key} connectivity must have shape (n, {_ELEMENT_NODES[key]})"
            )
        self.connectivity = cells
        if self.entity_ids is None:
            self.entity_ids = np.ones(len(cells), dtype=np.int32)
        else:
            entities = np.ascontiguousarray(self.entity_ids, dtype=np.int32)
            if entities.shape != (len(cells),):
                raise MeshError("entity_ids must have one value per element")
            self.entity_ids = entities
        self.metadata = dict(self.metadata)

    @property
    def dimension(self) -> int:
        return _ELEMENT_DIMENSION[self.element_type]

    @property
    def n_elements(self) -> int:
        return int(len(self.connectivity))

    @property
    def comsol_type_name(self) -> str:
        return _COMSOL_NAMES[self.element_type]


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    """Geometry-only quality summary for first-order volume elements."""

    volume_element_count: int
    sub_tetrahedron_count: int
    total_volume: float
    minimum_subvolume: float
    maximum_subvolume: float
    mean_subvolume: float
    minimum_scaled_jacobian: float
    mean_scaled_jacobian: float
    degenerate_subcells: int
    inverted_subcells: int

    @property
    def valid(self) -> bool:
        return self.degenerate_subcells == 0 and self.inverted_subcells == 0


@dataclass(slots=True)
class GeneralMesh:
    """Mixed first-order finite-element mesh.

    Volume blocks may contain Tet4, Hex8, Wedge6, and Pyramid5 elements, while
    explicit boundary blocks may contain Tri3 or Quad4 elements.  This is the
    common exchange representation for COMSOL and meshio-compatible solvers.
    """

    nodes: np.ndarray
    blocks: Sequence[ElementBlock]
    selections: Mapping[str, tuple[int, Sequence[int]]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = np.ascontiguousarray(self.nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
            raise MeshError("nodes must have shape (n,3) and be non-empty")
        if not np.isfinite(nodes).all():
            raise MeshError("nodes contain non-finite coordinates")
        self.nodes = nodes
        self.blocks = tuple(
            block if isinstance(block, ElementBlock) else ElementBlock(**block)
            for block in self.blocks
        )
        if not self.blocks:
            raise MeshError("at least one element block is required")
        for block in self.blocks:
            if block.connectivity.size:
                minimum = int(np.min(block.connectivity))
                maximum = int(np.max(block.connectivity))
                if minimum < 0 or maximum >= len(nodes):
                    raise MeshError(f"{block.element_type} contains invalid node indices")
        normalized: dict[str, tuple[int, tuple[int, ...]]] = {}
        for label, (dimension, entities) in self.selections.items():
            if int(dimension) not in (0, 1, 2, 3):
                raise MeshError("selection dimension must lie in [0,3]")
            normalized[str(label)] = (int(dimension), tuple(sorted(set(map(int, entities)))))
        self.selections = normalized
        self.metadata = dict(self.metadata)

    @property
    def n_nodes(self) -> int:
        return int(len(self.nodes))

    @property
    def n_elements(self) -> int:
        return int(sum(block.n_elements for block in self.blocks))

    @property
    def volume_blocks(self) -> tuple[ElementBlock, ...]:
        return tuple(block for block in self.blocks if block.dimension == 3)

    @property
    def boundary_blocks(self) -> tuple[ElementBlock, ...]:
        return tuple(block for block in self.blocks if block.dimension == 2)

    def quality_report(self, *, tolerance: float | None = None) -> MeshQualityReport:
        """Evaluate signed sub-tetrahedral volumes and scaled Jacobians.

        Hex8, Wedge6, and Pyramid5 blocks are decomposed into non-overlapping
        canonical Tet4 subcells.  A negative signed Jacobian is reported as an
        inversion rather than hidden by an absolute value.
        """

        if tolerance is None:
            span = np.ptp(self.nodes, axis=0)
            characteristic = max(float(np.max(span)), np.finfo(float).tiny)
            tolerance = max(100.0 * np.finfo(float).eps * characteristic**3, np.finfo(float).tiny)
        elif tolerance < 0.0:
            raise MeshError("volume tolerance cannot be negative")
        determinants: list[np.ndarray] = []
        scaled: list[np.ndarray] = []
        volume_elements = 0
        for block in self.volume_blocks:
            tets, _ = _volume_block_to_tets(block)
            volume_elements += block.n_elements
            x = self.nodes[tets]
            edges = np.stack(
                (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
                axis=1,
            )
            det = np.linalg.det(edges)
            denominator = np.prod(np.linalg.norm(edges, axis=2), axis=1)
            sj = np.divide(
                det, denominator, out=np.zeros_like(det),
                where=denominator > np.finfo(float).tiny,
            )
            determinants.append(det)
            scaled.append(sj)
        if not determinants:
            raise MeshError("mesh has no volume elements")
        determinant = np.concatenate(determinants)
        scaled_jacobian = np.concatenate(scaled)
        if np.any(~np.isfinite(determinant)) or np.any(~np.isfinite(scaled_jacobian)):
            raise MeshError("mesh quality contains non-finite values")
        volume = determinant / 6.0
        degenerate = int(np.count_nonzero(np.abs(volume) <= tolerance))
        inverted = int(np.count_nonzero(volume < -tolerance))
        absolute_volume = np.abs(volume)
        return MeshQualityReport(
            volume_element_count=int(volume_elements),
            sub_tetrahedron_count=int(len(volume)),
            total_volume=float(np.sum(absolute_volume)),
            minimum_subvolume=float(np.min(absolute_volume)),
            maximum_subvolume=float(np.max(absolute_volume)),
            mean_subvolume=float(np.mean(absolute_volume)),
            minimum_scaled_jacobian=float(np.min(scaled_jacobian)),
            mean_scaled_jacobian=float(np.mean(scaled_jacobian)),
            degenerate_subcells=degenerate,
            inverted_subcells=inverted,
        )

    def validate_positive_volume(self, *, tolerance: float | None = None) -> None:
        report = self.quality_report(tolerance=tolerance)
        if report.degenerate_subcells:
            raise MeshError(
                f"mesh contains {report.degenerate_subcells} degenerate subcells"
            )
        if report.inverted_subcells:
            raise MeshError(
                f"mesh contains {report.inverted_subcells} inverted subcells"
            )

    def as_tet4(self) -> Mesh:
        cells: list[np.ndarray] = []
        regions: list[np.ndarray] = []
        for block in self.volume_blocks:
            if block.element_type == "tet4":
                cells.append(block.connectivity)
                regions.append(block.entity_ids)
            elif block.element_type in {"hex8", "wedge6", "pyramid5"}:
                converted, multiplier = _volume_block_to_tets(block)
                cells.append(converted)
                regions.append(np.repeat(block.entity_ids, multiplier))
            else:  # pragma: no cover - all registered volume types handled above
                raise MeshError(f"automatic Tet4 conversion is not implemented for {block.element_type}")
        if not cells:
            raise MeshError("mesh has no volume elements")
        boundaries: dict[str, np.ndarray] = {}
        for index, block in enumerate(self.boundary_blocks):
            name = block.name or f"boundary_{index}"
            if block.element_type == "tri3":
                boundaries[name] = block.connectivity
            elif block.element_type == "quad4":
                boundaries[name] = _quad_to_triangles(block.connectivity)
        return Mesh(
            nodes=self.nodes.copy(),
            cells=np.concatenate(cells, axis=0),
            cell_regions=np.concatenate(regions).astype(np.int32, copy=False),
            boundary_faces=boundaries,
            metadata={**self.metadata, "converted_from_general_mesh": True},
        )


def voxel_to_general_mesh(
    phase_labels: np.ndarray,
    *,
    voxel_size_m: float | tuple[float, float, float] = 1.0,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    element_type: str = "hex8",
    active_labels: Sequence[int] | None = None,
    include_exterior_boundaries: bool = True,
    include_material_interfaces: bool = False,
) -> GeneralMesh:
    """Create a conforming Hex8 or Tet4 mesh from a voxel label field."""

    labels = np.ascontiguousarray(phase_labels, dtype=np.int32)
    if labels.ndim != 3 or min(labels.shape) < 1:
        raise MeshError("phase_labels must be a non-empty 3-D array")
    spacing = _normalize3(voxel_size_m)
    origin = _origin3(origin_m)
    kind = str(element_type).lower()
    if kind not in {"hex8", "tet4"}:
        raise MeshError("voxel meshing supports element_type='hex8' or 'tet4'")
    nx, ny, nz = map(int, labels.shape)
    gx, gy, gz = np.meshgrid(
        origin[0] + spacing[0] * np.arange(nx + 1),
        origin[1] + spacing[1] * np.arange(ny + 1),
        origin[2] + spacing[2] * np.arange(nz + 1),
        indexing="ij",
    )
    nodes = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    node_ids = np.arange((nx + 1) * (ny + 1) * (nz + 1), dtype=np.int64).reshape(nx + 1, ny + 1, nz + 1)
    hexes = np.stack(
        (
            node_ids[:-1, :-1, :-1],
            node_ids[1:, :-1, :-1],
            node_ids[1:, 1:, :-1],
            node_ids[:-1, 1:, :-1],
            node_ids[:-1, :-1, 1:],
            node_ids[1:, :-1, 1:],
            node_ids[1:, 1:, 1:],
            node_ids[:-1, 1:, 1:],
        ),
        axis=-1,
    ).reshape(-1, 8)
    phase_values = labels.ravel()
    if active_labels is not None:
        keep = np.isin(phase_values, np.asarray(tuple(active_labels), dtype=np.int32))
        hexes = hexes[keep]
        phase_values = phase_values[keep]
    # COMSOL geometric entities are positive and are not the same object as a
    # user phase label (which commonly starts at zero).  Preserve this mapping
    # explicitly instead of leaking raw phase labels into the mesh topology.
    phase_ids = tuple(sorted(map(int, np.unique(phase_values))))
    phase_to_entity = {phase: index for index, phase in enumerate(phase_ids, start=1)}
    entities = np.fromiter((phase_to_entity[int(value)] for value in phase_values), dtype=np.int32, count=len(phase_values))
    blocks: list[ElementBlock] = []
    if kind == "hex8":
        blocks.append(ElementBlock("hex8", hexes, entities, name="volume"))
    else:
        blocks.append(ElementBlock("tet4", _hex_to_tets(hexes), np.repeat(entities, 6), name="volume"))
    if include_exterior_boundaries and active_labels is None:
        boundaries = _structured_boundaries(node_ids)
        if kind == "tet4":
            boundaries = [_triangulated_boundary_block(block) for block in boundaries]
        blocks.extend(boundaries)
    interface_entities: dict[tuple[int, int], int] = {}
    if include_material_interfaces and active_labels is None:
        interface_block, interface_entities = _structured_material_interfaces(
            node_ids, labels, first_entity=7
        )
        if interface_block is not None:
            if kind == "tet4":
                interface_block = _triangulated_boundary_block(interface_block)
            blocks.append(interface_block)
    selections = {f"domain_{phase}": (3, (phase_to_entity[phase],)) for phase in phase_ids}
    if include_exterior_boundaries and active_labels is None:
        for index, name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), start=1):
            selections[name] = (2, (index,))
    for pair, entity in interface_entities.items():
        selections[f"interface_{pair[0]}_{pair[1]}"] = (2, (entity,))
    mesh = GeneralMesh(
        nodes=nodes,
        blocks=blocks,
        selections=selections,
        metadata={
            "source": "voxel_to_general_mesh",
            "voxel_shape": tuple(map(int, labels.shape)),
            "voxel_size_m": spacing,
            "origin_m": origin,
            "phase_to_entity": dict(phase_to_entity),
            "entity_to_phase": {entity: phase for phase, entity in phase_to_entity.items()},
            "interface_pair_to_entity": dict(interface_entities),
        },
    )
    mesh.validate_positive_volume()
    return mesh


def _structured_boundaries(node_ids: np.ndarray) -> list[ElementBlock]:
    def quads_on(axis: int, index: int, entity: int, name: str) -> ElementBlock:
        plane = np.take(node_ids, index, axis=axis)
        quads = np.stack(
            (plane[:-1, :-1], plane[1:, :-1], plane[1:, 1:], plane[:-1, 1:]),
            axis=-1,
        ).reshape(-1, 4)
        return ElementBlock("quad4", quads, np.full(len(quads), entity, dtype=np.int32), name=name)

    return [
        quads_on(0, 0, 1, "xmin"),
        quads_on(0, -1, 2, "xmax"),
        quads_on(1, 0, 3, "ymin"),
        quads_on(1, -1, 4, "ymax"),
        quads_on(2, 0, 5, "zmin"),
        quads_on(2, -1, 6, "zmax"),
    ]



def _structured_material_interfaces(
    node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    first_entity: int,
) -> tuple[ElementBlock | None, dict[tuple[int, int], int]]:
    """Extract conforming Quad4 faces at every unlike-label adjacency."""

    rows: list[np.ndarray] = []
    phase_pairs: list[np.ndarray] = []
    for axis in range(3):
        left_slices = [slice(None)] * 3
        right_slices = [slice(None)] * 3
        left_slices[axis] = slice(None, -1)
        right_slices[axis] = slice(1, None)
        left = labels[tuple(left_slices)]
        right = labels[tuple(right_slices)]
        local = np.argwhere(left != right)
        if not len(local):
            continue
        a = left[tuple(local.T)].astype(np.int64, copy=False)
        b = right[tuple(local.T)].astype(np.int64, copy=False)
        phase_pairs.append(np.column_stack((np.minimum(a, b), np.maximum(a, b))))
        i, j, k = local[:, 0], local[:, 1], local[:, 2]
        if axis == 0:
            plane = i + 1
            quads = np.column_stack((
                node_ids[plane, j, k], node_ids[plane, j + 1, k],
                node_ids[plane, j + 1, k + 1], node_ids[plane, j, k + 1],
            ))
        elif axis == 1:
            plane = j + 1
            quads = np.column_stack((
                node_ids[i, plane, k], node_ids[i + 1, plane, k],
                node_ids[i + 1, plane, k + 1], node_ids[i, plane, k + 1],
            ))
        else:
            plane = k + 1
            quads = np.column_stack((
                node_ids[i, j, plane], node_ids[i + 1, j, plane],
                node_ids[i + 1, j + 1, plane], node_ids[i, j + 1, plane],
            ))
        rows.append(np.ascontiguousarray(quads, dtype=np.int64))
    if not rows:
        return None, {}
    pairs = np.concatenate(phase_pairs, axis=0)
    unique_pairs = sorted({(int(a), int(b)) for a, b in pairs})
    mapping = {pair: first_entity + index for index, pair in enumerate(unique_pairs)}
    entities = np.fromiter(
        (mapping[(int(a), int(b))] for a, b in pairs),
        dtype=np.int32, count=len(pairs),
    )
    return (
        ElementBlock(
            "quad4", np.concatenate(rows, axis=0), entities,
            name="material_interfaces",
        ),
        mapping,
    )


def _triangulated_boundary_block(block: ElementBlock) -> ElementBlock:
    if block.element_type != "quad4":
        return block
    return ElementBlock(
        "tri3",
        _quad_to_triangles(block.connectivity),
        np.repeat(block.entity_ids, 2),
        name=block.name,
        metadata={**block.metadata, "triangulated_from": "quad4"},
    )

def _hex_to_tets(hexes: np.ndarray) -> np.ndarray:
    pattern = np.asarray(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    return np.ascontiguousarray(np.asarray(hexes, dtype=np.int64)[:, pattern].reshape(-1, 4))



def _wedge_to_tets(wedges: np.ndarray) -> np.ndarray:
    pattern = np.asarray(
        [[0, 1, 2, 3], [1, 4, 2, 3], [2, 4, 5, 3]], dtype=np.int64
    )
    return np.ascontiguousarray(np.asarray(wedges, dtype=np.int64)[:, pattern].reshape(-1, 4))


def _pyramid_to_tets(pyramids: np.ndarray) -> np.ndarray:
    pattern = np.asarray([[0, 1, 2, 4], [0, 2, 3, 4]], dtype=np.int64)
    return np.ascontiguousarray(np.asarray(pyramids, dtype=np.int64)[:, pattern].reshape(-1, 4))


def _quad_to_triangles(quads: np.ndarray) -> np.ndarray:
    pattern = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return np.ascontiguousarray(np.asarray(quads, dtype=np.int64)[:, pattern].reshape(-1, 3))


def _volume_block_to_tets(block: ElementBlock) -> tuple[np.ndarray, int]:
    if block.element_type == "tet4":
        return block.connectivity, 1
    if block.element_type == "hex8":
        return _hex_to_tets(block.connectivity), 6
    if block.element_type == "wedge6":
        return _wedge_to_tets(block.connectivity), 3
    if block.element_type == "pyramid5":
        return _pyramid_to_tets(block.connectivity), 2
    raise MeshError(f"{block.element_type} is not a supported volume element")


def _origin3(value: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(value):
        result = (float(value),) * 3
    else:
        result = tuple(map(float, value))
    if len(result) != 3 or not np.isfinite(result).all():
        raise ValueError("origin must contain three finite values")
    return result

def _normalize3(value: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(value):
        result = (float(value),) * 3
    else:
        result = tuple(map(float, value))
    if len(result) != 3 or not np.isfinite(result).all() or min(result) <= 0.0:
        raise ValueError("expected three positive finite values")
    return result  # type: ignore[return-value]


__all__ = [
    "ElementBlock",
    "GeneralMesh",
    "voxel_to_general_mesh",
]
