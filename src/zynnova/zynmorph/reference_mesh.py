"""Reference Tet4 mesh import and mesh-style profiling.

The profiler is intentionally geometry-agnostic: a COMSOL Tet4 mesh can be
used as a sizing reference for a completely different free-form PLC.  It
reports edge-length and tetra-volume distributions globally and per domain,
which can then seed TetGen region volume constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.exceptions import GeometryError
from ..geometry import VolumeMesh, tetrahedron_signed_volumes
from .tetgen import TetGenMeshingConfig


@dataclass(frozen=True, slots=True)
class RegionMeshProfile:
    region: int
    tetrahedra: int
    total_volume_m3: float
    edge_length_quantiles_m: tuple[float, ...]
    tetra_volume_quantiles_m3: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReferenceMeshProfile:
    nodes: int
    tetrahedra: int
    regions: tuple[int, ...]
    bbox_min_m_xyz: tuple[float, float, float]
    bbox_max_m_xyz: tuple[float, float, float]
    edge_length_quantiles_m: tuple[float, ...]
    tetra_volume_quantiles_m3: tuple[float, ...]
    region_profiles: Mapping[int, RegionMeshProfile] = field(default_factory=dict)
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    metadata: Mapping[str, Any] = field(default_factory=dict)




def _parse_zynnova_region_metadata(lines: list[str]) -> tuple[dict[int, int], dict[int, str]]:
    """Read optional ZynNova material metadata embedded as MPHTXT comments.

    COMSOL external meshes require positive geometric entity IDs, while
    ZynNova material labels may contain zero or arbitrary integers.  New files
    therefore carry a reversible mapping in comments that COMSOL safely
    ignores.  Legacy/external MPHTXT files simply return empty mappings.
    """

    prefix_map = "# ZYNNOVA_REGION_ENTITY_MAP "
    prefix_names = "# ZYNNOVA_REGION_NAMES "
    region_entity_map: dict[int, int] = {}
    region_names: dict[int, str] = {}
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(prefix_map):
            payload = json.loads(stripped[len(prefix_map):])
            region_entity_map = {int(k): int(v) for k, v in payload.items()}
        elif stripped.startswith(prefix_names):
            payload = json.loads(stripped[len(prefix_names):])
            region_names = {int(k): str(v) for k, v in payload.items()}
    if region_entity_map:
        entities = tuple(region_entity_map.values())
        if len(set(entities)) != len(entities) or any(entity <= 0 for entity in entities):
            raise GeometryError("invalid ZynNova region/entity metadata in MPHTXT")
    return region_entity_map, region_names

def _data_part(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _comment_part(line: str) -> str:
    return line.split("#", 1)[1].strip().lower() if "#" in line else ""


def _next_data(lines: list[str], index: int) -> tuple[str, int]:
    while index < len(lines):
        value = _data_part(lines[index])
        index += 1
        if value:
            return value, index
    raise GeometryError("unexpected end of COMSOL MPHTXT")


def _value_for_comment(
    lines: list[str],
    start: int,
    stop: int,
    phrases: tuple[str, ...],
) -> tuple[str, int] | None:
    phrases = tuple(item.lower() for item in phrases)
    for index in range(start, stop):
        line = lines[index]
        comment = _comment_part(line)
        stripped = line.strip().lower()
        if not any(phrase in comment or phrase in stripped for phrase in phrases):
            continue
        data = _data_part(line)
        if data:
            return data, index + 1
        return _next_data(lines, index + 1)
    return None


def load_comsol_tet4_mphtxt(path: str | Path) -> VolumeMesh:
    """Load the first COMSOL Mesh-v4 Tet4 object from an MPHTXT file.

    Both common native-text spellings are accepted: values may precede an
    inline comment (``3606 # number of mesh vertices``) or follow a comment on
    the next line (``# num of verts`` then ``3606``).  Mixed volume cell types
    are intentionally rejected rather than guessed.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    region_entity_map, embedded_region_names = _parse_zynnova_region_metadata(lines)

    try:
        coord_comment = next(
            i for i, line in enumerate(lines)
            if "mesh vertex coordinates" in line.strip().lower()
        )
    except StopIteration as exc:
        raise GeometryError("MPHTXT contains no Mesh vertex coordinate block") from exc

    nvalue = _value_for_comment(
        lines, max(0, coord_comment - 24), coord_comment,
        ("number of mesh vertices", "num of verts"),
    )
    svalue = _value_for_comment(
        lines, max(0, coord_comment - 24), coord_comment,
        ("start vertex index", "start vert idx"),
    )
    if nvalue is None or svalue is None:
        raise GeometryError("MPHTXT vertex count/start index could not be resolved")
    nverts = int(nvalue[0].split()[0])
    start_index = int(svalue[0].split()[0])

    vertices = []
    cursor = coord_comment + 1
    while len(vertices) < nverts:
        value, cursor = _next_data(lines, cursor)
        parts = value.split()
        if len(parts) < 3:
            raise GeometryError("invalid MPHTXT vertex row")
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    points = np.asarray(vertices, dtype=np.float64)

    tet_line = None
    for index in range(cursor, len(lines)):
        data = _data_part(lines[index]).split()
        if len(data) >= 2 and data[1].lower() == "tet":
            tet_line = index
            break
    if tet_line is None:
        raise GeometryError("MPHTXT contains no Tet4 element block")

    width_value = _value_for_comment(
        lines, tet_line + 1, min(len(lines), tet_line + 16),
        ("number of vertices per element", "number of vertice per element"),
    )
    if width_value is None or int(width_value[0].split()[0]) != 4:
        raise GeometryError("only four-node tetrahedra are supported")

    ne_value = _value_for_comment(
        lines, tet_line + 1, min(len(lines), tet_line + 24),
        ("number of elements",),
    )
    if ne_value is None:
        raise GeometryError("Tet4 element count could not be resolved")
    nelements = int(ne_value[0].split()[0])

    elements_comment = next(
        (
            i for i in range(tet_line, min(len(lines), tet_line + 32))
            if lines[i].strip().lower() in {"# elements", "# elements "}
        ),
        None,
    )
    if elements_comment is None:
        raise GeometryError("Tet4 connectivity marker is missing")
    cursor = elements_comment + 1
    elements = []
    while len(elements) < nelements:
        value, cursor = _next_data(lines, cursor)
        row = [int(item) - start_index for item in value.split()[:4]]
        if len(row) != 4:
            raise GeometryError("invalid Tet4 connectivity row")
        elements.append(row)
    tetrahedra = np.asarray(elements, dtype=np.int64)

    entity_comment = next(
        (
            i for i in range(cursor, min(len(lines), cursor + 32))
            if "entity indices" in lines[i].strip().lower()
            and "number" not in lines[i].strip().lower()
        ),
        None,
    )
    if entity_comment is None:
        regions = np.ones(nelements, dtype=np.int32)
    else:
        count_value = _value_for_comment(
            lines, max(cursor, entity_comment - 8), entity_comment,
            ("number of geometric entity indices", "number of entities"),
        )
        if count_value is None:
            raise GeometryError("MPHTXT entity indices have no element count")
        if int(count_value[0].split()[0]) != nelements:
            raise GeometryError("MPHTXT entity-index count differs from Tet4 count")
        cursor = entity_comment + 1
        values = []
        while len(values) < nelements:
            value, cursor = _next_data(lines, cursor)
            values.append(int(value.split()[0]))
        regions = np.asarray(values, dtype=np.int32)

    # Entity indices in COMSOL are positive and contiguous; recover the
    # original ZynNova material IDs when the writer embedded a reversible map.
    inverse_entity_map = {entity: region for region, entity in region_entity_map.items()}
    if inverse_entity_map:
        unknown = sorted(set(map(int, np.unique(regions))) - set(inverse_entity_map))
        if unknown:
            raise GeometryError(
                "MPHTXT contains COMSOL domain entities not present in its "
                f"embedded ZynNova region map: {unknown}"
            )
        regions = np.asarray(
            [inverse_entity_map[int(entity)] for entity in regions],
            dtype=np.int32,
        )

    unique_regions = tuple(map(int, np.unique(regions)))
    resolved_names = {
        region: embedded_region_names.get(region, f"domain_{region}")
        for region in unique_regions
    }
    mesh = VolumeMesh(
        nodes=points,
        tetrahedra=tetrahedra,
        cell_regions=regions,
        region_names=resolved_names,
        metadata={
            "source": "comsol-mphtxt",
            "source_path": str(source),
            "comsol_region_entity_map": dict(region_entity_map),
            "restored_original_regions": bool(region_entity_map),
        },
    )
    signed = tetrahedron_signed_volumes(mesh)
    negative = signed < 0.0
    if np.any(negative):
        tetrahedra = tetrahedra.copy()
        tetrahedra[negative, 0], tetrahedra[negative, 1] = (
            tetrahedra[negative, 1].copy(), tetrahedra[negative, 0].copy()
        )
        mesh = VolumeMesh(
            nodes=points,
            tetrahedra=tetrahedra,
            cell_regions=regions,
            region_names=mesh.region_names,
            metadata={
                **mesh.metadata,
                "reoriented_negative_tets": int(np.count_nonzero(negative)),
            },
        )
    return mesh


def _tetra_volumes(mesh: VolumeMesh) -> np.ndarray:
    t = mesh.nodes[mesh.tetrahedra]
    return np.abs(
        np.einsum("ij,ij->i", t[:, 1] - t[:, 0], np.cross(t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
    ) / 6.0


def _cell_edge_lengths(mesh: VolumeMesh, mask: np.ndarray | None = None) -> np.ndarray:
    cells = mesh.tetrahedra if mask is None else mesh.tetrahedra[mask]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return np.concatenate(
        [np.linalg.norm(mesh.nodes[cells[:, i]] - mesh.nodes[cells[:, j]], axis=1) for i, j in pairs]
    )


def profile_reference_mesh(
    mesh_or_path: VolumeMesh | str | Path,
    *,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
) -> ReferenceMeshProfile:
    mesh = load_comsol_tet4_mphtxt(mesh_or_path) if isinstance(mesh_or_path, (str, Path)) else mesh_or_path
    q = tuple(float(value) for value in quantiles)
    if not q or any(not 0.0 <= value <= 1.0 for value in q):
        raise ValueError("quantiles must lie in [0, 1]")
    volumes = _tetra_volumes(mesh)
    edges = _cell_edge_lengths(mesh)
    profiles: dict[int, RegionMeshProfile] = {}
    for region in map(int, np.unique(mesh.cell_regions)):
        mask = mesh.cell_regions == region
        region_volumes = volumes[mask]
        region_edges = _cell_edge_lengths(mesh, mask)
        profiles[region] = RegionMeshProfile(
            region=region,
            tetrahedra=int(np.count_nonzero(mask)),
            total_volume_m3=float(region_volumes.sum()),
            edge_length_quantiles_m=tuple(map(float, np.quantile(region_edges, q))),
            tetra_volume_quantiles_m3=tuple(map(float, np.quantile(region_volumes, q))),
        )
    return ReferenceMeshProfile(
        nodes=mesh.n_nodes,
        tetrahedra=mesh.n_cells,
        regions=tuple(map(int, np.unique(mesh.cell_regions))),
        bbox_min_m_xyz=tuple(map(float, np.min(mesh.nodes, axis=0))),
        bbox_max_m_xyz=tuple(map(float, np.max(mesh.nodes, axis=0))),
        edge_length_quantiles_m=tuple(map(float, np.quantile(edges, q))),
        tetra_volume_quantiles_m3=tuple(map(float, np.quantile(volumes, q))),
        region_profiles=profiles,
        quantiles=q,
        metadata={"source": mesh.metadata.get("source"), "source_path": mesh.metadata.get("source_path")},
    )


def tetgen_config_from_reference(
    profile: ReferenceMeshProfile,
    *,
    region_map: Mapping[int, int] | None = None,
    volume_quantile: float = 0.95,
    linear_scale: float = 1.0,
    radius_edge_ratio: float | None = None,
    minimum_dihedral_degrees: float | None = None,
    base: TetGenMeshingConfig | None = None,
) -> TetGenMeshingConfig:
    """Create region volume limits that reproduce a reference mesh scale.

    ``linear_scale`` rescales target edge lengths; tetra volume constraints are
    therefore scaled by its cube.  ``region_map`` maps target-region IDs to
    reference-region IDs when the new geometry uses different labels.
    """

    if not np.isfinite(linear_scale) or linear_scale <= 0.0:
        raise ValueError("linear_scale must be positive and finite")
    q = float(volume_quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("volume_quantile must lie in [0, 1]")
    mapping = dict(region_map or {region: region for region in profile.regions})
    scale3 = float(linear_scale) ** 3
    limits: dict[int, float] = {}
    for target, reference in mapping.items():
        if int(reference) not in profile.region_profiles:
            raise ValueError(f"reference region {reference} is absent from the mesh profile")
        rp = profile.region_profiles[int(reference)]
        # Interpolate the requested quantile from the stored profile quantile grid.
        value = float(np.interp(q, profile.quantiles, rp.tetra_volume_quantiles_m3))
        limits[int(target)] = value * scale3
    seed = base or TetGenMeshingConfig()
    resolved_ratio = seed.radius_edge_ratio if radius_edge_ratio is None else float(radius_edge_ratio)
    resolved_dihedral = (
        seed.minimum_dihedral_degrees
        if minimum_dihedral_degrees is None
        else float(minimum_dihedral_degrees)
    )
    return replace(
        seed,
        radius_edge_ratio=resolved_ratio,
        minimum_dihedral_degrees=resolved_dihedral,
        phase_maximum_tetra_volume_m3={
            **seed.phase_maximum_tetra_volume_m3,
            **limits,
        },
    )


__all__ = [
    "ReferenceMeshProfile",
    "RegionMeshProfile",
    "load_comsol_tet4_mphtxt",
    "profile_reference_mesh",
    "tetgen_config_from_reference",
]
