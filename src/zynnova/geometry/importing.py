"""Dependency-light mesh import with optional trimesh escalation."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..core.exceptions import BackendUnavailableError, GeometryError
from .types import TriangleMesh


def load_triangle_mesh(path: str | Path) -> TriangleMesh:
    """Load one triangle mesh from common interchange formats.

    OBJ, ASCII PLY, ASCII/binary STL, and ZynNova NPZ are parsed directly. GLB,
    glTF, FBX, USD, DAE, and complex textured scenes are delegated to trimesh.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".obj":
        return _load_obj(source)
    if suffix == ".ply":
        return _load_ply(source)
    if suffix == ".stl":
        return _load_stl(source)
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            return TriangleMesh(
                vertices=data["vertices"],
                faces=data["faces"],
                vertex_colors=_optional_npz(data, "vertex_colors"),
                vertex_normals=_optional_npz(data, "vertex_normals"),
                uv=_optional_npz(data, "uv"),
                face_materials=_optional_npz(data, "face_materials"),
                metadata={"source_path": str(source), "importer": "numpy"},
            )
    return _load_with_trimesh(source)


def _optional_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    value = data[key]
    if value.dtype == object and value.shape == () and value.item() is None:
        return None
    return value


def _load_obj(path: Path) -> TriangleMesh:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_uv: list[list[int | None]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 7:
                colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
        elif parts[0] == "vt" and len(parts) >= 3:
            texcoords.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f" and len(parts) >= 4:
            polygon: list[int] = []
            uv_polygon: list[int | None] = []
            for token in parts[1:]:
                chunks = token.split("/")
                index = int(chunks[0])
                polygon.append(index - 1 if index > 0 else len(vertices) + index)
                if len(chunks) > 1 and chunks[1]:
                    uv_index = int(chunks[1])
                    uv_polygon.append(
                        uv_index - 1 if uv_index > 0 else len(texcoords) + uv_index
                    )
                else:
                    uv_polygon.append(None)
            for offset in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
                face_uv.append([uv_polygon[0], uv_polygon[offset], uv_polygon[offset + 1]])
    if not vertices or not faces:
        raise GeometryError(f"OBJ contains no triangle geometry: {path}")
    vertex_colors = np.asarray(colors, dtype=float) if len(colors) == len(vertices) else None
    # A per-vertex UV field is valid only when every referenced position has one stable UV.
    uv = None
    if texcoords and all(item is not None for tri in face_uv for item in tri):
        assigned: dict[int, int] = {}
        consistent = True
        for face, tri_uv in zip(faces, face_uv, strict=True):
            for vertex, uv_index in zip(face, tri_uv, strict=True):
                assert uv_index is not None
                old = assigned.setdefault(vertex, uv_index)
                consistent &= old == uv_index
        if consistent and len(assigned) == len(vertices):
            uv = np.asarray([texcoords[assigned[i]] for i in range(len(vertices))])
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=vertex_colors,
        uv=uv,
        metadata={"source_path": str(path), "importer": "obj"},
    )


def _load_ply(path: Path) -> TriangleMesh:
    with path.open("rb") as stream:
        first = stream.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise GeometryError(f"not a PLY file: {path}")
        header: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise GeometryError("truncated PLY header")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break
        format_line = next((line for line in header if line.startswith("format ")), "")
        if "ascii" not in format_line:
            return _load_with_trimesh(path)
        vertex_count = 0
        face_count = 0
        vertex_properties: list[str] = []
        element: str | None = None
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                element = "vertex"
            elif parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
                element = "face"
            elif parts and parts[0] == "element":
                element = parts[1]
            elif parts and parts[0] == "property" and element == "vertex":
                vertex_properties.append(parts[-1])
        rows = [stream.readline().decode("ascii").split() for _ in range(vertex_count)]
        if len(rows) != vertex_count or any(not row for row in rows):
            raise GeometryError("truncated PLY vertices")
        values = np.asarray(rows, dtype=float)
        indices = {name: i for i, name in enumerate(vertex_properties)}
        try:
            vertices = values[:, [indices[axis] for axis in ("x", "y", "z")]]
        except KeyError as exc:
            raise GeometryError("PLY lacks x/y/z vertex properties") from exc
        colors = None
        if all(name in indices for name in ("red", "green", "blue")):
            colors = values[:, [indices[name] for name in ("red", "green", "blue")]]
        normals = None
        if all(name in indices for name in ("nx", "ny", "nz")):
            normals = values[:, [indices[name] for name in ("nx", "ny", "nz")]]
        faces: list[list[int]] = []
        for _ in range(face_count):
            parts = stream.readline().decode("ascii").split()
            if not parts:
                raise GeometryError("truncated PLY faces")
            count = int(parts[0])
            polygon = [int(item) for item in parts[1 : 1 + count]]
            for offset in range(1, count - 1):
                faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
    if not faces:
        raise GeometryError(f"PLY contains no faces: {path}")
    return TriangleMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=colors,
        vertex_normals=normals,
        metadata={"source_path": str(path), "importer": "ply-ascii"},
    )


def _load_stl(path: Path) -> TriangleMesh:
    data = path.read_bytes()
    is_ascii = data[:5].lower() == b"solid" and b"facet" in data[:1024]
    triangles: list[np.ndarray] = []
    if is_ascii:
        current: list[list[float]] = []
        for raw in data.decode("ascii", errors="replace").splitlines():
            parts = raw.strip().split()
            if parts[:1] == ["vertex"] and len(parts) >= 4:
                current.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(current) == 3:
                    triangles.append(np.asarray(current, dtype=float))
                    current = []
    else:
        if len(data) < 84:
            raise GeometryError("truncated binary STL")
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + 50 * count
        if len(data) < expected:
            raise GeometryError("truncated binary STL triangles")
        for index in range(count):
            offset = 84 + 50 * index + 12
            coords = struct.unpack_from("<9f", data, offset)
            triangles.append(np.asarray(coords, dtype=float).reshape(3, 3))
    if not triangles:
        raise GeometryError(f"STL contains no triangles: {path}")
    raw_vertices = np.concatenate(triangles, axis=0)
    vertices, inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return TriangleMesh(
        vertices=vertices,
        faces=faces,
        metadata={"source_path": str(path), "importer": "stl"},
    )


def _load_with_trimesh(path: Path) -> TriangleMesh:
    try:
        import trimesh
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{path.suffix} import requires trimesh; install zynnova[zynnova-geometry]"
        ) from exc
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if len(geometry.faces)]
        if not geometries:
            raise GeometryError(f"scene has no triangle geometry: {path}")
        value = trimesh.util.concatenate(geometries)
    else:
        value = loaded
    colors = None
    try:
        raw_colors = np.asarray(value.visual.vertex_colors)
        if raw_colors.shape[0] == len(value.vertices) and raw_colors.shape[1] >= 3:
            colors = raw_colors[:, :3]
    except (AttributeError, ValueError):
        pass
    normals = np.asarray(value.vertex_normals) if len(value.vertex_normals) else None
    return TriangleMesh(
        vertices=np.asarray(value.vertices),
        faces=np.asarray(value.faces),
        vertex_colors=colors,
        vertex_normals=normals,
        metadata={"source_path": str(path), "importer": "trimesh"},
    )


__all__ = ["load_triangle_mesh"]
