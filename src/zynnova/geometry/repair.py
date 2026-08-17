"""Dependency-light surface cleanup with optional trimesh/pymeshlab escalation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .quality import triangle_areas, triangle_quality
from .types import TriangleMesh


@dataclass(frozen=True, slots=True)
class RepairReport:
    input_vertices: int
    input_faces: int
    output_vertices: int
    output_faces: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    removed_unreferenced_vertices: int
    watertight: bool
    backend: str
    metadata: Mapping[str, Any]


def clean_triangle_mesh(
    mesh: TriangleMesh,
    *,
    area_tolerance: float = 1.0e-14,
    weld_tolerance: float = 0.0,
) -> tuple[TriangleMesh, RepairReport]:
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    face_materials = None if mesh.face_materials is None else mesh.face_materials.copy()
    before_faces = len(faces)

    areas = triangle_areas(mesh)
    keep = areas > area_tolerance
    faces = faces[keep]
    if face_materials is not None:
        face_materials = face_materials[keep]
    removed_degenerate = before_faces - len(faces)

    if weld_tolerance > 0.0 and len(vertices):
        keys = np.rint(vertices / weld_tolerance).astype(np.int64)
        _, unique_indices, inverse = np.unique(
            keys,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        vertices = vertices[np.sort(unique_indices)]
        remap_old_unique = np.empty(len(unique_indices), dtype=np.int64)
        sort_order = np.argsort(unique_indices)
        remap_old_unique[sort_order] = np.arange(len(unique_indices))
        faces = remap_old_unique[inverse[faces]]

    canonical = np.sort(faces, axis=1)
    if len(canonical):
        _, unique_face_indices = np.unique(canonical, axis=0, return_index=True)
        unique_face_indices.sort()
        removed_duplicates = len(faces) - len(unique_face_indices)
        faces = faces[unique_face_indices]
        if face_materials is not None:
            face_materials = face_materials[unique_face_indices]
    else:
        removed_duplicates = 0

    referenced = np.unique(faces.ravel()) if len(faces) else np.empty(0, dtype=np.int64)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced))
    output_vertices = vertices[referenced]
    output_faces = remap[faces] if len(faces) else np.empty((0, 3), dtype=np.int64)
    removed_vertices = len(vertices) - len(output_vertices)

    colors = None
    normals = None
    uv = None
    if mesh.vertex_colors is not None:
        colors = mesh.vertex_colors[referenced]
    if mesh.vertex_normals is not None:
        normals = mesh.vertex_normals[referenced]
    if mesh.uv is not None:
        uv = mesh.uv[referenced]

    cleaned = TriangleMesh(
        vertices=output_vertices,
        faces=output_faces,
        vertex_colors=colors,
        vertex_normals=normals,
        uv=uv,
        face_materials=face_materials,
        texture_paths=mesh.texture_paths,
        metadata={**mesh.metadata, "cleaned": True},
    )
    quality = triangle_quality(cleaned, tolerance=area_tolerance)
    report = RepairReport(
        input_vertices=mesh.n_vertices,
        input_faces=mesh.n_faces,
        output_vertices=cleaned.n_vertices,
        output_faces=cleaned.n_faces,
        removed_degenerate_faces=removed_degenerate,
        removed_duplicate_faces=removed_duplicates,
        removed_unreferenced_vertices=removed_vertices,
        watertight=quality.watertight,
        backend="numpy",
        metadata={"area_tolerance": area_tolerance, "weld_tolerance": weld_tolerance},
    )
    return cleaned, report


def normalize_mesh(
    mesh: TriangleMesh,
    *,
    target_extent: float = 1.0,
    center: bool = True,
) -> TriangleMesh:
    if target_extent <= 0.0:
        raise ValueError("target_extent must be positive")
    vertices = mesh.vertices.copy()
    if not len(vertices):
        return mesh
    if center:
        vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    if extent > 0.0:
        vertices *= target_extent / extent
    return TriangleMesh(
        vertices=vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=mesh.vertex_normals,
        uv=mesh.uv,
        face_materials=mesh.face_materials,
        texture_paths=mesh.texture_paths,
        metadata={**mesh.metadata, "normalized_extent": target_extent},
    )


__all__ = ["RepairReport", "clean_triangle_mesh", "normalize_mesh"]
