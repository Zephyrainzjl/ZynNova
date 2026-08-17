"""Optional advanced surface repair before FEM tetrahedralization."""

from __future__ import annotations

import numpy as np

from ..geometry import TriangleMesh, clean_triangle_mesh


def repair_surface_for_fem(mesh: TriangleMesh) -> TriangleMesh:
    """Repair duplicate/degenerate geometry and attempt small-hole closure.

    The dependency-light cleanup is always applied. If trimesh is installed, its
    topology repair routines are used as a second stage without remeshing textures.
    """

    cleaned, _ = clean_triangle_mesh(mesh, weld_tolerance=1.0e-9)
    try:
        import trimesh
    except ImportError:
        return cleaned
    value = trimesh.Trimesh(
        vertices=cleaned.vertices,
        faces=cleaned.faces,
        process=False,
    )
    try:
        value.merge_vertices()
        if hasattr(value, "remove_degenerate_faces"):
            value.remove_degenerate_faces()
        if hasattr(value, "remove_duplicate_faces"):
            value.remove_duplicate_faces()
        value.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(value, multibody=True)
        trimesh.repair.fill_holes(value)
    except (ValueError, RuntimeError, AttributeError):
        return cleaned
    colors = None
    try:
        raw = np.asarray(value.visual.vertex_colors)
        if raw.shape[0] == len(value.vertices) and raw.shape[1] >= 3:
            colors = raw[:, :3]
    except (AttributeError, ValueError):
        pass
    return TriangleMesh(
        vertices=np.asarray(value.vertices),
        faces=np.asarray(value.faces),
        vertex_colors=colors,
        vertex_normals=(
            np.asarray(value.vertex_normals) if len(value.vertex_normals) else None
        ),
        metadata={**cleaned.metadata, "advanced_repair": "trimesh"},
    )


__all__ = ["repair_surface_for_fem"]
