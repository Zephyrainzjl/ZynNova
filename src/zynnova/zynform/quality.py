"""Surface evidence used before render export and FEM conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import TriangleMesh, triangle_quality


@dataclass(frozen=True, slots=True)
class SurfaceAudit:
    vertices: int
    triangles: int
    connected_components: int
    watertight: bool
    boundary_edges: int
    nonmanifold_edges: int
    degenerate_faces: int
    area: float
    signed_volume: float
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    minimum_edge_length: float
    median_edge_length: float
    maximum_edge_length: float
    finite: bool


def audit_surface(mesh: TriangleMesh) -> SurfaceAudit:
    base = triangle_quality(mesh)
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    signed_volume = float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6.0
    )
    edges = np.concatenate(
        [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]], axis=0
    )
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    minimum = np.min(mesh.vertices, axis=0)
    maximum = np.max(mesh.vertices, axis=0)
    return SurfaceAudit(
        vertices=mesh.n_vertices,
        triangles=mesh.n_faces,
        connected_components=_face_components(mesh.faces),
        watertight=base.watertight,
        boundary_edges=base.boundary_edges,
        nonmanifold_edges=base.nonmanifold_edges,
        degenerate_faces=base.degenerate_faces,
        area=area,
        signed_volume=signed_volume,
        bounds_min_xyz=tuple(float(v) for v in minimum),
        bounds_max_xyz=tuple(float(v) for v in maximum),
        extent_xyz=tuple(float(v) for v in maximum - minimum),
        minimum_edge_length=float(np.min(lengths)) if len(lengths) else 0.0,
        median_edge_length=float(np.median(lengths)) if len(lengths) else 0.0,
        maximum_edge_length=float(np.max(lengths)) if len(lengths) else 0.0,
        finite=bool(np.all(np.isfinite(mesh.vertices))),
    )


def _face_components(faces: np.ndarray) -> int:
    if not len(faces):
        return 0
    edge_owner: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_owner.setdefault(edge, []).append(fi)
    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    for owners in edge_owner.values():
        if len(owners) > 1:
            root = owners[0]
            for other in owners[1:]:
                adjacency[root].append(other)
                adjacency[other].append(root)
    unseen = set(range(len(faces)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


__all__ = ["SurfaceAudit", "audit_surface"]
