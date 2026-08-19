"""Physical-scale transforms that preserve the high-fidelity native render asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import TriangleMesh


@dataclass(frozen=True, slots=True)
class PhysicalScaleTransform:
    matrix: np.ndarray
    source_extent: float
    target_extent_m: float
    scale_factor: float
    source_center_xyz: tuple[float, float, float]


def compute_physical_scale_transform(
    mesh: TriangleMesh,
    target_extent_m: float,
) -> PhysicalScaleTransform:
    if target_extent_m <= 0.0:
        raise ValueError("target_extent_m must be positive")
    if mesh.n_vertices == 0:
        raise ValueError("cannot scale an empty mesh")
    minimum = np.min(mesh.vertices, axis=0)
    maximum = np.max(mesh.vertices, axis=0)
    centre = 0.5 * (minimum + maximum)
    source_extent = float(np.max(maximum - minimum))
    if not np.isfinite(source_extent) or source_extent <= 0.0:
        raise ValueError("mesh has zero or non-finite spatial extent")
    factor = float(target_extent_m / source_extent)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= factor
    matrix[:3, 3] = -factor * centre
    return PhysicalScaleTransform(
        matrix=matrix,
        source_extent=source_extent,
        target_extent_m=float(target_extent_m),
        scale_factor=factor,
        source_center_xyz=tuple(float(v) for v in centre),
    )


def apply_physical_scale(
    mesh: TriangleMesh,
    transform: PhysicalScaleTransform,
) -> TriangleMesh:
    vertices = (
        mesh.vertices @ transform.matrix[:3, :3].T
        + transform.matrix[:3, 3]
    )
    normals = mesh.vertex_normals
    if normals is not None:
        linear = transform.matrix[:3, :3]
        inverse_transpose = np.linalg.inv(linear).T
        normals = normals @ inverse_transpose.T
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0.0)
    return TriangleMesh(
        vertices=vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=normals,
        uv=mesh.uv,
        face_materials=mesh.face_materials,
        texture_paths=mesh.texture_paths,
        metadata={
            **mesh.metadata,
            "physical_units": "meter",
            "physical_extent_m": transform.target_extent_m,
            "physical_scale_factor": transform.scale_factor,
        },
    )


def transform_native_asset(
    source: str | Path,
    destination: str | Path,
    transform: PhysicalScaleTransform,
) -> Path | None:
    """Apply the same metric transform to a native textured asset when possible.

    This path intentionally uses the native scene rather than recreating a mesh from
    ZynNova arrays, so GLB/GLTF material graphs, UVs, textures and scene hierarchy have
    a chance to survive.  Failure is explicit to the caller via ``None``; the raw
    backend asset should still be preserved separately.
    """

    try:
        import trimesh
    except ImportError:
        return None
    source = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        scene = trimesh.load(source, force="scene", process=False)
        scene.apply_transform(transform.matrix)
        exported = scene.export(file_type=target.suffix.lower().lstrip("."))
        if isinstance(exported, dict):
            # GLTF can be multi-file.  Prefer a self-contained GLB in the canonical
            # pipeline; for a dict export, write every referenced file next to target.
            for name, payload in exported.items():
                path = target.parent / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(payload, str):
                    path.write_text(payload, encoding="utf-8")
                else:
                    path.write_bytes(payload)
            candidate = target if target.is_file() else target.parent / target.name
            return candidate if candidate.is_file() else None
        if isinstance(exported, str):
            target.write_text(exported, encoding="utf-8")
        else:
            target.write_bytes(exported)
        return target if target.is_file() else None
    except Exception:
        return None


__all__ = [
    "PhysicalScaleTransform",
    "apply_physical_scale",
    "compute_physical_scale_transform",
    "transform_native_asset",
]
