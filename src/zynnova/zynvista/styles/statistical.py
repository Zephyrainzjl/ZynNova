"""Fast color-statistics style transfer for dependency-light scene assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core import Availability
from ...core.backend import module_availability
from ...geometry import PointCloud, TriangleMesh
from ..schema import SceneConfig
from ..types import SceneBackendOutput
from .base import SceneStyleBackend


class StatisticalColorStyle(SceneStyleBackend):
    """Apply covariance-aware RGB transfer while preserving all geometry."""

    name = "statistical-color"

    def __init__(self, *, strength: float = 1.0, **_: object) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("style strength must lie in [0,1]")
        self.strength = float(strength)

    def availability(self) -> Availability:
        return module_availability("PIL")

    def apply(
        self,
        output: SceneBackendOutput,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        if config.style_reference is None:
            raise ValueError("statistical-color requires SceneConfig.style_reference")
        from PIL import Image

        reference = np.asarray(
            Image.open(config.style_reference).convert("RGB"), dtype=np.float64
        ) / 255.0
        reference_colors = reference.reshape(-1, 3)
        cloud = output.point_cloud
        mesh = output.mesh
        if cloud is not None and cloud.colors is not None:
            cloud = PointCloud(
                points=cloud.points,
                colors=_transfer(cloud.colors, reference_colors, self.strength),
                normals=cloud.normals,
                confidence=cloud.confidence,
                metadata={**cloud.metadata, "style": self.name},
            )
        if mesh is not None and mesh.vertex_colors is not None:
            mesh = TriangleMesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_colors=_transfer(
                    mesh.vertex_colors, reference_colors, self.strength
                ),
                vertex_normals=mesh.vertex_normals,
                uv=mesh.uv,
                face_materials=mesh.face_materials,
                texture_paths=mesh.texture_paths,
                metadata={**mesh.metadata, "style": self.name},
            )
        if cloud is None and mesh is None:
            raise ValueError("statistical style needs parsed point-cloud or mesh geometry")
        return SceneBackendOutput(
            backend=output.backend,
            dense_views=output.dense_views,
            point_cloud=cloud,
            mesh=mesh,
            scene=output.scene,
            native_assets=output.native_assets,
            metadata={**output.metadata, "style_backend": self.name},
        )


def _transfer(source: np.ndarray, reference: np.ndarray, strength: float) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    src_mean = source.mean(axis=0)
    ref_mean = reference.mean(axis=0)
    src_center = source - src_mean
    ref_center = reference - ref_mean
    src_cov = np.cov(src_center, rowvar=False) + np.eye(3) * 1.0e-6
    ref_cov = np.cov(ref_center, rowvar=False) + np.eye(3) * 1.0e-6
    src_vals, src_vecs = np.linalg.eigh(src_cov)
    ref_vals, ref_vecs = np.linalg.eigh(ref_cov)
    whiten = (
        src_vecs
        @ np.diag(1.0 / np.sqrt(np.maximum(src_vals, 1.0e-8)))
        @ src_vecs.T
    )
    color = (
        ref_vecs @ np.diag(np.sqrt(np.maximum(ref_vals, 1.0e-8))) @ ref_vecs.T
    )
    transferred = src_center @ whiten.T @ color.T + ref_mean
    return np.clip((1.0 - strength) * source + strength * transferred, 0.0, 1.0)


__all__ = ["StatisticalColorStyle"]
