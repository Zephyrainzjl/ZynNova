"""Dense view predictions and normalized scene backend outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ..geometry import Camera, PointCloud, SceneBundle, TriangleMesh


@dataclass(frozen=True, slots=True)
class DenseView:
    points_world: np.ndarray
    image_rgb: np.ndarray
    confidence: np.ndarray | None = None
    mask: np.ndarray | None = None
    camera: Camera | None = None
    name: str = "view"

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world, dtype=np.float64)
        image = np.asarray(self.image_rgb)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points_world must have shape (H,W,3)")
        if image.shape != points.shape:
            raise ValueError("image_rgb must have the same (H,W,3) shape as points_world")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_world contains non-finite values")
        if image.dtype.kind in {"u", "i"}:
            colors = image.astype(np.float64) / 255.0
        else:
            colors = image.astype(np.float64)
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
        confidence = None
        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=np.float64).squeeze()
            if confidence.shape != points.shape[:2]:
                raise ValueError("confidence must have shape (H,W)")
        mask = None
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool).squeeze()
            if mask.shape != points.shape[:2]:
                raise ValueError("mask must have shape (H,W)")
        object.__setattr__(self, "points_world", np.ascontiguousarray(points))
        object.__setattr__(self, "image_rgb", np.ascontiguousarray(colors))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True, slots=True)
class SceneBackendOutput:
    backend: str
    dense_views: tuple[DenseView, ...] = ()
    point_cloud: PointCloud | None = None
    mesh: TriangleMesh | None = None
    scene: SceneBundle | None = None
    native_assets: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.dense_views
            and self.point_cloud is None
            and self.mesh is None
            and self.scene is None
            and not self.native_assets
        ):
            raise ValueError("scene backend returned no geometry or native assets")
        object.__setattr__(self, "dense_views", tuple(self.dense_views))
        object.__setattr__(
            self,
            "native_assets",
            {str(key): Path(value) for key, value in self.native_assets.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


__all__ = ["DenseView", "SceneBackendOutput"]
