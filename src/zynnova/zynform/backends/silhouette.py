"""Deterministic alpha/silhouette extrusion baseline for tests and CPU fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core import Availability
from ...core.backend import module_availability
from ...geometry import (
    clean_triangle_mesh,
    extract_material_surfaces,
    select_volume_regions,
    voxel_to_tetrahedra,
)
from ..schema import ObjectConfig, ObjectRequest
from ..types import ObjectBackendOutput
from .base import ObjectBackend


class SilhouetteExtrusionBackend(ObjectBackend):
    """Build a watertight relief volume; explicitly not a learned high-fidelity model."""

    name = "silhouette-extrusion-baseline"

    def __init__(
        self,
        *,
        maximum_image_size: int = 128,
        depth_ratio: float = 0.35,
        foreground_threshold: float | None = None,
        **_: object,
    ) -> None:
        if maximum_image_size < 8:
            raise ValueError("maximum_image_size must be at least eight")
        if depth_ratio <= 0.0:
            raise ValueError("depth_ratio must be positive")
        self.maximum_image_size = int(maximum_image_size)
        self.depth_ratio = float(depth_ratio)
        self.foreground_threshold = foreground_threshold

    def availability(self) -> Availability:
        return module_availability("PIL")

    def run(
        self,
        request: ObjectRequest,
        config: ObjectConfig,
        work_directory: Path,
    ) -> ObjectBackendOutput:
        from PIL import Image

        image = Image.open(request.image).convert("RGBA")
        scale = min(1.0, self.maximum_image_size / max(image.size))
        size = tuple(max(2, int(round(value * scale))) for value in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
        rgba = np.asarray(image, dtype=np.uint8)
        if request.foreground_mask is not None:
            mask_image = Image.open(request.foreground_mask).convert("L").resize(
                size, Image.Resampling.LANCZOS
            )
            mask = np.asarray(mask_image) >= 128
        elif np.ptp(rgba[..., 3]) > 8:
            mask = rgba[..., 3] >= 32
        else:
            mask = _background_difference_mask(rgba[..., :3], self.foreground_threshold)
        if not np.any(mask):
            raise ValueError("no foreground could be extracted from the image")
        distance = _distance_inside(mask)
        depth = max(4, int(round(max(size) * self.depth_ratio)))
        half = np.maximum(
            1,
            np.rint((depth / 2 - 1) * distance / max(float(distance.max()), 1.0)).astype(int),
        )
        labels = np.zeros((depth, mask.shape[0], mask.shape[1]), dtype=np.int32)
        center = 0.5 * (depth - 1)
        for z in range(depth):
            labels[z] = mask & (np.abs(z - center) <= half)
        result = voxel_to_tetrahedra(
            labels, spacing=1.0, region_names={0: "OUTSIDE", 1: "OBJECT"}
        )
        solid = select_volume_regions(result.volume_mesh, {1})
        boundary, _ = extract_material_surfaces(solid)
        surface, _ = clean_triangle_mesh(boundary, weld_tolerance=0.0)
        work_directory.mkdir(parents=True, exist_ok=True)
        return ObjectBackendOutput(
            backend=self.name,
            mesh=surface,
            metadata={
                "baseline": True,
                "voxel_shape": labels.shape,
                "warning": "Silhouette relief is a deterministic fallback, not photorealistic 3D inference.",
            },
        )


def _background_difference_mask(rgb: np.ndarray, threshold: float | None) -> np.ndarray:
    corners = np.concatenate(
        (
            rgb[:4, :4].reshape(-1, 3),
            rgb[:4, -4:].reshape(-1, 3),
            rgb[-4:, :4].reshape(-1, 3),
            rgb[-4:, -4:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(float)
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb.astype(float) - background, axis=-1)
    if threshold is not None:
        level = float(threshold)
    else:
        # A single high percentile fails when the foreground occupies more than
        # roughly one third of the image (the percentile can equal the foreground
        # plateau). Bound it by half the observed contrast while retaining a noise
        # floor for compressed images.
        robust = float(np.percentile(distance, 65.0))
        contrast = float(np.max(distance))
        level = max(12.0, min(robust, 0.5 * contrast))
    return distance > level


def _distance_inside(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(mask)
    except ImportError:
        # Dependency-light Manhattan erosion depth.
        distance = mask.astype(np.int32)
        current = mask.copy()
        for value in range(2, max(mask.shape) + 1):
            padded = np.pad(current, 1, mode="constant")
            current = (
                padded[1:-1, 1:-1]
                & padded[:-2, 1:-1]
                & padded[2:, 1:-1]
                & padded[1:-1, :-2]
                & padded[1:-1, 2:]
            )
            if not np.any(current):
                break
            distance[current] = value
        return distance.astype(float)


__all__ = ["SilhouetteExtrusionBackend"]
