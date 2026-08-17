"""Direct MapAnything adapter with metric dense-view normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core import Availability
from ...core.backend import module_availability
from ...geometry import Camera
from ..schema import SceneConfig, SceneRequest
from ..types import DenseView, SceneBackendOutput
from .base import SceneBackend


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _first(value: Any) -> np.ndarray:
    array = _numpy(value)
    return array[0] if array.ndim and array.shape[0] == 1 else array


class MapAnythingBackend(SceneBackend):
    """Run the official feed-forward metric reconstruction Python API lazily."""

    name = "mapanything"

    def __init__(
        self,
        *,
        model_id: str = "facebook/map-anything-apache",
        device: str = "auto",
        memory_efficient_inference: bool = True,
        minibatch_size: int | None = None,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        use_multiview_confidence: bool = False,
        **_: object,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.memory_efficient_inference = bool(memory_efficient_inference)
        self.minibatch_size = minibatch_size
        self.use_amp = bool(use_amp)
        self.amp_dtype = str(amp_dtype)
        self.apply_mask = bool(apply_mask)
        self.mask_edges = bool(mask_edges)
        self.use_multiview_confidence = bool(use_multiview_confidence)

    def availability(self) -> Availability:
        map_status = module_availability("mapanything")
        torch_status = module_availability("torch")
        if not map_status.available:
            return map_status
        if not torch_status.available:
            return torch_status
        return Availability(True, details={"model_id": self.model_id, "device": self.device})

    def run(
        self,
        request: SceneRequest,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        if request.video is not None:
            raise ValueError("MapAnything direct adapter expects image paths, not a video")
        if not request.images:
            raise ValueError("MapAnything requires at least one image")
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images

        device = self.device if self.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_id = request.model_id or self.model_id
        model = MapAnything.from_pretrained(model_id).to(device)
        model.eval()
        views = load_images([str(path) for path in request.images])
        predictions = model.infer(
            views,
            memory_efficient_inference=self.memory_efficient_inference,
            minibatch_size=self.minibatch_size,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            apply_mask=self.apply_mask,
            mask_edges=self.mask_edges,
            apply_confidence_mask=False,
            confidence_percentile=config.confidence_percentile,
            use_multiview_confidence=self.use_multiview_confidence,
        )
        dense: list[DenseView] = []
        for index, prediction in enumerate(predictions):
            points = _first(prediction["pts3d"])
            image = _first(prediction["img_no_norm"])
            confidence = _first(prediction["conf"]).squeeze()
            mask_value = prediction.get("mask")
            mask = None if mask_value is None else _first(mask_value).squeeze().astype(bool)
            intrinsics = _first(prediction["intrinsics"])
            pose = _first(prediction["camera_poses"])
            height, width = points.shape[:2]
            camera = Camera(
                pose=pose,
                intrinsics=intrinsics,
                width=width,
                height=height,
                name=(
                    request.images[index].name
                    if index < len(request.images)
                    else f"view_{index:04d}.png"
                ),
                metadata={"convention": "OpenCV_c2w"},
            )
            dense.append(
                DenseView(
                    points_world=points,
                    image_rgb=image,
                    confidence=confidence,
                    mask=mask,
                    camera=camera,
                    name=camera.name,
                )
            )
        work_directory.mkdir(parents=True, exist_ok=True)
        return SceneBackendOutput(
            backend=self.name,
            dense_views=tuple(dense),
            metadata={
                "model_id": model_id,
                "device": device,
                "coordinate_convention": "OpenCV camera-to-world",
                "input_count": len(request.images),
            },
        )


__all__ = ["MapAnythingBackend"]
