"""Requests and controls for high-fidelity image-to-scene workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from ..core.exceptions import ConfigurationError


class SceneMode(str, Enum):
    RECONSTRUCT = "reconstruct"
    GENERATE = "generate"


@dataclass(frozen=True, slots=True)
class SceneRequest:
    """Input media and reconstruction/generation intent."""

    images: tuple[Path, ...] = ()
    video: Path | None = None
    prompt: str | None = None
    mode: SceneMode = SceneMode.RECONSTRUCT
    backend: str = "auto"
    device: str = "auto"
    model_id: str | None = None
    prior_camera: Path | None = None
    prior_depth_directory: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images = tuple(Path(path) for path in self.images)
        video = None if self.video is None else Path(self.video)
        if self.mode == SceneMode.RECONSTRUCT and not images and video is None:
            raise ConfigurationError("scene reconstruction requires images or a video")
        if self.mode == SceneMode.GENERATE and not images and not self.prompt:
            raise ConfigurationError("scene generation requires a source image or text prompt")
        missing = [path for path in images if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        if video is not None and not video.is_file():
            raise FileNotFoundError(video)
        if self.prior_camera is not None and not Path(self.prior_camera).is_file():
            raise FileNotFoundError(self.prior_camera)
        if self.prior_depth_directory is not None and not Path(
            self.prior_depth_directory
        ).is_dir():
            raise FileNotFoundError(self.prior_depth_directory)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "video", video)
        object.__setattr__(
            self,
            "prior_camera",
            None if self.prior_camera is None else Path(self.prior_camera),
        )
        object.__setattr__(
            self,
            "prior_depth_directory",
            None
            if self.prior_depth_directory is None
            else Path(self.prior_depth_directory),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SceneConfig:
    output_directory: str = "zynnova_runs/zynvista"
    confidence_percentile: float = 10.0
    fusion_voxel_size_m: float = 0.005
    maximum_points: int = 2_000_000
    build_mesh: bool = True
    mesh_edge_factor: float = 3.5
    export_formats: tuple[str, ...] = ("ply", "obj", "glb")
    export_colmap: bool = True
    video_sample_fps: float = 2.0
    video_max_frames: int = 96
    extract_video_for_image_backends: bool = True
    build_world_hierarchy: bool = True
    world_chunk_size_m: float = 25.0
    world_lod_levels: int = 3
    world_overlap_m: float = 0.25
    world_up_axis: str = "Y"
    geometry_lock_during_style: bool = True
    require_metric_geometry: bool = True
    style_backend: str | None = None
    style_reference: Path | None = None
    style_prompt: str | None = None
    backend_options: Mapping[str, object] = field(default_factory=dict)
    style_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_percentile < 100.0:
            raise ConfigurationError("confidence_percentile must lie in [0,100)")
        if self.fusion_voxel_size_m <= 0.0:
            raise ConfigurationError("fusion_voxel_size_m must be positive")
        if self.maximum_points < 1:
            raise ConfigurationError("maximum_points must be positive")
        if self.mesh_edge_factor <= 0.0:
            raise ConfigurationError("mesh_edge_factor must be positive")
        if self.video_sample_fps <= 0.0:
            raise ConfigurationError("video_sample_fps must be positive")
        if self.video_max_frames < 1:
            raise ConfigurationError("video_max_frames must be positive")
        if self.world_chunk_size_m <= 0.0:
            raise ConfigurationError("world_chunk_size_m must be positive")
        if self.world_lod_levels < 1:
            raise ConfigurationError("world_lod_levels must be positive")
        if self.world_overlap_m < 0.0:
            raise ConfigurationError("world_overlap_m cannot be negative")
        up_axis = str(self.world_up_axis).strip().upper()
        if up_axis not in {"X", "Y", "Z"}:
            raise ConfigurationError("world_up_axis must be X, Y, or Z")
        object.__setattr__(self, "world_up_axis", up_axis)
        formats = tuple(
            dict.fromkeys(str(item).strip().lower().lstrip(".") for item in self.export_formats)
        )
        supported = {
            "ply", "obj", "stl", "npz", "glb", "gltf", "fbx", "usd", "usda",
            "usdc", "dae", "abc",
        }
        unsupported = sorted(set(formats) - supported)
        if unsupported:
            raise ConfigurationError(f"unsupported scene export formats: {unsupported}")
        if not formats:
            raise ConfigurationError("at least one scene export format is required")
        object.__setattr__(self, "export_formats", formats)
        if self.style_reference is not None:
            path = Path(self.style_reference)
            if not path.is_file():
                raise FileNotFoundError(path)
            object.__setattr__(self, "style_reference", path)
        object.__setattr__(self, "backend_options", dict(self.backend_options))
        object.__setattr__(self, "style_options", dict(self.style_options))


__all__ = ["SceneConfig", "SceneMode", "SceneRequest"]
