"""ZynVista: metric reconstruction, world generation, export and style transfer."""

from .export import export_scene_output, write_colmap_text
from .fusion import dense_view_to_mesh, dense_views_to_mesh, fuse_dense_views
from .pipeline import SceneResult, run_scene
from .quality import (
    GeometryFingerprint,
    SceneGeometryAudit,
    assert_geometry_preserved,
    audit_scene_geometry,
    geometry_fingerprint,
)
from .registry import GENERATION_BACKENDS, PUBLIC_SCENE_SOURCES, RECONSTRUCTION_BACKENDS
from .schema import SceneConfig, SceneMode, SceneRequest
from .styles import PUBLIC_STYLE_SOURCES, STYLE_BACKENDS, register_external_style
from .types import DenseView, SceneBackendOutput
from .video import extract_video_frames
from .world import WorldChunkRecord, WorldIndex, export_world_hierarchy

from .external import (
    CommandSceneEngine, GenerativeSceneRequest, PythonSceneEngine,
    SceneAssetBundle, SceneEngineProfile,
)
from .model_hub import download_scene_model, scene_workspace
from .studio import SceneStudio

__all__ = [
    "scene_workspace",
    "download_scene_model",
    "SceneStudio",
    "SceneEngineProfile",
    "SceneAssetBundle",
    "PythonSceneEngine",
    "GenerativeSceneRequest",
    "CommandSceneEngine",
    "DenseView",
    "GeometryFingerprint",
    "GENERATION_BACKENDS",
    "PUBLIC_SCENE_SOURCES",
    "PUBLIC_STYLE_SOURCES",
    "RECONSTRUCTION_BACKENDS",
    "STYLE_BACKENDS",
    "SceneBackendOutput",
    "SceneGeometryAudit",
    "SceneConfig",
    "SceneMode",
    "SceneRequest",
    "SceneResult",
    "WorldChunkRecord",
    "WorldIndex",
    "assert_geometry_preserved",
    "audit_scene_geometry",
    "dense_view_to_mesh",
    "dense_views_to_mesh",
    "export_scene_output",
    "export_world_hierarchy",
    "extract_video_frames",
    "fuse_dense_views",
    "geometry_fingerprint",
    "register_external_style",
    "run_scene",
    "write_colmap_text",
]
