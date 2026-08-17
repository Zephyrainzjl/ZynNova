"""Mode-specific scene backend registries and public provenance."""

from __future__ import annotations

from ..core import BackendDescriptor, BackendRegistry
from .backends import (
    ExternalSceneBackend,
    HYWorld2GenerationBackend,
    HYWorld2ReconstructionBackend,
    MapAnythingBackend,
    SceneBackend,
)

RECONSTRUCTION_BACKENDS: BackendRegistry[SceneBackend] = BackendRegistry(
    "scene-reconstruction"
)
GENERATION_BACKENDS: BackendRegistry[SceneBackend] = BackendRegistry("scene-generation")

RECONSTRUCTION_BACKENDS.register(
    BackendDescriptor(
        name="mapanything",
        task="scene-reconstruction",
        factory=MapAnythingBackend,
        summary=(
            "Universal feed-forward metric multi-view reconstruction with dense "
            "points and cameras."
        ),
        license_id="Apache-2.0 model option",
        source="https://github.com/facebookresearch/map-anything",
        default_rank=10,
        extras=("zynnova-scene",),
    )
)
RECONSTRUCTION_BACKENDS.register(
    BackendDescriptor(
        name="hy-world-2-reconstruct",
        task="scene-reconstruction",
        factory=HYWorld2ReconstructionBackend,
        summary="WorldMirror 2.0: point cloud, depth, normals, cameras and 3DGS.",
        license_id="Tencent Hunyuan Community License",
        source="https://github.com/Tencent-Hunyuan/HY-World-2.0",
        default_rank=20,
        extras=("external isolated environment",),
    )
)
GENERATION_BACKENDS.register(
    BackendDescriptor(
        name="hy-world-2-generate",
        task="scene-generation",
        factory=HYWorld2GenerationBackend,
        summary=(
            "HY-Pano + WorldNav + WorldStereo + 3DGS five-stage large-world "
            "generation."
        ),
        license_id="Tencent Hunyuan Community License",
        source="https://github.com/Tencent-Hunyuan/HY-World-2.0",
        default_rank=10,
        extras=("external isolated multi-GPU environment",),
    )
)

RECONSTRUCTION_BACKENDS.register(
    BackendDescriptor(
        name="external-scene-contract",
        task="scene-reconstruction",
        factory=ExternalSceneBackend,
        summary="Versioned request/output contract for another audited scene repository.",
        license_id="user-supplied",
        default_rank=90,
    )
)
GENERATION_BACKENDS.register(
    BackendDescriptor(
        name="external-scene-contract",
        task="scene-generation",
        factory=ExternalSceneBackend,
        summary="Versioned request/output contract for another audited world generator.",
        license_id="user-supplied",
        default_rank=90,
    )
)

PUBLIC_SCENE_SOURCES = (
    {
        "name": "MapAnything",
        "source": "https://github.com/facebookresearch/map-anything",
        "role": "metric reconstruction",
        "license": "Apache-2.0 code; Apache model option",
    },
    {
        "name": "HY-World 2.0",
        "source": "https://github.com/Tencent-Hunyuan/HY-World-2.0",
        "role": "reconstruction and generated large worlds",
        "license": "Tencent Hunyuan Community License",
    },
)

__all__ = [
    "GENERATION_BACKENDS",
    "PUBLIC_SCENE_SOURCES",
    "RECONSTRUCTION_BACKENDS",
]
