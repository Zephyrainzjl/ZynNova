"""Image-to-object backend registry and audited source catalogue."""

from __future__ import annotations

from ..core import BackendDescriptor, BackendRegistry
from .backends import (
    ExternalObjectBackend,
    ObjectBackend,
    Pixal3DBackend,
    SilhouetteExtrusionBackend,
    Trellis2Backend,
)

OBJECT_BACKENDS: BackendRegistry[ObjectBackend] = BackendRegistry("image-to-object")
OBJECT_BACKENDS.register(
    BackendDescriptor(
        name="pixal3d",
        task="image-to-object",
        factory=Pixal3DBackend,
        summary="Pixel-aligned high-resolution image-to-3D with PBR material output.",
        license_id="MIT code + dependency terms",
        source="https://github.com/TencentARC/Pixal3D",
        default_rank=5,
        extras=("external isolated GPU environment",),
    )
)
OBJECT_BACKENDS.register(
    BackendDescriptor(
        name="trellis2",
        task="image-to-object",
        factory=Trellis2Backend,
        summary="4B O-Voxel image-to-3D generation with complex topology and PBR materials.",
        license_id="MIT code/model + dependency terms",
        source="https://github.com/microsoft/TRELLIS.2",
        default_rank=10,
        extras=("external isolated GPU environment",),
    )
)
OBJECT_BACKENDS.register(
    BackendDescriptor(
        name="silhouette-extrusion-baseline",
        task="image-to-object",
        factory=SilhouetteExtrusionBackend,
        summary="Deterministic CPU relief baseline used for smoke tests and fallback only.",
        license_id="MIT (ZynNova implementation)",
        default_rank=100,
        extras=("zynnova-object",),
    )
)

OBJECT_BACKENDS.register(
    BackendDescriptor(
        name="external-object-contract",
        task="image-to-object",
        factory=ExternalObjectBackend,
        summary="Versioned request/output contract for another audited image-to-3D repository.",
        license_id="user-supplied",
        default_rank=90,
    )
)

PUBLIC_OBJECT_SOURCES = (
    {
        "name": "Pixal3D",
        "source": "https://github.com/TencentARC/Pixal3D",
        "venue": "SIGGRAPH 2026",
        "role": "preferred high-fidelity pixel-aligned object generation",
    },
    {
        "name": "TRELLIS.2",
        "source": "https://github.com/microsoft/TRELLIS.2",
        "venue": "technical report / open implementation",
        "role": "high-resolution O-Voxel and PBR object generation",
    },
    {
        "name": "SPAR3D",
        "source": "https://github.com/Stability-AI/stable-point-aware-3d",
        "venue": "CVPR 2025",
        "role": "visible-surface regression plus generative hidden-surface completion",
    },
    {
        "name": "Stable Fast 3D",
        "source": "https://github.com/Stability-AI/stable-fast-3d",
        "venue": "CVPR 2025",
        "role": "explicit mesh, UV and material-aware single-image reconstruction",
    },
)

__all__ = ["OBJECT_BACKENDS", "PUBLIC_OBJECT_SOURCES"]
