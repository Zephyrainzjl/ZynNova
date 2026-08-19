"""Style registry and audited public-source catalogue."""

from __future__ import annotations

from ...core import BackendDescriptor, BackendRegistry
from .base import SceneStyleBackend
from .external import ExternalSceneStyle
from .instruct_gs2gs import InstructGS2GSStyle
from .statistical import StatisticalColorStyle

STYLE_BACKENDS: BackendRegistry[SceneStyleBackend] = BackendRegistry("scene-style")
STYLE_BACKENDS.register(
    BackendDescriptor(
        name="statistical-color",
        task="scene-style",
        factory=StatisticalColorStyle,
        summary="Geometry-preserving covariance color transfer; fast local baseline.",
        license_id="MIT (ZynNova implementation)",
        default_rank=10,
        extras=("zynnova-scene",),
    )
)

STYLE_BACKENDS.register(
    BackendDescriptor(
        name="instruct-gs2gs",
        task="scene-style",
        factory=InstructGS2GSStyle,
        summary="Official text-guided 3D Gaussian scene editing through Nerfstudio.",
        license_id="MIT + model/dependency terms",
        source="https://github.com/cvachha/instruct-gs2gs",
        default_rank=20,
        extras=("isolated Nerfstudio/Instruct-GS2GS environment",),
    )
)

PUBLIC_STYLE_SOURCES = (
    {
        "name": "StyleGaussian",
        "task": "3D Gaussian scene stylization",
        "source": "https://github.com/Kunhao-Liu/StyleGaussian",
        "venue": "SIGGRAPH Asia 2024",
        "adapter": "ExternalSceneStyle",
    },
    {
        "name": "StylOS",
        "task": "single-forward stylization of 3D Gaussian scenes",
        "source": "https://github.com/HanzhouLiu/StylOS",
        "venue": "ICLR 2026",
        "adapter": "ExternalSceneStyle",
    },
    {
        "name": "Instruct-GS2GS",
        "task": "text-guided 3D Gaussian scene editing",
        "source": "https://github.com/cvachha/instruct-gs2gs",
        "venue": "official Nerfstudio extension (2024)",
        "adapter": "InstructGS2GSStyle",
    },
    {
        "name": "A3GS",
        "task": "zero-shot arbitrary artistic style transfer for 3D Gaussian scenes",
        "source": "paper/external implementation",
        "venue": "ICCV 2025",
        "adapter": "ExternalSceneStyle",
    },
    {
        "name": "SGSST",
        "task": "scalable Gaussian style transfer for ultra-high-resolution scenes",
        "source": "paper/external implementation",
        "venue": "CVPR 2025",
        "adapter": "ExternalSceneStyle",
    },
)


def register_external_style(
    name: str,
    *,
    command: tuple[str, ...],
    output_files: dict[str, str],
    cwd: str | None = None,
    replace: bool = False,
) -> None:
    def factory(**kwargs: object) -> ExternalSceneStyle:
        return ExternalSceneStyle(
            name=name,
            command=command,
            output_files=output_files,
            cwd=cwd,
            **kwargs,
        )

    STYLE_BACKENDS.register(
        BackendDescriptor(
            name=name,
            task="scene-style",
            factory=factory,
            summary="User-registered isolated scene-style backend.",
            default_rank=100,
        ),
        replace=replace,
    )


__all__ = ["PUBLIC_STYLE_SOURCES", "STYLE_BACKENDS", "register_external_style"]
