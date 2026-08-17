"""Scene style-transfer interfaces."""

from .base import SceneStyleBackend
from .external import ExternalSceneStyle
from .instruct_gs2gs import InstructGS2GSStyle
from .registry import PUBLIC_STYLE_SOURCES, STYLE_BACKENDS, register_external_style
from .statistical import StatisticalColorStyle

__all__ = [
    "ExternalSceneStyle",
    "InstructGS2GSStyle",
    "PUBLIC_STYLE_SOURCES",
    "STYLE_BACKENDS",
    "SceneStyleBackend",
    "StatisticalColorStyle",
    "register_external_style",
]
