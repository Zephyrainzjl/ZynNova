"""Editable scientific illustrations for PowerPoint 2019 and newer.

The package preserves the upstream AutoFigure API through an optional bridge
and adds a semantic scene graph plus native DrawingML/Chart rendering.
"""

from .agent import AutoFigureAgent, AutoFigurePPTXAgent, Config
from .config import AgentConfig, PPTXMode, PPTXRenderConfig, PowerPointTarget, UpstreamMode
from .extractor import PaperMethodExtractor
from .importers import SVGSceneImporter
from .judge import JudgeResult, SceneJudge
from .pptx import NativePPTXRenderer, PPTXValidator
from .renderers import MxGraphRenderer, PPTXRenderer, PreviewRenderer, SVGRenderer
from .result import EditabilityReport, GenerationResult
from . import scene as _scene
from .scene import *
from .upstream import UpstreamAutoFigureBridge, UpstreamStatus

__all__ = [
    "AgentConfig",
    "AutoFigureAgent",
    "AutoFigurePPTXAgent",
    "Config",
    "EditabilityReport",
    "GenerationResult",
    "JudgeResult",
    "MxGraphRenderer",
    "NativePPTXRenderer",
    "PPTXMode",
    "PPTXRenderConfig",
    "PPTXRenderer",
    "PPTXValidator",
    "PaperMethodExtractor",
    "PowerPointTarget",
    "PreviewRenderer",
    "SVGRenderer",
    "SVGSceneImporter",
    "SceneJudge",
    "UpstreamAutoFigureBridge",
    "UpstreamMode",
    "UpstreamStatus",
    *_scene.__all__,
]
