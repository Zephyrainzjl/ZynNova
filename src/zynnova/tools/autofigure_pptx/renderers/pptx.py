from __future__ import annotations

from pathlib import Path

from ..config import PPTXRenderConfig
from ..pptx.native import NativePPTXRenderer
from ..scene.schema import FigureScene


class PPTXRenderer:
    """Public renderer facade kept separate from the low-level PPTX package."""

    def __init__(self, config: PPTXRenderConfig | None = None):
        self.native = NativePPTXRenderer(config)

    def render(self, scene: FigureScene, output_path: str | Path):
        return self.native.render(scene, output_path)
