from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class PowerPointTarget(str, Enum):
    """PowerPoint compatibility target.

    The implementation deliberately emits ECMA-376 DrawingML and standard
    chart parts supported by perpetual PowerPoint 2019 and newer.
    """

    PPT2019 = "ppt2019"
    PPT2021 = "ppt2021"
    MICROSOFT_365 = "microsoft365"


class PPTXMode(str, Enum):
    """How aggressively to preserve editability."""

    NATIVE = "native"
    HYBRID = "hybrid"
    SVG = "svg"


class UpstreamMode(str, Enum):
    """Relationship with the MIT-licensed upstream ``autofigure`` package."""

    AUTO = "auto"
    REQUIRED = "required"
    DISABLED = "disabled"


@dataclass(slots=True)
class PPTXRenderConfig:
    target: PowerPointTarget = PowerPointTarget.PPT2019
    mode: PPTXMode = PPTXMode.HYBRID
    template_path: Path | None = None
    slide_layout_index: int | None = None
    group_native_diagram: bool = True
    inherit_theme_fonts: bool = True
    inherit_theme_colors: bool = True
    default_font_size_pt: float = 18.0
    title_font_size_pt: float = 26.0
    minimum_font_size_pt: float = 8.0
    connector_attachment: bool = True
    include_alt_text: bool = True
    svg_fallback_dpi: int = 192
    strict: bool = True

    def __post_init__(self) -> None:
        if self.svg_fallback_dpi < 72:
            raise ValueError("svg_fallback_dpi must be at least 72")
        if self.minimum_font_size_pt <= 0:
            raise ValueError("minimum_font_size_pt must be positive")
        if self.template_path is not None:
            self.template_path = Path(self.template_path).expanduser().resolve()


@dataclass(slots=True)
class AgentConfig:
    """Configuration for the compatible AutoFigure + native-PPTX agent."""

    output_dir: Path = Path("autofigure_outputs")
    upstream_mode: UpstreamMode = UpstreamMode.AUTO
    generation_api_key: str | None = None
    generation_provider: str = "openrouter"
    generation_model: str | None = None
    evaluation_api_key: str | None = None
    evaluation_provider: str | None = None
    evaluation_model: str | None = None
    quality_threshold: float = 8.0
    max_iterations: int = 4
    topic: str = "paper"
    pptx: PPTXRenderConfig = field(default_factory=PPTXRenderConfig)
    upstream_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if not (0.0 <= self.quality_threshold <= 10.0):
            raise ValueError("quality_threshold must lie in [0, 10]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AgentConfig":
        data = dict(values)
        pptx = data.get("pptx")
        if isinstance(pptx, Mapping):
            data["pptx"] = PPTXRenderConfig(**dict(pptx))
        return cls(**data)
