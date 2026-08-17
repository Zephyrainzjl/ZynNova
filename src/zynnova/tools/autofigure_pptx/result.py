from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EditabilityReport:
    native_shape_count: int = 0
    native_text_count: int = 0
    native_connector_count: int = 0
    native_chart_count: int = 0
    picture_count: int = 0
    grouped_shape_count: int = 0
    explicit_font_count: int = 0
    text_overflow_risk_count: int = 0
    detached_connector_count: int = 0
    out_of_bounds_count: int = 0
    compatibility_target: str = "ppt2019"
    package_valid: bool = False
    editability_score: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class GenerationResult:
    success: bool = False
    svg_path: Path | None = None
    mxgraph_path: Path | None = None
    pptx_path: Path | None = None
    preview_path: Path | None = None
    enhanced_path: Path | None = None
    enhanced_paths: tuple[Path, ...] = ()
    scene_path: Path | None = None
    methodology_text: str | None = None
    final_score: float = 0.0
    iterations: int = 0
    slide_index: int | None = None
    editability: EditabilityReport | None = None
    upstream_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def require_success(self) -> "GenerationResult":
        if not self.success:
            raise RuntimeError(self.error or "figure generation failed")
        return self
