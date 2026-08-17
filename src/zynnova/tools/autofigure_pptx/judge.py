from __future__ import annotations

from dataclasses import dataclass

from .scene.schema import ConnectorElement, FigureScene, ImageElement, TextElement
from .scene.validation import SceneValidationReport, validate_scene


@dataclass(slots=True)
class JudgeResult:
    score: float
    design_quality: float
    clarity: float
    logical_flow: float
    editability: float
    compatibility: float
    feedback: tuple[str, ...]


class SceneJudge:
    """Deterministic structural judge used before optional visual LLM review."""

    def evaluate(self, scene: FigureScene) -> JudgeResult:
        report = validate_scene(scene, strict=False)
        errors = len(report.errors)
        warnings = len(report.warnings)
        connectors = [item for item in scene.elements if isinstance(item, ConnectorElement)]
        texts = [item for item in scene.elements if isinstance(item, TextElement)]
        pictures = [item for item in scene.elements if isinstance(item, ImageElement)]
        design = max(0.0, 10.0 - 1.8 * errors - 0.35 * warnings)
        clarity = min(10.0, 6.0 + min(len(texts), 5) * 0.5)
        flow = min(10.0, 6.0 + min(len(connectors), 5) * 0.6)
        editability = max(0.0, 10.0 - 1.2 * len(pictures))
        compatibility = 10.0
        score = 0.24 * design + 0.20 * clarity + 0.20 * flow + 0.26 * editability + 0.10 * compatibility
        feedback = tuple(item.message for item in report.issues)
        return JudgeResult(score, design, clarity, flow, editability, compatibility, feedback)
