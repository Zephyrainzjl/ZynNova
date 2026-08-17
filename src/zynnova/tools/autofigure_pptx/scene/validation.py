from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..exceptions import SceneValidationError
from .schema import ConnectorElement, FigureScene, TextElement


@dataclass(slots=True)
class SceneIssue:
    code: str
    message: str
    element_id: str | None = None
    severity: str = "warning"


@dataclass(slots=True)
class SceneValidationReport:
    valid: bool
    issues: tuple[SceneIssue, ...] = ()

    @property
    def errors(self) -> tuple[SceneIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "error")

    @property
    def warnings(self) -> tuple[SceneIssue, ...]:
        return tuple(item for item in self.issues if item.severity != "error")

    def require_valid(self) -> "SceneValidationReport":
        if not self.valid:
            raise SceneValidationError("; ".join(item.message for item in self.errors))
        return self


def _overlap_area(a, b) -> float:
    dx = max(0.0, min(a.right, b.right) - max(a.x, b.x))
    dy = max(0.0, min(a.bottom, b.bottom) - max(a.y, b.y))
    return dx * dy


def validate_scene(scene: FigureScene, *, strict: bool = False) -> SceneValidationReport:
    issues: list[SceneIssue] = []
    ids = [element.id for element in scene.elements]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    for item in duplicates:
        issues.append(SceneIssue("duplicate-id", f"duplicate element id {item!r}", item, "error"))

    known = set(ids)
    for element in scene.elements:
        bounds = element.bounds
        if bounds.x < 0 or bounds.y < 0 or bounds.right > scene.width or bounds.bottom > scene.height:
            issues.append(
                SceneIssue(
                    "out-of-bounds",
                    f"element {element.id!r} is outside the slide canvas",
                    element.id,
                    "error" if strict else "warning",
                )
            )
        if isinstance(element, ConnectorElement):
            if element.source_id not in known:
                issues.append(SceneIssue("missing-source", f"connector source {element.source_id!r} is missing", element.id, "error"))
            if element.target_id not in known:
                issues.append(SceneIssue("missing-target", f"connector target {element.target_id!r} is missing", element.id, "error"))
        if isinstance(element, TextElement):
            estimated_chars = max(1.0, element.bounds.width * element.bounds.height * 18.0)
            if len(element.text) > estimated_chars * 2.2:
                issues.append(SceneIssue("text-overflow-risk", f"text in {element.id!r} may overflow", element.id))

    non_connectors = [item for item in scene.elements if not isinstance(item, ConnectorElement)]
    for index, left in enumerate(non_connectors):
        for right in non_connectors[index + 1 :]:
            if left.group_id and left.group_id == right.group_id:
                continue
            area = _overlap_area(left.bounds, right.bounds)
            smaller = min(left.bounds.width * left.bounds.height, right.bounds.width * right.bounds.height)
            if smaller > 0 and area / smaller > 0.35:
                issues.append(SceneIssue("large-overlap", f"{left.id!r} and {right.id!r} overlap substantially"))

    valid = not any(item.severity == "error" for item in issues)
    return SceneValidationReport(valid=valid, issues=tuple(issues))
