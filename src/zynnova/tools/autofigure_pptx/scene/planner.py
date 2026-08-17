from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .layout import AutoLayoutEngine
from .schema import (
    Bounds,
    ColorRef,
    ConnectorElement,
    ConnectorKind,
    FigureScene,
    ShapeElement,
    ShapeKind,
    ShapeStyle,
    TextElement,
    TextStyle,
)
from .serialization import scene_from_dict


class ScenePlanner(Protocol):
    def plan(self, description: str, *, topic: str = "paper", title: str | None = None) -> FigureScene: ...


def _clean_stage(value: str) -> str:
    value = re.sub(r"^[\s\-–—•*\d.()]+", "", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;:")


def _split_stages(description: str) -> list[str]:
    arrows = re.split(r"\s*(?:→|➜|=>|->|⟶)\s*", description)
    if len(arrows) > 1:
        stages = [_clean_stage(item) for item in arrows]
    else:
        lines = [item for item in re.split(r"[\n;]+", description) if item.strip()]
        if len(lines) <= 1:
            lines = re.split(r"(?<=[.!?。！？])\s+", description)
        stages = [_clean_stage(item) for item in lines]
    stages = [item for item in stages if item]
    if len(stages) > 8:
        stages = stages[:8]
    return stages or ["Input", "Scientific processing", "Result"]


@dataclass(slots=True)
class RuleBasedPlanner:
    """Offline planner that always produces a valid fully editable pipeline."""

    width: float = 13.333333
    height: float = 7.5

    def plan(self, description: str, *, topic: str = "paper", title: str | None = None) -> FigureScene:
        stages = _split_stages(description)
        scene = FigureScene(
            title=title or "Scientific workflow",
            width=self.width,
            height=self.height,
            description=description,
            metadata={"planner": "rule-based", "topic": topic},
        )
        scene.add(
            TextElement(
                id="figure-title",
                bounds=Bounds(0.65, 0.22, self.width - 1.3, 0.52),
                z_index=20,
                text=title or "Scientific workflow",
                style=TextStyle(font_size_pt=27.0, font_role="major", bold=True, align="left"),
                alt_text="Figure title",
            )
        )
        palette = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"]
        node_ids: list[str] = []
        for index, stage in enumerate(stages):
            node_id = f"stage-{index + 1}"
            node_ids.append(node_id)
            color = palette[index % len(palette)]
            lower = stage.lower()
            shape = ShapeKind.ROUND_RECTANGLE
            if any(token in lower for token in ("decision", "select", "判断", "筛选")):
                shape = ShapeKind.DIAMOND
            elif any(token in lower for token in ("database", "dataset", "data", "数据库", "数据")):
                shape = ShapeKind.CYLINDER
            scene.add(
                ShapeElement(
                    id=node_id,
                    bounds=Bounds(0.0, 0.0, 2.0, 1.0),
                    z_index=10,
                    group_id="main-diagram",
                    shape=shape,
                    text=stage,
                    style=ShapeStyle(
                        fill=ColorRef(color, 0.10),
                        line=ColorRef(color),
                        line_width_pt=1.7,
                    ),
                    text_style=TextStyle(font_size_pt=17.0, bold=True),
                    alt_text=f"Workflow stage: {stage}",
                )
            )
        AutoLayoutEngine().pipeline(scene, node_ids)
        for index in range(len(node_ids) - 1):
            source = scene.get(node_ids[index])
            target = scene.get(node_ids[index + 1])
            x1, y1 = source.bounds.center
            x2, y2 = target.bounds.center
            scene.add(
                ConnectorElement(
                    id=f"connector-{index + 1}",
                    bounds=Bounds(min(x1, x2), min(y1, y2), max(abs(x2 - x1), 0.01), max(abs(y2 - y1), 0.01)),
                    z_index=5,
                    group_id="main-diagram",
                    source_id=source.id,
                    target_id=target.id,
                    connector=ConnectorKind.ELBOW if abs(y2 - y1) > 0.4 else ConnectorKind.STRAIGHT,
                    line=ColorRef("tx1", 0.15),
                    line_width_pt=1.5,
                    end_arrow="triangle",
                    alt_text=f"Flow from {source.id} to {target.id}",
                )
            )
        return scene


@dataclass(slots=True)
class CallableScenePlanner:
    """Adapter for an LLM or application callback returning scene JSON."""

    callback: Callable[[str, str, str | None], dict[str, Any] | FigureScene]

    def plan(self, description: str, *, topic: str = "paper", title: str | None = None) -> FigureScene:
        result = self.callback(description, topic, title)
        if isinstance(result, FigureScene):
            return result
        return scene_from_dict(result)


@dataclass(slots=True)
class OpenAICompatibleScenePlanner:
    """Small optional OpenAI-compatible JSON planner.

    This client is intentionally provider-neutral and uses only the Python
    standard library.  It targets endpoints exposing ``/chat/completions``.
    """

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 120.0

    def plan(self, description: str, *, topic: str = "paper", title: str | None = None) -> FigureScene:
        import urllib.request

        schema_instruction = (
            "Return JSON only. Create a scientific FigureScene with width 13.333333 and height 7.5. "
            "Use elements of kind text, shape, connector, chart, or image. Coordinates are inches. "
            "All text must remain as text elements or shape.text, never paths. Use theme colors accent1..accent6, tx1, bg1. "
            "Every connector must reference valid source_id and target_id. Keep all elements in bounds."
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": schema_instruction},
                {"role": "user", "content": f"Topic: {topic}\nTitle: {title or ''}\nDescription:\n{description}"},
            ],
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return scene_from_dict(json.loads(content))
