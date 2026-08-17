from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    Bounds,
    ChartElement,
    ChartKind,
    ChartSeries,
    ColorRef,
    ConnectorElement,
    ConnectorKind,
    FigureScene,
    ImageElement,
    ShapeElement,
    ShapeKind,
    ShapeStyle,
    TextElement,
    TextStyle,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def scene_to_dict(scene: FigureScene) -> dict[str, Any]:
    payload = _jsonable(asdict(scene))
    for element, source in zip(payload["elements"], scene.elements):
        element["kind"] = source.kind.value
    return payload


def scene_to_json(scene: FigureScene, path: str | Path | None = None, *, indent: int = 2) -> str:
    text = json.dumps(scene_to_dict(scene), indent=indent, ensure_ascii=False)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text


def _color(data: Mapping[str, Any] | str | None, default: str) -> ColorRef:
    if data is None:
        return ColorRef(default)
    if isinstance(data, str):
        return ColorRef(data)
    return ColorRef(**dict(data))


def _text_style(data: Mapping[str, Any] | None) -> TextStyle:
    values = dict(data or {})
    values["color"] = _color(values.get("color"), "tx1")
    return TextStyle(**values)


def _shape_style(data: Mapping[str, Any] | None) -> ShapeStyle:
    values = dict(data or {})
    values["fill"] = _color(values.get("fill"), "accent1")
    values["line"] = _color(values.get("line"), "accent1")
    return ShapeStyle(**values)


def scene_from_dict(data: Mapping[str, Any]) -> FigureScene:
    scene = FigureScene(
        title=str(data.get("title", "Scientific figure")),
        width=float(data.get("width", 13.333333)),
        height=float(data.get("height", 7.5)),
        description=data.get("description"),
        background=_color(data.get("background"), "bg1"),
        metadata=dict(data.get("metadata") or {}),
    )
    for raw in data.get("elements", []):
        item = dict(raw)
        kind = item.pop("kind")
        item["bounds"] = Bounds(**dict(item["bounds"]))
        if kind == "text":
            item["style"] = _text_style(item.get("style"))
            element = TextElement(**item)
        elif kind == "shape":
            item["shape"] = ShapeKind(item.get("shape", ShapeKind.ROUND_RECTANGLE))
            item["style"] = _shape_style(item.get("style"))
            item["text_style"] = _text_style(item.get("text_style"))
            element = ShapeElement(**item)
        elif kind == "connector":
            item["connector"] = ConnectorKind(item.get("connector", ConnectorKind.STRAIGHT))
            item["line"] = _color(item.get("line"), "accent1")
            element = ConnectorElement(**item)
        elif kind == "chart":
            item["chart"] = ChartKind(item.get("chart", ChartKind.LINE))
            item["series"] = [ChartSeries(**dict(series)) for series in item.get("series", [])]
            element = ChartElement(**item)
        elif kind == "image":
            if item.get("path") is not None:
                item["path"] = Path(item["path"])
            element = ImageElement(**item)
        else:
            raise ValueError(f"unknown scene element kind: {kind!r}")
        scene.elements.append(element)
    return scene


def scene_from_json(source: str | Path) -> FigureScene:
    path = Path(source)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(str(source))
    return scene_from_dict(payload)
