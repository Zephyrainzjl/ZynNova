from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from ..scene.schema import (
    ChartElement,
    ChartKind,
    ConnectorElement,
    FigureScene,
    ImageElement,
    ShapeElement,
    ShapeKind,
    TextElement,
)


_THEME = {
    "bg1": "#FFFFFF",
    "bg2": "#F2F2F2",
    "tx1": "#1F1F1F",
    "tx2": "#595959",
    "accent1": "#4472C4",
    "accent2": "#ED7D31",
    "accent3": "#A5A5A5",
    "accent4": "#FFC000",
    "accent5": "#5B9BD5",
    "accent6": "#70AD47",
}


def _color(value: str) -> str:
    return _THEME.get(value.lower(), value)


def _px(value: float, scale: float) -> str:
    return f"{value * scale:.3f}"


def _opacity(transparency: float) -> str:
    return f"{1.0 - transparency:.4f}"


def _shape_node(parent: Element, element: ShapeElement, scale: float) -> Element:
    b = element.bounds
    common = {
        "id": element.id,
        "fill": _color(element.style.fill.value),
        "fill-opacity": _opacity(element.style.fill.transparency),
        "stroke": _color(element.style.line.value),
        "stroke-opacity": _opacity(element.style.line.transparency),
        "stroke-width": str(element.style.line_width_pt * 1.3333),
    }
    if element.style.dash != "solid":
        common["stroke-dasharray"] = "8 5" if element.style.dash == "dash" else "2 4"
    x, y, w, h = (_px(b.x, scale), _px(b.y, scale), _px(b.width, scale), _px(b.height, scale))
    if element.shape in {ShapeKind.RECTANGLE, ShapeKind.ROUND_RECTANGLE, ShapeKind.PROCESS, ShapeKind.DOCUMENT}:
        if element.shape == ShapeKind.ROUND_RECTANGLE:
            common["rx"] = _px(min(element.style.radius, b.height / 3.0), scale)
        node = SubElement(parent, "rect", {**common, "x": x, "y": y, "width": w, "height": h})
    elif element.shape == ShapeKind.ELLIPSE:
        node = SubElement(parent, "ellipse", {**common, "cx": _px(b.x + b.width / 2, scale), "cy": _px(b.y + b.height / 2, scale), "rx": _px(b.width / 2, scale), "ry": _px(b.height / 2, scale)})
    elif element.shape in {ShapeKind.DIAMOND, ShapeKind.DECISION}:
        points = [(b.x + b.width / 2, b.y), (b.right, b.y + b.height / 2), (b.x + b.width / 2, b.bottom), (b.x, b.y + b.height / 2)]
        node = SubElement(parent, "polygon", {**common, "points": " ".join(f"{_px(px, scale)},{_px(py, scale)}" for px, py in points)})
    elif element.shape == ShapeKind.TRIANGLE:
        points = [(b.x + b.width / 2, b.y), (b.right, b.bottom), (b.x, b.bottom)]
        node = SubElement(parent, "polygon", {**common, "points": " ".join(f"{_px(px, scale)},{_px(py, scale)}" for px, py in points)})
    elif element.shape == ShapeKind.CYLINDER:
        node = SubElement(parent, "g", {"id": element.id})
        SubElement(node, "rect", {**common, "x": x, "y": _px(b.y + b.height * 0.12, scale), "width": w, "height": _px(b.height * 0.76, scale)})
        SubElement(node, "ellipse", {**common, "cx": _px(b.x + b.width / 2, scale), "cy": _px(b.y + b.height * 0.12, scale), "rx": _px(b.width / 2, scale), "ry": _px(b.height * 0.12, scale)})
        SubElement(node, "ellipse", {**common, "cx": _px(b.x + b.width / 2, scale), "cy": _px(b.bottom - b.height * 0.12, scale), "rx": _px(b.width / 2, scale), "ry": _px(b.height * 0.12, scale)})
    else:
        node = SubElement(parent, "rect", {**common, "x": x, "y": y, "width": w, "height": h, "rx": _px(0.10, scale)})
    if element.text:
        _text(parent, element.id + "-text", element.text, b, element.text_style, scale)
    return node


def _text(parent: Element, element_id: str, text: str, bounds, style, scale: float) -> Element:
    anchor = {"left": "start", "center": "middle", "right": "end"}.get(style.align, "middle")
    x = bounds.x + (0.0 if anchor == "start" else bounds.width if anchor == "end" else bounds.width / 2.0)
    y = bounds.y + bounds.height / 2.0
    attrs = {
        "id": element_id,
        "x": _px(x, scale),
        "y": _px(y, scale),
        "text-anchor": anchor,
        "dominant-baseline": "middle",
        "font-size": str(style.font_size_pt or 18.0),
        "font-weight": "700" if style.bold else "400",
        "font-style": "italic" if style.italic else "normal",
        "fill": _color(style.color.value),
    }
    if style.font_name:
        attrs["font-family"] = style.font_name
    node = SubElement(parent, "text", attrs)
    max_chars = max(6, int(bounds.width * 9.5))
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    if not lines:
        lines = [text]
    line_height = (style.font_size_pt or 18.0) * 1.15
    offset = -(len(lines) - 1) * line_height / 2.0
    for index, line in enumerate(lines):
        span = SubElement(node, "tspan", {"x": attrs["x"], "dy": str(offset if index == 0 else line_height)})
        span.text = line
        offset = line_height
    return node


def _connector(parent: Element, scene: FigureScene, element: ConnectorElement, scale: float) -> None:
    source = scene.get(element.source_id).bounds
    target = scene.get(element.target_id).bounds
    sx, sy = source.center
    tx, ty = target.center
    if abs(tx - sx) >= abs(ty - sy):
        start = (source.right if tx >= sx else source.x, sy)
        end = (target.x if tx >= sx else target.right, ty)
    else:
        start = (sx, source.bottom if ty >= sy else source.y)
        end = (tx, target.y if ty >= sy else target.bottom)
    attrs = {
        "id": element.id,
        "x1": _px(start[0], scale),
        "y1": _px(start[1], scale),
        "x2": _px(end[0], scale),
        "y2": _px(end[1], scale),
        "stroke": _color(element.line.value),
        "stroke-width": str(element.line_width_pt * 1.3333),
        "fill": "none",
    }
    if element.end_arrow != "none":
        attrs["marker-end"] = "url(#arrow-end)"
    SubElement(parent, "line", attrs)


def _chart(parent: Element, element: ChartElement, scale: float) -> None:
    # SVG renderer provides a compact preview. The PPTX renderer creates the
    # actual editable Office chart.
    b = element.bounds
    SubElement(parent, "rect", {"x": _px(b.x, scale), "y": _px(b.y, scale), "width": _px(b.width, scale), "height": _px(b.height, scale), "fill": "#FFFFFF", "stroke": "#A0A0A0"})
    if not element.series:
        return
    all_values = [value for series in element.series for value in series.values]
    if not all_values:
        return
    minimum, maximum = min(all_values), max(all_values)
    span = maximum - minimum or 1.0
    colors = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"]
    for index, series in enumerate(element.series):
        points = []
        count = max(1, len(series.values) - 1)
        for i, value in enumerate(series.values):
            px = b.x + 0.08 * b.width + 0.84 * b.width * i / count
            py = b.y + 0.88 * b.height - 0.72 * b.height * (value - minimum) / span
            points.append(f"{_px(px, scale)},{_px(py, scale)}")
        SubElement(parent, "polyline", {"points": " ".join(points), "fill": "none", "stroke": _color((series.color.value if series.color else colors[index % len(colors)])), "stroke-width": "2.0"})
    if element.title:
        _text(parent, element.id + "-title", element.title, type("B", (), {"x": b.x, "y": b.y, "width": b.width, "height": 0.35})(), type("S", (), {"align": "center", "font_size_pt": 14.0, "bold": True, "italic": False, "color": type("C", (), {"value": "tx1"})(), "font_name": None})(), scale)


class SVGRenderer:
    def __init__(self, *, dpi: int = 96):
        self.dpi = dpi

    def render(self, scene: FigureScene, output_path: str | Path) -> Path:
        scale = self.dpi
        root = Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(scene.width * scale),
            "height": str(scene.height * scale),
            "viewBox": f"0 0 {scene.width * scale} {scene.height * scale}",
            "role": "img",
            "aria-label": scene.title,
        })
        defs = SubElement(root, "defs")
        marker = SubElement(defs, "marker", {"id": "arrow-end", "viewBox": "0 0 10 10", "refX": "8", "refY": "5", "markerWidth": "7", "markerHeight": "7", "orient": "auto-start-reverse"})
        SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": _color("tx1")})
        SubElement(root, "rect", {"x": "0", "y": "0", "width": str(scene.width * scale), "height": str(scene.height * scale), "fill": _color(scene.background.value)})
        for element in scene.sorted_elements():
            if isinstance(element, ShapeElement):
                _shape_node(root, element, scale)
            elif isinstance(element, TextElement):
                _text(root, element.id, element.text, element.bounds, element.style, scale)
            elif isinstance(element, ConnectorElement):
                _connector(root, scene, element, scale)
            elif isinstance(element, ChartElement):
                _chart(root, element, scale)
            elif isinstance(element, ImageElement):
                href = element.data_uri
                if href is None and element.path is not None:
                    suffix = Path(element.path).suffix.lower().lstrip(".")
                    mime = "image/svg+xml" if suffix == "svg" else f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
                    href = f"data:{mime};base64," + base64.b64encode(Path(element.path).read_bytes()).decode("ascii")
                if href:
                    b = element.bounds
                    SubElement(root, "image", {"id": element.id, "href": href, "x": _px(b.x, scale), "y": _px(b.y, scale), "width": _px(b.width, scale), "height": _px(b.height, scale), "preserveAspectRatio": "xMidYMid meet"})
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(tostring(root, encoding="utf-8", xml_declaration=True))
        return output
