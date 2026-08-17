from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..scene.schema import (
    Bounds,
    ColorRef,
    ConnectorElement,
    FigureScene,
    ImageElement,
    ShapeElement,
    ShapeKind,
    ShapeStyle,
    TextElement,
    TextStyle,
)


_LENGTH_RE = re.compile(r"^\s*([-+0-9.eE]+)")


def _number(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = _LENGTH_RE.match(value)
    return float(match.group(1)) if match else default


def _style(node: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = node.attrib.get("style", "")
    for item in raw.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            result[key.strip()] = value.strip()
    result.update({key: value for key, value in node.attrib.items() if key in {"fill", "stroke", "stroke-width", "font-size", "font-family", "font-weight", "text-anchor"}})
    return result


def _color(value: str | None, fallback: str) -> ColorRef:
    if not value or value in {"none", "transparent"}:
        return ColorRef(fallback, 1.0 if value in {"none", "transparent"} else 0.0)
    return ColorRef(value)


class SVGSceneImporter:
    """Import the editable subset of an SVG into the semantic scene graph.

    Rectangles, ellipses, circles, lines, polygons and text remain native.
    Unsupported path-heavy art is preserved as an image fallback in hybrid mode.
    """

    def import_file(self, path: str | Path, *, title: str | None = None, width_in: float = 13.333333, height_in: float = 7.5) -> FigureScene:
        source = Path(path).expanduser().resolve()
        root = ET.fromstring(source.read_bytes())
        view_box = root.attrib.get("viewBox")
        if view_box:
            _, _, vw, vh = (float(item) for item in view_box.replace(",", " ").split())
        else:
            vw = _number(root.attrib.get("width"), 1280.0)
            vh = _number(root.attrib.get("height"), 720.0)
        sx, sy = width_in / max(vw, 1.0), height_in / max(vh, 1.0)
        scene = FigureScene(title=title or source.stem, width=width_in, height=height_in, metadata={"imported_from_svg": str(source)})
        unsupported = False
        sequence = 0
        for node in root.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag in {"svg", "g", "defs", "marker", "tspan"}:
                continue
            sequence += 1
            element_id = node.attrib.get("id", f"svg-{sequence}")
            style = _style(node)
            if tag == "rect":
                x, y = _number(node.attrib.get("x")), _number(node.attrib.get("y"))
                w, h = _number(node.attrib.get("width")), _number(node.attrib.get("height"))
                scene.add(ShapeElement(
                    id=element_id,
                    bounds=Bounds(x * sx, y * sy, max(w * sx, 0.01), max(h * sy, 0.01)),
                    shape=ShapeKind.ROUND_RECTANGLE if _number(node.attrib.get("rx")) > 0 else ShapeKind.RECTANGLE,
                    style=ShapeStyle(
                        fill=_color(style.get("fill"), "bg1"),
                        line=_color(style.get("stroke"), "tx1"),
                        line_width_pt=_number(style.get("stroke-width"), 1.0) * 0.75,
                    ),
                ))
            elif tag in {"circle", "ellipse"}:
                cx, cy = _number(node.attrib.get("cx")), _number(node.attrib.get("cy"))
                rx = _number(node.attrib.get("r"), _number(node.attrib.get("rx"), 1.0))
                ry = _number(node.attrib.get("r"), _number(node.attrib.get("ry"), rx))
                scene.add(ShapeElement(
                    id=element_id,
                    bounds=Bounds((cx - rx) * sx, (cy - ry) * sy, max(2 * rx * sx, 0.01), max(2 * ry * sy, 0.01)),
                    shape=ShapeKind.ELLIPSE,
                    style=ShapeStyle(fill=_color(style.get("fill"), "accent1"), line=_color(style.get("stroke"), "tx1")),
                ))
            elif tag == "line":
                x1, y1, x2, y2 = (_number(node.attrib.get(key)) for key in ("x1", "y1", "x2", "y2"))
                # SVG lines do not carry semantic endpoint identifiers. Keep a
                # small editable line as a free connector between synthetic anchors.
                source_id, target_id = f"{element_id}-source", f"{element_id}-target"
                scene.add(
                    ShapeElement(id=source_id, bounds=Bounds(x1 * sx, y1 * sy, 0.01, 0.01), shape=ShapeKind.ELLIPSE, style=ShapeStyle(fill=ColorRef("tx1", 1.0), line=ColorRef("tx1", 1.0))),
                    ShapeElement(id=target_id, bounds=Bounds(x2 * sx, y2 * sy, 0.01, 0.01), shape=ShapeKind.ELLIPSE, style=ShapeStyle(fill=ColorRef("tx1", 1.0), line=ColorRef("tx1", 1.0))),
                    ConnectorElement(id=element_id, bounds=Bounds(min(x1, x2) * sx, min(y1, y2) * sy, max(abs(x2 - x1) * sx, 0.01), max(abs(y2 - y1) * sy, 0.01)), source_id=source_id, target_id=target_id, line=_color(style.get("stroke"), "tx1")),
                )
            elif tag == "text":
                x, y = _number(node.attrib.get("x")), _number(node.attrib.get("y"))
                text = "".join(node.itertext()).strip()
                if text:
                    scene.add(TextElement(
                        id=element_id,
                        bounds=Bounds(max(0.0, (x - 80) * sx), max(0.0, (y - 20) * sy), min(160 * sx, width_in), min(40 * sy, height_in)),
                        text=text,
                        style=TextStyle(
                            font_size_pt=_number(style.get("font-size"), 18.0) * 0.75,
                            font_name=style.get("font-family"),
                            bold=style.get("font-weight") in {"bold", "600", "700", "800", "900"},
                            color=_color(style.get("fill"), "tx1"),
                            align={"start": "left", "middle": "center", "end": "right"}.get(style.get("text-anchor", "middle"), "center"),
                        ),
                    ))
            elif tag in {"image"}:
                unsupported = True
            elif tag in {"path", "polyline", "polygon", "foreignObject", "use"}:
                unsupported = True
        if unsupported:
            scene.add(ImageElement(
                id="svg-complex-fallback",
                bounds=Bounds(0.0, 0.0, width_in, height_in),
                path=source,
                preserve_as_svg=True,
                z_index=-100,
                alt_text="Complex SVG fallback layer",
                metadata={"fallback_reason": "unsupported SVG path or image content"},
            ))
        return scene
