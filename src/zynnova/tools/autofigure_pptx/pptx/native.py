from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from ..config import PPTXRenderConfig, PPTXMode
from ..exceptions import MissingDependencyError
from ..scene.schema import (
    Bounds,
    ChartElement,
    ConnectorElement,
    ConnectorKind,
    FigureScene,
    ImageElement,
    ShapeElement,
    ShapeKind,
    TextElement,
    TextStyle,
)
from ..scene.validation import validate_scene
from .charts import add_native_chart
from .ooxml import attach_connector, set_arrowheads, set_shape_alt_text, set_solid_fill_transparency
from .theme import apply_color


_SHAPES = {
    ShapeKind.RECTANGLE: MSO_AUTO_SHAPE_TYPE.RECTANGLE,
    ShapeKind.ROUND_RECTANGLE: MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
    ShapeKind.ELLIPSE: MSO_AUTO_SHAPE_TYPE.OVAL,
    ShapeKind.DIAMOND: MSO_AUTO_SHAPE_TYPE.DIAMOND,
    ShapeKind.HEXAGON: MSO_AUTO_SHAPE_TYPE.HEXAGON,
    ShapeKind.CHEVRON: MSO_AUTO_SHAPE_TYPE.CHEVRON,
    ShapeKind.PARALLELOGRAM: MSO_AUTO_SHAPE_TYPE.PARALLELOGRAM,
    ShapeKind.TRIANGLE: MSO_AUTO_SHAPE_TYPE.ISOSCELES_TRIANGLE,
    ShapeKind.CLOUD: MSO_AUTO_SHAPE_TYPE.CLOUD,
    ShapeKind.CYLINDER: MSO_AUTO_SHAPE_TYPE.CAN,
    ShapeKind.DOCUMENT: MSO_AUTO_SHAPE_TYPE.FLOWCHART_DOCUMENT,
    ShapeKind.PROCESS: MSO_AUTO_SHAPE_TYPE.FLOWCHART_PROCESS,
    ShapeKind.DECISION: MSO_AUTO_SHAPE_TYPE.FLOWCHART_DECISION,
}

_CONNECTORS = {
    ConnectorKind.STRAIGHT: MSO_CONNECTOR.STRAIGHT,
    ConnectorKind.ELBOW: MSO_CONNECTOR.ELBOW,
    ConnectorKind.CURVED: MSO_CONNECTOR.CURVE,
}

_ALIGN = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "justify": PP_ALIGN.JUSTIFY}
_VALIGN = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}
_DASH = {
    "solid": MSO_LINE_DASH_STYLE.SOLID,
    "dash": MSO_LINE_DASH_STYLE.DASH,
    "dot": MSO_LINE_DASH_STYLE.ROUND_DOT,
    "dash_dot": MSO_LINE_DASH_STYLE.DASH_DOT,
}


def _apply_text(text_frame: Any, text: str, style: TextStyle, config: PPTXRenderConfig) -> None:
    text_frame.clear()
    text_frame.word_wrap = style.wrap
    text_frame.margin_left = Inches(style.margin_left)
    text_frame.margin_right = Inches(style.margin_right)
    text_frame.margin_top = Inches(style.margin_top)
    text_frame.margin_bottom = Inches(style.margin_bottom)
    text_frame.vertical_anchor = _VALIGN.get(style.valign, MSO_ANCHOR.MIDDLE)
    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = _ALIGN.get(style.align, PP_ALIGN.CENTER)
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(style.font_size_pt or config.default_font_size_pt)
    run.font.bold = style.bold
    run.font.italic = style.italic
    apply_color(run.font.color, style.color)
    # Omit typeface to inherit the template's theme.  Only explicit user
    # requests write a concrete font name.
    if style.font_name:
        run.font.name = style.font_name


def _anchor(bounds, side: str) -> tuple[float, float]:
    if side == "left":
        return bounds.x, bounds.y + bounds.height / 2.0
    if side == "right":
        return bounds.right, bounds.y + bounds.height / 2.0
    if side == "top":
        return bounds.x + bounds.width / 2.0, bounds.y
    return bounds.x + bounds.width / 2.0, bounds.bottom


def _connector_points(source, target) -> tuple[tuple[float, float], tuple[float, float], int, int]:
    sx, sy = source.bounds.center
    tx, ty = target.bounds.center
    dx, dy = tx - sx, ty - sy
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return _anchor(source.bounds, "right"), _anchor(target.bounds, "left"), 3, 1
        return _anchor(source.bounds, "left"), _anchor(target.bounds, "right"), 1, 3
    if dy >= 0:
        return _anchor(source.bounds, "bottom"), _anchor(target.bounds, "top"), 2, 0
    return _anchor(source.bounds, "top"), _anchor(target.bounds, "bottom"), 0, 2


def _image_stream(element: ImageElement, config: PPTXRenderConfig) -> str | io.BytesIO:
    if element.path is not None:
        path = Path(element.path)
        if path.suffix.lower() == ".svg":
            try:
                import cairosvg
            except Exception as exc:
                raise MissingDependencyError("cairosvg is required to place SVG fallbacks in PowerPoint 2019") from exc
            png = cairosvg.svg2png(url=str(path), dpi=config.svg_fallback_dpi)
            return io.BytesIO(png)
        return str(path)
    if element.data_uri:
        header, payload = element.data_uri.split(",", 1)
        raw = base64.b64decode(payload)
        if "svg" in header:
            try:
                import cairosvg
            except Exception as exc:
                raise MissingDependencyError("cairosvg is required for SVG data URIs") from exc
            raw = cairosvg.svg2png(bytestring=raw, dpi=config.svg_fallback_dpi)
        return io.BytesIO(raw)
    raise ValueError(f"image {element.id!r} has neither path nor data_uri")


class NativePPTXRenderer:
    """Render a semantic scene as native DrawingML and Office charts."""

    def __init__(self, config: PPTXRenderConfig | None = None):
        self.config = config or PPTXRenderConfig()

    def render(self, scene: FigureScene, output_path: str | Path) -> tuple[Path, int, dict[str, Any]]:
        validate_scene(scene, strict=self.config.strict).require_valid()
        if self.config.template_path:
            presentation = Presentation(str(self.config.template_path))
        else:
            presentation = Presentation()
        presentation.slide_width = Inches(scene.width)
        presentation.slide_height = Inches(scene.height)
        if self.config.slide_layout_index is None:
            layout_index = min(6, len(presentation.slide_layouts) - 1)
        else:
            layout_index = self.config.slide_layout_index
        slide = presentation.slides.add_slide(presentation.slide_layouts[layout_index])

        if self.config.mode == PPTXMode.SVG:
            import tempfile
            from ..renderers.svg import SVGRenderer

            with tempfile.TemporaryDirectory(prefix="zynnova-autofigure-svg-") as temporary:
                svg_path = SVGRenderer().render(scene, Path(temporary) / "figure.svg")
                source = _image_stream(
                    ImageElement(
                        id="scene-svg",
                        bounds=Bounds(0.0, 0.0, scene.width, scene.height),
                        path=svg_path,
                        preserve_as_svg=True,
                    ),
                    self.config,
                )
                picture = slide.shapes.add_picture(source, Inches(0), Inches(0), Inches(scene.width), Inches(scene.height))
                picture.name = "scene-svg"
                if self.config.include_alt_text:
                    set_shape_alt_text(picture, scene.title)
            output = Path(output_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            presentation.save(output)
            return output, len(presentation.slides) - 1, {
                "slide_index": len(presentation.slides) - 1,
                "native_shape_count": 0,
                "native_connector_count": 0,
                "grouped_shape_count": 0,
                "element_shape_ids": {"scene-svg": picture.shape_id},
            }

        shape_by_id: dict[str, Any] = {}
        native_created: list[Any] = []
        group_candidates: dict[str, list[Any]] = {}

        for element in scene.sorted_elements():
            if isinstance(element, ConnectorElement):
                continue
            b = element.bounds
            if isinstance(element, TextElement):
                shape = slide.shapes.add_textbox(Inches(b.x), Inches(b.y), Inches(b.width), Inches(b.height))
                _apply_text(shape.text_frame, element.text, element.style, self.config)
            elif isinstance(element, ShapeElement):
                shape = slide.shapes.add_shape(_SHAPES[element.shape], Inches(b.x), Inches(b.y), Inches(b.width), Inches(b.height))
                shape.fill.solid()
                apply_color(shape.fill.fore_color, element.style.fill)
                set_solid_fill_transparency(shape, element.style.fill.transparency)
                shape.line.fill.solid()
                apply_color(shape.line.color, element.style.line)
                shape.line.width = Pt(element.style.line_width_pt)
                shape.line.dash_style = _DASH.get(element.style.dash, MSO_LINE_DASH_STYLE.SOLID)
                if element.text:
                    _apply_text(shape.text_frame, element.text, element.text_style, self.config)
            elif isinstance(element, ChartElement):
                shape = add_native_chart(slide, element, inherit_theme_fonts=self.config.inherit_theme_fonts)
            elif isinstance(element, ImageElement):
                source = _image_stream(element, self.config)
                shape = slide.shapes.add_picture(source, Inches(b.x), Inches(b.y), Inches(b.width), Inches(b.height))
            else:
                continue
            shape.name = element.id
            if self.config.include_alt_text:
                set_shape_alt_text(shape, element.alt_text or element.id)
            shape_by_id[element.id] = shape
            native_created.append(shape)
            if element.group_id and not isinstance(element, (ChartElement, ImageElement)):
                group_candidates.setdefault(element.group_id, []).append(shape)

        connector_shapes: list[Any] = []
        for element in scene.sorted_elements():
            if not isinstance(element, ConnectorElement):
                continue
            source_element = scene.get(element.source_id)
            target_element = scene.get(element.target_id)
            start, end, source_site, target_site = _connector_points(source_element, target_element)
            connector = slide.shapes.add_connector(
                _CONNECTORS[element.connector],
                Inches(start[0]), Inches(start[1]), Inches(end[0]), Inches(end[1]),
            )
            connector.name = element.id
            connector.line.fill.solid()
            apply_color(connector.line.color, element.line)
            connector.line.width = Pt(element.line_width_pt)
            connector.line.dash_style = _DASH.get(element.dash, MSO_LINE_DASH_STYLE.SOLID)
            set_arrowheads(connector, start=element.start_arrow, end=element.end_arrow)
            if self.config.connector_attachment:
                source_shape = shape_by_id.get(element.source_id)
                target_shape = shape_by_id.get(element.target_id)
                if source_shape is not None and target_shape is not None:
                    attach_connector(
                        connector,
                        source_shape,
                        target_shape,
                        source_site=element.source_site if element.source_site is not None else source_site,
                        target_site=element.target_site if element.target_site is not None else target_site,
                    )
            if self.config.include_alt_text:
                set_shape_alt_text(connector, element.alt_text or element.id)
            connector_shapes.append(connector)
            shape_by_id[element.id] = connector
            if element.group_id:
                group_candidates.setdefault(element.group_id, []).append(connector)
            if element.label:
                mx = (start[0] + end[0]) / 2.0
                my = (start[1] + end[1]) / 2.0
                label = slide.shapes.add_textbox(Inches(mx - 0.55), Inches(my - 0.18), Inches(1.1), Inches(0.36))
                _apply_text(label.text_frame, element.label, TextStyle(font_size_pt=11.0), self.config)
                native_created.append(label)

        grouped = 0
        if self.config.group_native_diagram:
            for group_id, shapes in group_candidates.items():
                if len(shapes) < 2:
                    continue
                try:
                    group = slide.shapes.add_group_shape(shapes)
                    group.name = group_id
                    grouped += 1
                except Exception:
                    # Grouping is a convenience only.  Individual native shapes
                    # remain fully editable when a chart/picture or unusual XML
                    # prevents grouping.
                    continue

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        presentation.save(output)
        metadata = {
            "slide_index": len(presentation.slides) - 1,
            "native_shape_count": len(native_created),
            "native_connector_count": len(connector_shapes),
            "grouped_shape_count": grouped,
            "element_shape_ids": {key: value.shape_id for key, value in shape_by_id.items()},
        }
        return output, metadata["slide_index"], metadata
