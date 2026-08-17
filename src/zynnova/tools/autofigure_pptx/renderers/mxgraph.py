from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from ..scene.schema import ConnectorElement, FigureScene, ShapeElement, TextElement


_SHAPE_STYLE = {
    "rectangle": "rounded=0;whiteSpace=wrap;html=1;",
    "round_rectangle": "rounded=1;whiteSpace=wrap;html=1;arcSize=12;",
    "ellipse": "ellipse;whiteSpace=wrap;html=1;",
    "diamond": "rhombus;whiteSpace=wrap;html=1;",
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;",
    "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
}


class MxGraphRenderer:
    """Render a scene as draw.io-compatible mxGraph XML."""

    def render(self, scene: FigureScene, output_path: str | Path) -> Path:
        model = Element("mxGraphModel", {"dx": "1422", "dy": "794", "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "1333", "pageHeight": "750", "math": "0", "shadow": "0"})
        root = SubElement(model, "root")
        SubElement(root, "mxCell", {"id": "0"})
        SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        for element in scene.sorted_elements():
            if isinstance(element, ShapeElement):
                style = _SHAPE_STYLE.get(element.shape.value, _SHAPE_STYLE["round_rectangle"])
                cell = SubElement(root, "mxCell", {"id": element.id, "value": element.text, "style": style, "vertex": "1", "parent": "1"})
                b = element.bounds
                SubElement(cell, "mxGeometry", {"x": str(b.x * 100), "y": str(b.y * 100), "width": str(b.width * 100), "height": str(b.height * 100), "as": "geometry"})
            elif isinstance(element, TextElement):
                cell = SubElement(root, "mxCell", {"id": element.id, "value": element.text, "style": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;", "vertex": "1", "parent": "1"})
                b = element.bounds
                SubElement(cell, "mxGeometry", {"x": str(b.x * 100), "y": str(b.y * 100), "width": str(b.width * 100), "height": str(b.height * 100), "as": "geometry"})
            elif isinstance(element, ConnectorElement):
                cell = SubElement(root, "mxCell", {"id": element.id, "value": element.label or "", "style": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;", "edge": "1", "parent": "1", "source": element.source_id, "target": element.target_id})
                SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(tostring(model, encoding="utf-8", xml_declaration=True))
        return output
