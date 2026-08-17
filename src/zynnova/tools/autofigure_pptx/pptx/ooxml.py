from __future__ import annotations

from typing import Any

from lxml import etree
from pptx.oxml.ns import qn

_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _child(parent: Any, tag: str) -> Any:
    found = parent.find(qn(tag))
    if found is None:
        found = etree.SubElement(parent, qn(tag))
    return found


def set_arrowheads(connector_shape: Any, *, start: str = "none", end: str = "triangle") -> None:
    line = connector_shape.line._get_or_add_ln()  # pyright: ignore[reportPrivateUsage]
    for child_name, arrow in (("a:headEnd", start), ("a:tailEnd", end)):
        existing = line.find(qn(child_name))
        if existing is not None:
            line.remove(existing)
        if arrow and arrow != "none":
            node = etree.SubElement(line, qn(child_name))
            node.set("type", arrow)
            node.set("w", "med")
            node.set("len", "med")


def attach_connector(connector_shape: Any, source_shape: Any, target_shape: Any, *, source_site: int = 2, target_site: int = 0) -> None:
    """Attach a connector to source and target connection sites.

    Connection-site indices are intentionally simple defaults.  PowerPoint can
    reroute the connector interactively after opening the file.
    """

    nv = connector_shape._element.find(qn("p:nvCxnSpPr"))  # pyright: ignore[reportPrivateUsage]
    if nv is None:
        return
    c_nv = nv.find(qn("p:cNvCxnSpPr"))
    if c_nv is None:
        c_nv = etree.SubElement(nv, qn("p:cNvCxnSpPr"))
    for tag, shape, site in (
        ("a:stCxn", source_shape, source_site),
        ("a:endCxn", target_shape, target_site),
    ):
        old = c_nv.find(qn(tag))
        if old is not None:
            c_nv.remove(old)
        node = etree.SubElement(c_nv, qn(tag))
        node.set("id", str(shape.shape_id))
        node.set("idx", str(site))


def set_shape_alt_text(shape: Any, text: str) -> None:
    """Set Office non-visual description used by accessibility tools."""

    element = shape._element  # pyright: ignore[reportPrivateUsage]
    c_nv_pr = element.find(".//" + qn("p:cNvPr"))
    if c_nv_pr is not None:
        c_nv_pr.set("descr", text)


def set_solid_fill_transparency(shape: Any, transparency: float) -> None:
    if transparency <= 0.0:
        return
    sp_pr = shape._element.spPr  # pyright: ignore[reportPrivateUsage]
    solid_fill = sp_pr.find(qn("a:solidFill"))
    if solid_fill is None or len(solid_fill) == 0:
        return
    color = solid_fill[0]
    for old in color.findall(qn("a:alpha")):
        color.remove(old)
    alpha = etree.SubElement(color, qn("a:alpha"))
    alpha.set("val", str(int(round((1.0 - transparency) * 100000))))


def set_chart_text_theme(chart_shape: Any) -> None:
    """Remove explicit typeface declarations from a chart's rich text."""

    chart = chart_shape.chart
    root = chart._element  # pyright: ignore[reportPrivateUsage]
    for latin in root.findall(".//" + qn("a:latin")):
        parent = latin.getparent()
        if parent is not None:
            parent.remove(latin)
    for ea in root.findall(".//" + qn("a:ea")):
        parent = ea.getparent()
        if parent is not None:
            parent.remove(ea)
