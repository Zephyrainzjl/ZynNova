from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

from lxml import etree

from ..config import PowerPointTarget
from ..result import EditabilityReport


_NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}


class PPTXValidator:
    """Structural validator for native editability and PPT 2019 compatibility."""

    def __init__(self, target: PowerPointTarget = PowerPointTarget.PPT2019):
        self.target = target

    def validate(self, path: str | Path) -> EditabilityReport:
        target = Path(path)
        warnings: list[str] = []
        if not zipfile.is_zipfile(target):
            return EditabilityReport(
                compatibility_target=self.target.value,
                package_valid=False,
                warnings=("not a valid Open Packaging Convention ZIP",),
            )
        counts = {
            "native_shape_count": 0,
            "native_text_count": 0,
            "native_connector_count": 0,
            "native_chart_count": 0,
            "picture_count": 0,
            "grouped_shape_count": 0,
            "explicit_font_count": 0,
            "detached_connector_count": 0,
        }
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "ppt/presentation.xml"}
            package_valid = required.issubset(names)
            if not package_valid:
                warnings.append("missing required PPTX package parts")
            slide_names = sorted(name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
            for name in slide_names:
                root = etree.fromstring(archive.read(name))
                counts["native_shape_count"] += len(root.xpath(".//p:sp", namespaces=_NS))
                counts["native_text_count"] += len(root.xpath(".//a:t", namespaces=_NS))
                connectors = root.xpath(".//p:cxnSp", namespaces=_NS)
                counts["native_connector_count"] += len(connectors)
                counts["native_chart_count"] += len(root.xpath(".//p:graphicFrame", namespaces=_NS))
                counts["picture_count"] += len(root.xpath(".//p:pic", namespaces=_NS))
                counts["grouped_shape_count"] += len(root.xpath(".//p:grpSp", namespaces=_NS))
                counts["explicit_font_count"] += len(root.xpath(".//a:latin[@typeface] | .//a:ea[@typeface]", namespaces=_NS))
                for connector in connectors:
                    if not connector.xpath(".//a:stCxn", namespaces=_NS) or not connector.xpath(".//a:endCxn", namespaces=_NS):
                        counts["detached_connector_count"] += 1
        native_total = counts["native_shape_count"] + counts["native_connector_count"] + counts["native_chart_count"]
        visual_total = native_total + counts["picture_count"]
        native_ratio = native_total / max(visual_total, 1)
        text_bonus = min(1.0, counts["native_text_count"] / max(counts["native_shape_count"], 1))
        attached_ratio = 1.0 - counts["detached_connector_count"] / max(counts["native_connector_count"], 1)
        score = 10.0 * (0.58 * native_ratio + 0.22 * text_bonus + 0.20 * attached_ratio)
        if counts["explicit_font_count"]:
            warnings.append("explicit font names are present; those runs will not follow the PowerPoint theme")
            score -= min(1.0, counts["explicit_font_count"] * 0.05)
        if counts["picture_count"]:
            warnings.append("picture fallbacks are not internally editable")
        return EditabilityReport(
            **counts,
            compatibility_target=self.target.value,
            package_valid=package_valid,
            editability_score=max(0.0, min(10.0, score)),
            warnings=tuple(warnings),
        )
