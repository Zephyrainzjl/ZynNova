from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Sequence

from .schema import Bounds, ConnectorElement, FigureScene, ShapeElement, TextElement


class AutoLayoutEngine:
    """Deterministic overlap-removing layout for editable scientific diagrams."""

    def __init__(self, *, margin: float = 0.55, horizontal_gap: float = 0.55, vertical_gap: float = 0.45):
        self.margin = margin
        self.horizontal_gap = horizontal_gap
        self.vertical_gap = vertical_gap

    def pipeline(self, scene: FigureScene, node_ids: Sequence[str], *, title_height: float = 0.75) -> FigureScene:
        if not node_ids:
            return scene
        nodes = [scene.get(item) for item in node_ids]
        count = len(nodes)
        usable_width = scene.width - 2.0 * self.margin
        max_columns = min(count, 5)
        columns = max(1, max_columns)
        rows = ceil(count / columns)
        gap_x = self.horizontal_gap
        gap_y = self.vertical_gap
        cell_width = (usable_width - gap_x * (columns - 1)) / columns
        usable_height = scene.height - title_height - self.margin * 1.5
        cell_height = (usable_height - gap_y * (rows - 1)) / rows
        for index, node in enumerate(nodes):
            row, col = divmod(index, columns)
            width = min(max(1.45, cell_width * 0.82), cell_width)
            height = min(max(0.85, cell_height * 0.55), cell_height)
            x = self.margin + col * (cell_width + gap_x) + (cell_width - width) / 2.0
            y = title_height + self.margin * 0.45 + row * (cell_height + gap_y) + (cell_height - height) / 2.0
            node.bounds = Bounds(x, y, width, height)
        return scene

    def clamp(self, scene: FigureScene) -> FigureScene:
        for element in scene.elements:
            bounds = element.bounds
            x = min(max(0.0, bounds.x), max(0.0, scene.width - bounds.width))
            y = min(max(0.0, bounds.y), max(0.0, scene.height - bounds.height))
            element.bounds = Bounds(x, y, min(bounds.width, scene.width), min(bounds.height, scene.height))
        return scene
