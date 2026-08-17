from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ElementKind(str, Enum):
    TEXT = "text"
    SHAPE = "shape"
    CONNECTOR = "connector"
    CHART = "chart"
    IMAGE = "image"


class ShapeKind(str, Enum):
    RECTANGLE = "rectangle"
    ROUND_RECTANGLE = "round_rectangle"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    CHEVRON = "chevron"
    PARALLELOGRAM = "parallelogram"
    TRIANGLE = "triangle"
    CLOUD = "cloud"
    CYLINDER = "cylinder"
    DOCUMENT = "document"
    PROCESS = "process"
    DECISION = "decision"


class ConnectorKind(str, Enum):
    STRAIGHT = "straight"
    ELBOW = "elbow"
    CURVED = "curved"


class ChartKind(str, Enum):
    LINE = "line"
    COLUMN = "column"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    SCATTER = "scatter"


@dataclass(frozen=True, slots=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounds width and height must be positive")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def padded(self, padding: float) -> "Bounds":
        return Bounds(
            self.x - padding,
            self.y - padding,
            self.width + 2.0 * padding,
            self.height + 2.0 * padding,
        )


@dataclass(slots=True)
class ColorRef:
    """Color expressed either as a theme key or an RGB hex string."""

    value: str = "accent1"
    transparency: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.transparency <= 1.0):
            raise ValueError("transparency must lie in [0, 1]")


@dataclass(slots=True)
class TextStyle:
    font_size_pt: float | None = None
    font_name: str | None = None
    font_role: str = "minor"
    bold: bool = False
    italic: bool = False
    color: ColorRef = field(default_factory=lambda: ColorRef("tx1"))
    align: str = "center"
    valign: str = "middle"
    margin_left: float = 0.08
    margin_right: float = 0.08
    margin_top: float = 0.04
    margin_bottom: float = 0.04
    wrap: bool = True


@dataclass(slots=True)
class ShapeStyle:
    fill: ColorRef = field(default_factory=lambda: ColorRef("accent1", 0.12))
    line: ColorRef = field(default_factory=lambda: ColorRef("accent1"))
    line_width_pt: float = 1.5
    dash: str = "solid"
    radius: float = 0.12


@dataclass(slots=True)
class SceneElement:
    id: str
    bounds: Bounds
    z_index: int = 0
    group_id: str | None = None
    alt_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> ElementKind:
        raise NotImplementedError


@dataclass(slots=True)
class TextElement(SceneElement):
    text: str = ""
    style: TextStyle = field(default_factory=TextStyle)

    @property
    def kind(self) -> ElementKind:
        return ElementKind.TEXT


@dataclass(slots=True)
class ShapeElement(SceneElement):
    shape: ShapeKind = ShapeKind.ROUND_RECTANGLE
    text: str = ""
    style: ShapeStyle = field(default_factory=ShapeStyle)
    text_style: TextStyle = field(default_factory=TextStyle)

    @property
    def kind(self) -> ElementKind:
        return ElementKind.SHAPE


@dataclass(slots=True)
class ConnectorElement(SceneElement):
    source_id: str = ""
    target_id: str = ""
    connector: ConnectorKind = ConnectorKind.STRAIGHT
    label: str | None = None
    line: ColorRef = field(default_factory=lambda: ColorRef("accent1"))
    line_width_pt: float = 1.5
    dash: str = "solid"
    start_arrow: str = "none"
    end_arrow: str = "triangle"
    source_site: int | None = None
    target_site: int | None = None

    @property
    def kind(self) -> ElementKind:
        return ElementKind.CONNECTOR


@dataclass(slots=True)
class ChartSeries:
    name: str
    values: list[float]
    x_values: list[float] | None = None
    color: ColorRef | None = None


@dataclass(slots=True)
class ChartElement(SceneElement):
    chart: ChartKind = ChartKind.LINE
    categories: list[str] = field(default_factory=list)
    series: list[ChartSeries] = field(default_factory=list)
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    show_legend: bool = True
    show_data_labels: bool = False

    @property
    def kind(self) -> ElementKind:
        return ElementKind.CHART


@dataclass(slots=True)
class ImageElement(SceneElement):
    path: Path | None = None
    data_uri: str | None = None
    preserve_as_svg: bool = False
    crop: bool = False

    @property
    def kind(self) -> ElementKind:
        return ElementKind.IMAGE


Element = TextElement | ShapeElement | ConnectorElement | ChartElement | ImageElement


@dataclass(slots=True)
class FigureScene:
    title: str
    width: float = 13.333333
    height: float = 7.5
    elements: list[Element] = field(default_factory=list)
    description: str | None = None
    background: ColorRef = field(default_factory=lambda: ColorRef("bg1"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, *elements: Element) -> "FigureScene":
        self.elements.extend(elements)
        return self

    def get(self, element_id: str) -> Element:
        for element in self.elements:
            if element.id == element_id:
                return element
        raise KeyError(element_id)

    def sorted_elements(self) -> list[Element]:
        return sorted(self.elements, key=lambda item: (item.z_index, item.id))

    def groups(self) -> dict[str, list[Element]]:
        result: dict[str, list[Element]] = {}
        for element in self.elements:
            if element.group_id:
                result.setdefault(element.group_id, []).append(element)
        return result

    def copy_with(self, *, elements: Sequence[Element] | None = None) -> "FigureScene":
        return FigureScene(
            title=self.title,
            width=self.width,
            height=self.height,
            elements=list(self.elements if elements is None else elements),
            description=self.description,
            background=self.background,
            metadata=dict(self.metadata),
        )
