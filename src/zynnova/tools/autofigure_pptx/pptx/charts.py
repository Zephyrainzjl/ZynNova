from __future__ import annotations

from typing import Any

from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

from ..scene.schema import ChartElement, ChartKind
from .ooxml import set_chart_text_theme
from .theme import apply_color


_CHART_TYPES = {
    ChartKind.LINE: XL_CHART_TYPE.LINE_MARKERS,
    ChartKind.COLUMN: XL_CHART_TYPE.COLUMN_CLUSTERED,
    ChartKind.BAR: XL_CHART_TYPE.BAR_CLUSTERED,
    ChartKind.AREA: XL_CHART_TYPE.AREA,
    ChartKind.PIE: XL_CHART_TYPE.PIE,
    ChartKind.SCATTER: XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
}


def _build_data(element: ChartElement):
    if element.chart == ChartKind.SCATTER:
        data = XyChartData()
        for item in element.series:
            series = data.add_series(item.name)
            x_values = item.x_values or list(range(len(item.values)))
            for x, y in zip(x_values, item.values):
                series.add_data_point(float(x), float(y))
        return data
    data = CategoryChartData()
    data.categories = element.categories or [str(index + 1) for index in range(max((len(item.values) for item in element.series), default=0))]
    for item in element.series:
        data.add_series(item.name, tuple(float(value) for value in item.values))
    return data


def add_native_chart(slide: Any, element: ChartElement, *, inherit_theme_fonts: bool = True) -> Any:
    data = _build_data(element)
    bounds = element.bounds
    chart_shape = slide.shapes.add_chart(
        _CHART_TYPES[element.chart],
        Inches(bounds.x),
        Inches(bounds.y),
        Inches(bounds.width),
        Inches(bounds.height),
        data,
    )
    chart = chart_shape.chart
    chart.has_title = bool(element.title)
    if element.title:
        chart.chart_title.text_frame.text = element.title
    chart.has_legend = element.show_legend and len(element.series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
    if element.chart != ChartKind.PIE:
        if element.x_title:
            chart.category_axis.has_title = True
            chart.category_axis.axis_title.text_frame.text = element.x_title
        if element.y_title:
            chart.value_axis.has_title = True
            chart.value_axis.axis_title.text_frame.text = element.y_title
        chart.value_axis.has_major_gridlines = True
    for index, series in enumerate(chart.series):
        if index < len(element.series) and element.series[index].color is not None:
            series.format.line.fill.solid()
            apply_color(series.format.line.fill.fore_color, element.series[index].color)
        if element.show_data_labels:
            series.has_data_labels = True
    if inherit_theme_fonts:
        set_chart_text_theme(chart_shape)
    return chart_shape
