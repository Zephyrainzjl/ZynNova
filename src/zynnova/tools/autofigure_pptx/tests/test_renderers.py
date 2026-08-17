from __future__ import annotations

import zipfile

from lxml import etree

from zynnova.tools.autofigure_pptx import (
    AgentConfig,
    AutoFigurePPTXAgent,
    Bounds,
    ChartElement,
    ChartKind,
    ChartSeries,
    FigureScene,
    PPTXRenderConfig,
    PPTXValidator,
    RuleBasedPlanner,
    TextElement,
    TextStyle,
)


def test_pipeline_pptx_is_native_and_theme_inherited(tmp_path) -> None:
    agent = AutoFigurePPTXAgent(
        AgentConfig(
            output_dir=tmp_path,
            pptx=PPTXRenderConfig(group_native_diagram=False),
        )
    )
    result = agent.generate(
        "Microscopy → segmentation → reconstruction → finite-element mesh → simulation",
        title="Workflow",
        output_format="all",
    )
    assert result.success, result.error
    assert result.pptx_path and result.pptx_path.is_file()
    assert result.svg_path and result.svg_path.is_file()
    assert result.mxgraph_path and result.mxgraph_path.is_file()
    report = result.editability
    assert report is not None and report.package_valid
    assert report.native_shape_count >= 6
    assert report.native_connector_count >= 4
    assert report.native_text_count >= 6
    assert report.picture_count == 0
    assert report.detached_connector_count == 0
    assert report.editability_score >= 8.5

    with zipfile.ZipFile(result.pptx_path) as archive:
        slide = etree.fromstring(archive.read("ppt/slides/slide1.xml"))
        ns = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        # No explicit typeface means PowerPoint uses the template theme font.
        assert not slide.xpath(".//a:latin[@typeface]", namespaces=ns)
        assert slide.xpath(".//a:stCxn", namespaces=ns)
        assert slide.xpath(".//a:endCxn", namespaces=ns)


def test_native_chart_is_editable(tmp_path) -> None:
    scene = FigureScene("Chart")
    scene.add(
        TextElement(id="title", bounds=Bounds(0.5, 0.2, 8.0, 0.5), text="Editable chart", style=TextStyle(font_size_pt=24, bold=True)),
        ChartElement(
            id="chart",
            bounds=Bounds(0.7, 1.0, 8.0, 4.5),
            chart=ChartKind.LINE,
            categories=["1", "2", "3"],
            series=[ChartSeries("A", [1.0, 2.0, 3.0]), ChartSeries("B", [1.5, 1.8, 2.7])],
            x_title="Step",
            y_title="Value",
        ),
    )
    result = AutoFigurePPTXAgent(AgentConfig(output_dir=tmp_path)).generate("chart", scene=scene, output_format="pptx")
    assert result.success, result.error
    report = PPTXValidator().validate(result.pptx_path)
    assert report.native_chart_count == 1
    assert report.picture_count == 0
    with zipfile.ZipFile(result.pptx_path) as archive:
        assert any(name.startswith("ppt/charts/chart") for name in archive.namelist())
        assert any(name.startswith("ppt/embeddings/") for name in archive.namelist())
