from pathlib import Path

from zynnova.tools.autofigure_pptx import (
    AgentConfig,
    AutoFigurePPTXAgent,
    Bounds,
    ChartElement,
    ChartKind,
    ChartSeries,
    FigureScene,
    TextElement,
    TextStyle,
)


scene = FigureScene("Battery cycling result")
scene.add(
    TextElement(
        id="title",
        bounds=Bounds(0.8, 0.3, 11.8, 0.6),
        text="Cycling performance",
        style=TextStyle(font_size_pt=28, bold=True, align="left"),
    ),
    ChartElement(
        id="chart",
        bounds=Bounds(0.9, 1.1, 11.4, 5.7),
        chart=ChartKind.LINE,
        categories=["1", "20", "40", "60", "80", "100"],
        series=[
            ChartSeries("Baseline", [180, 176, 171, 166, 160, 153]),
            ChartSeries("Proposed", [180, 179, 177, 175, 173, 171]),
        ],
        x_title="Cycle",
        y_title="Capacity (mAh g⁻¹)",
        show_legend=True,
    ),
)

output = (Path.cwd().parent / "autofigure_pptx_outputs").resolve()
result = AutoFigurePPTXAgent(AgentConfig(output_dir=output)).generate(
    "native chart", scene=scene, output_format="pptx"
)
print(result.pptx_path)
