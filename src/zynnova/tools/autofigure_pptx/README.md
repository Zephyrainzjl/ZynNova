# ZynNova AutoFigure PPTX

`zynnova.tools.autofigure_pptx` is an editable scientific-figure engine for
PowerPoint 2019 and newer.  It keeps the upstream AutoFigure workflow available
through a strict compatibility bridge and adds a semantic scene graph that can
be rendered as:

- native PowerPoint text boxes, AutoShapes, connectors and charts;
- SVG;
- draw.io-compatible mxGraph XML;
- PNG preview;
- JSON scene graph.

## Why a scene graph

Converting a finished SVG into PowerPoint often turns text into paths or creates
hundreds of unstructured shapes.  This project retains scientific semantics
before rendering.  A workflow stage stays a workflow node; a label stays text;
a connection stays a connector; a line/scatter/bar/pie dataset stays an Office
chart.  Consequently, the PowerPoint result can be edited without redrawing it.

## PowerPoint 2019 compatibility

The native renderer emits standard ECMA-376 PresentationML/DrawingML and Office
chart parts.  It avoids Morph, Designer metadata and Microsoft 365-only effects.
Complex unsupported SVG content is rasterized only as a fallback; generated
text and diagram primitives remain native.

Theme behavior:

- no font family is written unless the caller explicitly requests one;
- generated runs inherit the template's PowerPoint theme font;
- theme colors such as `accent1`, `accent2`, `tx1` and `bg1` are written as
  theme references rather than fixed RGB values;
- changing the PPT theme updates native generated text and shapes.

## Installation

Required runtime dependencies:

```bash
pip install python-pptx>=1.0 lxml>=5 pillow>=10 pypdf>=5 cairosvg>=2.7
```

To preserve and use the complete original AutoFigure SVG/mxGraph/refinement/
enhancement pipeline, install the MIT-licensed upstream package separately:

```bash
git clone https://github.com/ResearAI/AutoFigure.git
pip install -e ./AutoFigure
playwright install chromium
```

The PPTX scene/rendering path remains usable without AutoFigure or an API key.

## Quick start

```python
from pathlib import Path
from zynnova.tools.autofigure_pptx import (
    AgentConfig,
    AutoFigurePPTXAgent,
    PPTXRenderConfig,
)

agent = AutoFigurePPTXAgent(
    AgentConfig(
        output_dir=Path("../figure_outputs"),
        pptx=PPTXRenderConfig(
            template_path=Path("my_template.pptx"),
            group_native_diagram=True,
        ),
    )
)

result = agent.generate(
    "Raw microscopy → segmentation → 3D reconstruction → finite-element mesh → electrochemical simulation",
    title="GEM-Sim workflow",
    output_format="all",
)
result.require_success()

print(result.pptx_path)
print(result.editability.editability_score)
```

## Native chart

```python
from zynnova.tools.autofigure_pptx import (
    Bounds, ChartElement, ChartKind, ChartSeries, FigureScene
)

scene = FigureScene("Battery result")
scene.add(
    ChartElement(
        id="cycling-chart",
        bounds=Bounds(1.0, 1.0, 7.5, 4.8),
        chart=ChartKind.LINE,
        categories=["1", "20", "40", "60", "80", "100"],
        series=[ChartSeries("Capacity", [180, 178, 175, 172, 168, 165])],
        title="Cycling performance",
        x_title="Cycle",
        y_title="Capacity (mAh g⁻¹)",
    )
)
result = agent.generate("chart", scene=scene, output_format="pptx")
```

The chart data, series, axes, title and legend are editable in PowerPoint.

## Original AutoFigure passthrough

```python
upstream_result = agent.generate_original(
    "A transformer training pipeline",
    output_format="svg",
    max_iterations=5,
    enable_enhancement=True,
)
```

All keyword arguments are passed to the installed upstream `autofigure`
implementation unchanged.

## Paper to editable PowerPoint

```python
result = agent.generate_from_paper(
    "paper.pdf",
    title="Method overview",
    output_format="all",
)
```

The local extractor handles PDF, Markdown, reStructuredText and plain text.
When exact upstream paper extraction is required, call
`generate_original_from_paper()`.

## CLI

```bash
python -m zynnova.tools.autofigure_pptx \
  --description "Dataset → ML potential → MD → property extraction" \
  --title "Multiscale workflow" \
  --format all \
  --output-dir ../figure_outputs
```

## Editability report

`PPTXValidator` opens the generated OPC package and reports:

- native shape count;
- native text count;
- native connector count;
- native chart count;
- raster fallback count;
- grouped shape count;
- explicit-font count;
- detached connectors;
- package validity;
- editability score.

## Attribution

The compatibility layer is designed around the public API of AutoFigure and
AutoFigure-Edit.  Those projects are MIT licensed and remain separately owned
by their original authors.  This package does not vendor or modify their source
code; it calls an installed upstream package when original behavior is
requested.
