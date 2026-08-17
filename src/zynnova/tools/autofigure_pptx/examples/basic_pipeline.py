from pathlib import Path

from zynnova.tools.autofigure_pptx import AgentConfig, AutoFigurePPTXAgent, PPTXRenderConfig


output_dir = (Path.cwd().parent / "autofigure_pptx_outputs").resolve()
agent = AutoFigurePPTXAgent(
    AgentConfig(
        output_dir=output_dir,
        pptx=PPTXRenderConfig(group_native_diagram=True),
    )
)

result = agent.generate(
    "2D characterization → generative reconstruction → conformal tetrahedral mesh → multiphysics simulation → validation",
    title="Editable multiscale workflow",
    output_format="all",
)
result.require_success()
print("PPTX:", result.pptx_path)
print("SVG:", result.svg_path)
print("draw.io:", result.mxgraph_path)
print("Editability:", result.editability)
