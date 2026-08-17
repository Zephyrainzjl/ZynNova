from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AutoFigurePPTXAgent
from .config import AgentConfig, PPTXMode, PPTXRenderConfig, PowerPointTarget
from .scene import scene_from_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate editable scientific figures for PowerPoint 2019+")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--description")
    source.add_argument("--paper", type=Path)
    source.add_argument("--scene", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path, default=Path("autofigure_outputs"))
    parser.add_argument("--format", choices=("pptx", "svg", "mxgraphxml", "all"), default="all")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--pptx-mode", choices=tuple(item.value for item in PPTXMode), default=PPTXMode.HYBRID.value)
    parser.add_argument("--target", choices=tuple(item.value for item in PowerPointTarget), default=PowerPointTarget.PPT2019.value)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig(
        output_dir=args.output_dir,
        pptx=PPTXRenderConfig(
            template_path=args.template,
            mode=PPTXMode(args.pptx_mode),
            target=PowerPointTarget(args.target),
        ),
    )
    agent = AutoFigurePPTXAgent(config)
    if args.scene:
        result = agent.generate("scene input", scene=scene_from_json(args.scene), title=args.title, output_format=args.format)
    elif args.paper:
        result = agent.generate_from_paper(args.paper, title=args.title, output_format=args.format)
    else:
        result = agent.generate(args.description, title=args.title, output_format=args.format)
    print(json.dumps({
        "success": result.success,
        "pptx_path": str(result.pptx_path) if result.pptx_path else None,
        "svg_path": str(result.svg_path) if result.svg_path else None,
        "mxgraph_path": str(result.mxgraph_path) if result.mxgraph_path else None,
        "preview_path": str(result.preview_path) if result.preview_path else None,
        "score": result.final_score,
        "editability_score": result.editability.editability_score if result.editability else None,
        "error": result.error,
    }, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
