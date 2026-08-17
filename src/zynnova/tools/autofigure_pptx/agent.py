from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AgentConfig, UpstreamMode
from .extractor import PaperMethodExtractor
from .importers import SVGSceneImporter
from .judge import SceneJudge
from .pptx import PPTXValidator
from .renderers import MxGraphRenderer, PPTXRenderer, PreviewRenderer, SVGRenderer
from .result import GenerationResult
from .scene import AutoLayoutEngine, FigureScene, RuleBasedPlanner, ScenePlanner, scene_to_json, validate_scene
from .upstream import UpstreamAutoFigureBridge


class AutoFigurePPTXAgent:
    """AutoFigure-compatible scientific illustration agent with native PPTX output.

    Original AutoFigure requests are passed through when the upstream package is
    installed.  Native PPTX generation is scene-graph based and remains usable
    offline without an API key.
    """

    def __init__(self, config: AgentConfig | None = None, *, planner: ScenePlanner | None = None):
        self.config = config or AgentConfig()
        self.planner = planner or RuleBasedPlanner()
        self.upstream = UpstreamAutoFigureBridge(self.config)
        self.extractor = PaperMethodExtractor()
        self.judge = SceneJudge()

    def generate_original(self, description: str, **kwargs: Any) -> Any:
        """Execute the unmodified upstream AutoFigure generation pipeline."""

        return self.upstream.generate(description=description, **kwargs)

    def generate_original_from_paper(self, paper_path: str | Path, **kwargs: Any) -> Any:
        return self.upstream.generate_from_paper(paper_path=str(paper_path), **kwargs)

    def generate(
        self,
        description: str,
        *,
        max_iterations: int | None = None,
        output_format: str = "pptx",
        quality_threshold: float | None = None,
        topic: str | None = None,
        title: str | None = None,
        scene: FigureScene | None = None,
        use_upstream_svg: bool = False,
        enable_enhancement: bool = False,
        **upstream_kwargs: Any,
    ) -> GenerationResult:
        output_format = output_format.lower().replace("-", "")
        if output_format in {"mxgraph", "xml"}:
            output_format = "mxgraphxml"
        if output_format in {"svg", "mxgraphxml"} and self.config.upstream_mode != UpstreamMode.DISABLED:
            status = self.upstream.status()
            if status.available:
                result = self.upstream.generate(
                    description=description,
                    max_iterations=max_iterations or self.config.max_iterations,
                    output_format=output_format,
                    quality_threshold=quality_threshold or self.config.quality_threshold,
                    topic=topic or self.config.topic,
                    enable_enhancement=enable_enhancement,
                    **upstream_kwargs,
                )
                return self._from_upstream(result)
            if self.config.upstream_mode == UpstreamMode.REQUIRED:
                return GenerationResult(success=False, error=status.error or "upstream unavailable")

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = self._slug(title or "scientific_figure")
        try:
            upstream_result = None
            if scene is None and use_upstream_svg and self.upstream.status().available:
                upstream_result = self.upstream.generate(
                    description=description,
                    max_iterations=max_iterations or self.config.max_iterations,
                    output_format="svg",
                    quality_threshold=quality_threshold or self.config.quality_threshold,
                    topic=topic or self.config.topic,
                    enable_enhancement=enable_enhancement,
                    **upstream_kwargs,
                )
                svg_path = getattr(upstream_result, "svg_path", None)
                if svg_path:
                    scene = SVGSceneImporter().import_file(svg_path, title=title)
            if scene is None:
                scene = self.planner.plan(description, topic=topic or self.config.topic, title=title)

            threshold = quality_threshold or self.config.quality_threshold
            iterations = 0
            judge_result = self.judge.evaluate(scene)
            while judge_result.score < threshold and iterations < (max_iterations or self.config.max_iterations) - 1:
                AutoLayoutEngine().clamp(scene)
                judge_result = self.judge.evaluate(scene)
                iterations += 1
            validate_scene(scene, strict=self.config.pptx.strict).require_valid()

            scene_path = output_dir / f"{slug}.scene.json"
            scene_to_json(scene, scene_path)
            svg_path = SVGRenderer().render(scene, output_dir / f"{slug}.svg")
            mxgraph_path = MxGraphRenderer().render(scene, output_dir / f"{slug}.drawio.xml")
            preview_path = None
            try:
                preview_path = PreviewRenderer().svg_to_png(svg_path, output_dir / f"{slug}.png")
            except Exception:
                preview_path = None
            pptx_path = None
            slide_index = None
            editability = None
            metadata: dict[str, Any] = {"judge_feedback": judge_result.feedback}
            if output_format in {"pptx", "all"}:
                pptx_path, slide_index, render_meta = PPTXRenderer(self.config.pptx).render(scene, output_dir / f"{slug}.pptx")
                editability = PPTXValidator(self.config.pptx.target).validate(pptx_path)
                metadata.update(render_meta)
            return GenerationResult(
                success=True,
                svg_path=svg_path if output_format in {"svg", "all", "pptx"} else None,
                mxgraph_path=mxgraph_path if output_format in {"mxgraphxml", "all", "pptx"} else None,
                pptx_path=pptx_path,
                preview_path=preview_path,
                scene_path=scene_path,
                final_score=judge_result.score,
                iterations=iterations + 1,
                slide_index=slide_index,
                editability=editability,
                upstream_result=upstream_result,
                metadata=metadata,
            )
        except Exception as exc:
            return GenerationResult(success=False, error=f"{type(exc).__name__}: {exc}")

    def generate_from_paper(self, paper_path: str | Path, **kwargs: Any) -> GenerationResult:
        methodology = self.extractor.extract(paper_path)
        result = self.generate(methodology, **kwargs)
        result.methodology_text = methodology
        return result

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
        return "_".join(part for part in cleaned.split("_") if part)[:80] or "scientific_figure"

    @staticmethod
    def _from_upstream(result: Any) -> GenerationResult:
        def _path(name: str):
            value = getattr(result, name, None)
            return Path(value) if value else None

        enhanced = tuple(Path(item) for item in (getattr(result, "enhanced_paths", None) or ()))
        return GenerationResult(
            success=bool(getattr(result, "success", False)),
            svg_path=_path("svg_path"),
            mxgraph_path=_path("mxgraph_path"),
            preview_path=_path("preview_path"),
            enhanced_path=_path("enhanced_path"),
            enhanced_paths=enhanced,
            methodology_text=getattr(result, "methodology_text", None),
            final_score=float(getattr(result, "final_score", 0.0) or 0.0),
            upstream_result=result,
            error=getattr(result, "error", None),
        )


# Compatibility aliases matching the upstream README usage.
AutoFigureAgent = AutoFigurePPTXAgent
Config = AgentConfig
