from __future__ import annotations

from pathlib import Path

from ..exceptions import MissingDependencyError


class PreviewRenderer:
    def __init__(self, *, dpi: int = 160):
        self.dpi = dpi

    def svg_to_png(self, svg_path: str | Path, png_path: str | Path) -> Path:
        try:
            import cairosvg
        except Exception as exc:
            raise MissingDependencyError("cairosvg is required for PNG previews") from exc
        output = Path(png_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(url=str(Path(svg_path).resolve()), write_to=str(output), dpi=self.dpi)
        return output
