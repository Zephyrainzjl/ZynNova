from __future__ import annotations

from zynnova.tools.autofigure_pptx import SVGSceneImporter, ShapeElement, TextElement


def test_simple_svg_import_keeps_text_and_shapes_native(tmp_path) -> None:
    svg = tmp_path / "simple.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">'
        '<rect id="box" x="20" y="30" width="140" height="70" rx="10" fill="#4472C4"/>'
        '<text id="label" x="90" y="68" text-anchor="middle" font-size="20">Encoder</text>'
        '</svg>',
        encoding="utf-8",
    )
    scene = SVGSceneImporter().import_file(svg)
    assert any(isinstance(item, ShapeElement) for item in scene.elements)
    assert any(isinstance(item, TextElement) and item.text == "Encoder" for item in scene.elements)
