from __future__ import annotations

from zynnova.tools.autofigure_pptx import PaperMethodExtractor


def test_markdown_method_extraction(tmp_path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Introduction\nBackground.\n# Methods\nAcquire images.\nSegment phases.\n# Results\nAccuracy was high.\n",
        encoding="utf-8",
    )
    text = PaperMethodExtractor().extract(paper)
    assert "Acquire images" in text
    assert "Accuracy was high" not in text
