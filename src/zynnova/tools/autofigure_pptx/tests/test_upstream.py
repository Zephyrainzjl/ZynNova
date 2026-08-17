from __future__ import annotations

import sys
import types
from pathlib import Path

from zynnova.tools.autofigure_pptx import AgentConfig, AutoFigurePPTXAgent


def test_upstream_arguments_are_passed_through(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeResult:
        success = True
        svg_path = str(tmp_path / "figure.svg")
        mxgraph_path = None
        preview_path = None
        enhanced_paths = []
        final_score = 9.1
        methodology_text = None
        error = None

    class FakeAgent:
        def __init__(self, config):
            calls["agent_config"] = config

        def generate(self, *args, **kwargs):
            calls["generate"] = (args, kwargs)
            Path(FakeResult.svg_path).write_text("<svg/>", encoding="utf-8")
            return FakeResult()

    module = types.ModuleType("autofigure")
    module.Config = FakeConfig
    module.AutoFigureAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "autofigure", module)

    agent = AutoFigurePPTXAgent(
        AgentConfig(
            output_dir=tmp_path,
            generation_api_key="key",
            generation_provider="provider",
            generation_model="model",
        )
    )
    result = agent.generate(
        "description",
        output_format="svg",
        max_iterations=7,
        enable_enhancement=True,
        enhancement_count=3,
    )
    assert result.success
    kwargs = calls["generate"][1]
    assert kwargs["max_iterations"] == 7
    assert kwargs["enable_enhancement"] is True
    assert kwargs["enhancement_count"] == 3
    assert calls["config"]["generation_api_key"] == "key"
