from __future__ import annotations

from zynnova.tools.autofigure_pptx import RuleBasedPlanner, scene_from_json, scene_to_json, validate_scene


def test_rule_planner_round_trip(tmp_path) -> None:
    scene = RuleBasedPlanner().plan(
        "Dataset → model training → uncertainty selection → DFT labeling → retraining",
        title="Active learning",
    )
    report = validate_scene(scene, strict=True)
    assert report.valid
    path = tmp_path / "scene.json"
    scene_to_json(scene, path)
    loaded = scene_from_json(path)
    assert loaded.title == scene.title
    assert len(loaded.elements) == len(scene.elements)
