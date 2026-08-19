from __future__ import annotations

import json
import sys
from pathlib import Path

from zynnova.zynform.external import CommandObjectEngine, GenerativeObjectRequest, ObjectEngineProfile
from zynnova.zynvista.external import CommandSceneEngine, GenerativeSceneRequest, SceneEngineProfile


def _driver(tmp_path: Path, kind: str) -> Path:
    script = tmp_path / f"{kind}_driver.py"
    script.write_text(
        """
import json, pathlib, sys
job = pathlib.Path(sys.argv[sys.argv.index('--zynnova-job') + 1])
data = json.loads(job.read_text())
out = pathlib.Path(data['output_dir'])
if data['contract'] == 'zynnova-zynvista-v1':
    asset = out / 'scene.glb'; asset.write_bytes(b'glb')
    result = {'assets': {'mesh': 'scene.glb'}, 'metadata': {'metric': True}}
else:
    asset = out / 'object.ply'; asset.write_text('ply')
    result = {'assets': {'mesh': 'object.ply'}}
pathlib.Path(data['result_json']).write_text(json.dumps(result))
""",
        encoding="utf-8",
    )
    return script


def test_scene_command_contract(tmp_path: Path) -> None:
    script = _driver(tmp_path, "scene")
    engine = CommandSceneEngine(
        SceneEngineProfile("dummy-world", tmp_path, command=(sys.executable, str(script)))
    )
    result = engine.run(GenerativeSceneRequest(prompt="test world"), tmp_path / "scene-out")
    assert result.assets["mesh"].is_file()


def test_object_command_contract(tmp_path: Path) -> None:
    script = _driver(tmp_path, "object")
    engine = CommandObjectEngine(
        ObjectEngineProfile("dummy-object", tmp_path, command=(sys.executable, str(script)))
    )
    result = engine.run(GenerativeObjectRequest(prompt="test object"), tmp_path / "object-out")
    assert result.assets["mesh"].is_file()
