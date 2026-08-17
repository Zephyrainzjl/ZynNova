from __future__ import annotations

from pathlib import Path

from zynnova.zynvista import SceneConfig, SceneMode, SceneRequest, run_scene

# Configure a separate MapAnything environment/repository through backend_options.
images = tuple(sorted(Path("inputs/scene_views").glob("*.png")))
request = SceneRequest(images=images, mode=SceneMode.RECONSTRUCT, backend="mapanything")
config = SceneConfig(
    output_directory="zynnova_runs/zynvista_example",
    export_formats=("ply", "obj", "glb", "colmap"),
    backend_options={
        "model_id": "facebook/map-anything-apache",
        "device": "cuda",
        "confidence_threshold": 0.35,
    },
)
result = run_scene(request, config)
print(result.run_directory)
print(result.exports)
