from __future__ import annotations

from pathlib import Path

from zynnova.zynform import (
    FEMConfig,
    FEMMethod,
    ObjectConfig,
    ObjectRequest,
    run_object,
)

request = ObjectRequest(
    image=Path("inputs/object.png"),
    prompt="a single complete engineering component, preserve small geometric details",
    backend="trellis2",
    physical_extent_m=1.0e-3,
)
config = ObjectConfig(
    output_directory="zynnova_runs/zynform_example",
    export_formats=("glb", "obj", "ply", "stl"),
    generate_fem=True,
    fem=FEMConfig(method=FEMMethod.AUTO, maximum_cells=2_000_000),
    backend_options={
        "repository": "external/zynnova/trellis2",
        "model_id": "microsoft/TRELLIS.2-4B",
        "device": "cuda",
    },
)
result = run_object(request, config)
print(result.run_directory)
print(result.surface_quality)
print(result.volume_quality)
