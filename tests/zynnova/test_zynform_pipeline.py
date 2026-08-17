from __future__ import annotations

import json

from zynnova.geometry import tetra_quality, triangle_quality
from zynnova.zynform import (
    FEMConfig,
    FEMMethod,
    ObjectConfig,
    ObjectRequest,
    run_object,
)


def test_image_to_object_surface_multiformat_and_fem(tmp_path, image_factory) -> None:
    image = image_factory(tmp_path / "object.png", size=28)
    result = run_object(
        ObjectRequest(
            image=image,
            prompt="compact engineering component",
            backend="silhouette-extrusion-baseline",
            physical_extent_m=1.0,
        ),
        ObjectConfig(
            output_directory=str(tmp_path / "runs"),
            export_formats=("obj", "ply", "stl", "npz", "glb"),
            normalize_extent=1.0,
            generate_fem=True,
            fem=FEMConfig(
                method=FEMMethod.VOXEL,
                voxel_pitch=0.20,
                maximum_cells=100_000,
                require_watertight=True,
            ),
            fem_export_formats=("vtk", "msh", "inp", "npz"),
            backend_options={"maximum_image_size": 28, "depth_ratio": 0.28},
        ),
    )

    surface = triangle_quality(result.surface_mesh)
    assert surface.watertight
    assert surface.degenerate_faces == 0
    assert surface.nonmanifold_edges == 0
    assert result.volume_mesh is not None
    volume = tetra_quality(result.volume_mesh)
    assert volume.fem_ready
    assert volume.inverted_cells == 0
    assert volume.degenerate_cells == 0
    assert {path.suffix for path in result.exported_surface_files} >= {
        ".obj", ".ply", ".stl", ".npz", ".glb"
    }
    assert {path.suffix for path in result.exported_volume_files} == {
        ".vtk", ".msh", ".inp", ".npz"
    }
    assert all(path.is_file() for path in (*result.exported_surface_files, *result.exported_volume_files))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert any(event["name"] == "fem_generated" for event in manifest["events"])
