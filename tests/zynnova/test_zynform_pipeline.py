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


def test_physical_scale_evidence_and_fem_render_tracks(tmp_path, image_factory) -> None:
    from zynnova.zynform import PhysicalScaleBasis

    image = image_factory(tmp_path / "scaled_object.png", size=24)
    evidence = tmp_path / "scale_record.txt"
    evidence.write_text("calibration target width = 0.25 m", encoding="utf-8")
    result = run_object(
        ObjectRequest(
            image=image,
            backend="silhouette-extrusion-baseline",
            physical_extent_m=0.25,
            physical_scale_basis=PhysicalScaleBasis.CALIBRATION_TARGET,
            physical_scale_evidence=evidence,
        ),
        ObjectConfig(
            output_directory=str(tmp_path / "scaled-runs"),
            export_formats=("ply",),
            generate_fem=True,
            fem=FEMConfig(
                method=FEMMethod.VOXEL,
                voxel_pitch=0.05,
                maximum_cells=100_000,
            ),
            fem_export_formats=("npz",),
            backend_options={"maximum_image_size": 24, "depth_ratio": 0.25},
        ),
    )
    extent = result.surface_mesh.vertices.max(axis=0) - result.surface_mesh.vertices.min(axis=0)
    assert abs(float(extent.max()) - 0.25) < 1.0e-10
    assert result.fem_surface_mesh is not None
    assert result.exported_fem_surface_files
    assert all(path.is_file() for path in result.exported_fem_surface_files)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["provenance"]["physical_scale_evidence"]["sha256"]
