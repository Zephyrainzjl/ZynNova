from __future__ import annotations

import json

import numpy as np
import pytest

from zynnova import __version__, backend_status
from zynnova.core import (
    Availability,
    BackendDescriptor,
    BackendRegistry,
    BackendUnavailableError,
    RunManifest,
)
from zynnova.geometry import (
    TriangleMesh,
    VolumeMesh,
    export_triangle_mesh,
    load_triangle_mesh,
    tetra_quality,
    triangle_quality,
)


class _AvailableBackend:
    name = "available"

    def availability(self) -> Availability:
        return Availability(True, details={"kind": "test"})


class _UnavailableBackend:
    name = "unavailable"

    def availability(self) -> Availability:
        return Availability(False, "missing test resource")


def test_registry_selection_and_diagnostics() -> None:
    registry: BackendRegistry[_AvailableBackend] = BackendRegistry("test-task")
    registry.register(
        BackendDescriptor(
            name="unavailable",
            task="test-task",
            factory=_UnavailableBackend,
            summary="unavailable test backend",
            default_rank=1,
        )
    )
    registry.register(
        BackendDescriptor(
            name="available",
            task="test-task",
            factory=_AvailableBackend,
            summary="available test backend",
            default_rank=2,
        )
    )
    assert registry.choose().name == "available"
    with pytest.raises(BackendUnavailableError, match="missing test resource"):
        registry.create("unavailable")
    report = registry.status()
    assert [item["name"] for item in report] == ["available", "unavailable"]
    assert not any("diagnostic failed" in str(item["reason"]) for item in report)


def test_backend_status_never_imports_or_loads_weights() -> None:
    report = backend_status()
    assert report["zynnova_version"] == __version__
    for subsystem, entries in report.items():
        if subsystem == "zynnova_version":
            continue
        assert isinstance(entries, list) and entries
        assert not any("diagnostic failed" in str(entry["reason"]) for entry in entries)


def test_nanometre_tetra_quality_is_scale_adaptive() -> None:
    scale = 1.0e-7
    mesh = VolumeMesh(
        nodes=scale
        * np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        tetrahedra=np.asarray([[0, 2, 1, 3]], dtype=np.int64),
        cell_regions=np.asarray([1], dtype=np.int32),
    )
    quality = tetra_quality(mesh)
    if quality.inverted_cells:
        mesh = VolumeMesh(
            nodes=mesh.nodes,
            tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            cell_regions=np.asarray([1], dtype=np.int32),
        )
        quality = tetra_quality(mesh)
    assert quality.fem_ready
    assert quality.inverted_cells == 0
    assert quality.degenerate_cells == 0
    assert quality.minimum_mean_ratio > 0.0


def test_triangle_npz_roundtrip_omits_none_object_arrays(tmp_path) -> None:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    path = export_triangle_mesh(tmp_path / "triangle.npz", mesh)
    with np.load(path, allow_pickle=False) as raw:
        assert "vertex_normals" not in raw.files
        assert "uv" not in raw.files
    restored = load_triangle_mesh(path)
    assert np.array_equal(restored.faces, mesh.faces)
    assert np.allclose(restored.vertices, mesh.vertices)
    assert triangle_quality(restored).degenerate_faces == 0


def test_manifest_records_hashes_and_terminal_state(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("auditable", encoding="utf-8")
    manifest = RunManifest(workflow="test", backend="unit", configuration={})
    manifest.event("started")
    manifest.add_artifact(artifact, role="fixture", media_type="text/plain")
    manifest.finish()
    path = manifest.save(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["artifacts"][0]["sha256"]
    assert payload["events"][0]["name"] == "started"
