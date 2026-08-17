from __future__ import annotations

import json

import numpy as np

from zynnova.zynmorph import (
    GenerationConfig,
    MicrostructureCondition,
    MicrostructureVolume,
    run_zynmorph,
)


def test_conditional_microstructure_exact_counts_percolation_and_fem(tmp_path) -> None:
    condition = MicrostructureCondition(
        shape=(5, 6, 7),
        phase_fractions={1: 0.50, 2: 0.30, 5: 0.20},
        voxel_size_m=(80e-9, 100e-9, 120e-9),
        correlation_lengths_voxels={1: (2.0, 2.5, 3.0), 2: 1.5, 5: 1.0},
        interface_affinity={(1, 2): 0.5, (2, 5): -0.2},
        percolation_axes={1: (0, 1, 2), 2: (2,)},
        manufacturing={"calendering_ratio": 0.12, "binder_fraction": 0.20},
        periodic=(False, True, True),
        seed=123,
    )
    config = GenerationConfig(
        backend="spectral-exact",
        refinement_steps=2,
        output_directory=str(tmp_path / "runs"),
        export_volume_formats=("npz", "npy", "raw", "tiff"),
        export_mesh_formats=("vtk", "msh", "inp"),
        maximum_tetrahedra=10_000,
    )
    result = run_zynmorph(condition, config)

    labels = result.generation.volume.labels
    counts = {int(phase): int(np.count_nonzero(labels == phase)) for phase in np.unique(labels)}
    assert counts == condition.exact_phase_counts()
    assert result.fem.mesh.n_cells == labels.size * 6
    assert result.fem.quality.fem_ready
    assert result.fem.quality.inverted_cells == 0
    assert result.fem.quality.degenerate_cells == 0

    for phase, axes in condition.percolation_axes.items():
        observed = result.metrics.phases[phase].percolates
        for axis in axes:
            assert observed[axis], (phase, axis, observed)

    expected_artifacts = {
        "volume-npz",
        "volume-npy",
        "volume-raw",
        "volume-tiff",
        "volume-metadata",
        "mesh-vtk",
        "mesh-msh",
        "mesh-inp",
        "mesh-quality",
        "condition",
        "generation",
        "metrics",
        "manifest",
    }
    assert expected_artifacts <= set(result.artifacts)
    assert all(path.is_file() for path in result.artifacts.values())

    restored = MicrostructureVolume.load_npz(result.artifacts["volume-npz"])
    assert np.array_equal(restored.labels, labels)
    assert restored.voxel_size_m == condition.voxel_size_m

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["workflow"] == "zynnova.zynmorph.generate"
    assert any(event["name"] == "meshing_completed" for event in manifest["events"])
