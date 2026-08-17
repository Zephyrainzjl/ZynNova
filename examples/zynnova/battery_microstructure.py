from __future__ import annotations

from zynnova.zynmorph import (
    BatteryPhase,
    GenerationConfig,
    MicrostructureCondition,
    run_zynmorph,
)

condition = MicrostructureCondition(
    shape=(48, 64, 64),
    phase_fractions={
        int(BatteryPhase.POSITIVE_ACTIVE): 0.55,
        int(BatteryPhase.POSITIVE_ELECTROLYTE): 0.30,
        int(BatteryPhase.POSITIVE_CBD): 0.15,
    },
    voxel_size_m=100e-9,
    correlation_lengths_voxels={1: (4.0, 5.0, 5.0), 2: 3.0, 5: 2.0},
    interface_affinity={(1, 5): 0.6, (1, 2): -0.15},
    percolation_axes={1: (0, 1, 2), 2: (2,)},
    manufacturing={"calendering_ratio": 0.18, "binder_fraction": 0.15},
    descriptor_targets={"specific_interface_area": 0.12},
    periodic=(False, True, True),
    seed=23,
)

result = run_zynmorph(
    condition,
    GenerationConfig(
        backend="spectral-exact",
        refinement_steps=6,
        export_volume_formats=("npz", "tiff"),
        export_mesh_formats=("vtk", "msh", "inp"),
        output_directory="zynnova_runs/zynmorph_example",
    ),
)
print("run:", result.directory)
print("phase fractions:", result.metrics.phase_fractions)
print("mesh quality:", result.fem.quality)
