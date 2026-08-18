# ZynNova

<p align="center">
  <strong>Materials intelligence, multiphysics simulation, generative 3D engineering, high-quality meshing, and speech workflows in one extensible Python/C++ framework.</strong>
</p>

<p align="center">
  <code>Python ≥ 3.10</code> · <code>C++17</code> · <code>scikit-build-core</code> · <code>pybind11</code>
</p>

## Overview

**ZynNova** is a standalone scientific and generative-engineering framework with a single public namespace:

```python
import zynnova
```

The project combines a retained materials-science stack with isolated subframeworks for difficult generation and reconstruction tasks. The architecture is deliberately modular: heavy models, native kernels, FEM utilities, data pipelines, and 3D/voice backends can evolve independently without forcing unrelated modules to share the same dependency stack.

The major public subsystems are:

| Module | Purpose |
|---|---|
| `zynnova.structure` | Crystal, molecular, and polymer structures; graph conversion and round-trips |
| `zynnova.data` | Dataset adapters, schemas, validation, statistics, and local/remote data pipelines |
| `zynnova.dft` | DFT-oriented workflows, electronic calculations, AIMD, and native kernels |
| `zynnova.dynamics` | Relaxation, MD ensembles, trajectory analysis, constraints, and observers |
| `zynnova.ml` | Materials ML, interatomic-potential research, generation, prediction, and model registries |
| `zynnova.physics` | Physics-discovery and symbolic/numerical utilities |
| `zynnova.visualization` | Reusable scientific visualization workflows |
| `zynnova.zynsim` | Battery multiphysics, FEM, phase-field, multiscale, inverse, and digital-twin workflows |
| `zynnova.zynmorph` | Conditional battery microstructures, COMSOL export, conforming multi-material surfaces, and adaptive TetGen meshing |
| `zynnova.zynvista` | Image/video-conditioned 3D scene reconstruction and world-generation adapters |
| `zynnova.zynform` | Image-to-object generation, geometry repair, physical scaling, and FEM-ready export |
| `zynnova.zynvox` | Voice conversion, speech synthesis, evaluation, provenance, CLI, and optional UI |

---

## ZynMorph: battery microstructures and production FEM meshes

ZynMorph is the microstructure-generation and meshing layer for heterogeneous battery materials. It supports labeled multi-phase volumes, conditional generation/reconstruction, morphology metrics, COMSOL Mesh-v4 text export, and two explicit meshing backends:

- `method="tetgen"` — **production path**. Builds a conforming multi-material PLC and calls the compiled TetGen 1.6.0 C++ kernel through pybind11.
- `method="structured"` — compatibility/debug path. Splits each voxel into six tetrahedra. It is intentionally not the default production method.

### COMSOL Hex8 topology correction

The streaming Hex8 exporter uses **COMSOL tensor-product local node ordering**, not VTK/cyclic ordering. This matters because an incorrect local ordering can make COMSOL report that two elements connected to a shared face lie on the same side of that face.

The current exporter provides topology evidence before a Python-path Hex8 write:

```python
from zynnova.zynmorph import audit_comsol_hex8_topology

audit = audit_comsol_hex8_topology(
    (3, 3, 3),
    spacing=(2.25e-7, 4.5e-7, 2.25e-7),
)

assert audit.valid
assert audit.nonpositive_jacobians == 0
assert audit.same_side_shared_faces == 0
assert audit.overconnected_faces == 0
```

The same tensor ordering is implemented in the native C++ voxel backend. A current native build reports the convention as:

```text
comsol-v4-tensor-1
```

### Full and diagnostic MPHTXT export

Full Hex8 volume with domain entities, exterior faces, and material interfaces:

```python
from zynnova.zynmorph import export_voxel_comsol_mphtxt

report = export_voxel_comsol_mphtxt(
    "electrode_hex8.mphtxt",
    volume,
    element_type="hex8",
    include_domain_entity_indices=True,
    include_volume_elements=True,
    validate_topology=True,
    verify=True,
)
```

Surface-only diagnostic export, useful when isolating whether a COMSOL import failure comes from volume topology or from the surface/interface representation:

```python
report = export_voxel_comsol_mphtxt(
    "electrode_surface_only.mphtxt",
    volume,
    element_type="hex8",
    include_domain_entity_indices=False,
    include_volume_elements=False,
    include_exterior_boundaries=True,
    include_material_interfaces=True,
    prefer_native=False,
    verify=True,
)

assert report.diagnostic_mode == "surface-only"
```

### Adaptive multi-material TetGen pipeline

The production tetrahedral workflow is:

```text
multi-phase voxel field
        ↓
non-manifold voxel-junction audit
        ↓
minimal deterministic junction regularization
        ↓
one globally conforming multi-material PLC
        ↓
interface/junction-preserving surface smoothing
        ↓
region seeds + per-phase size constraints
        ↓
TetGen 1.6 C++ constrained-Delaunay tetrahedralization
        ↓
quality / orientation / region checks
        ↓
COMSOL MPHTXT · VTK · Gmsh MSH · Abaqus INP
```

Internal material interfaces are emitted once and shared by both neighboring materials. Multi-phase junctions use common vertices, avoiding the duplicated/coincident interface surfaces produced by independently meshing each phase.

Example:

```python
from zynnova.zynmorph import (
    LocalRefinementZone,
    TetGenMeshingConfig,
    mesh_microstructure,
)

voxel_volume = float(volume.voxel_size_m[0] * volume.voxel_size_m[1] * volume.voxel_size_m[2])

config = TetGenMeshingConfig(
    radius_edge_ratio=1.45,
    minimum_dihedral_degrees=8.0,
    optimization_level=2,
    phase_maximum_tetra_volume_m3={
        1: 0.45 * voxel_volume,
        2: 0.80 * voxel_volume,
        5: 0.35 * voxel_volume,
        7: 0.18 * voxel_volume,
        9: 0.12 * voxel_volume,
    },
    local_refinement_zones=(
        LocalRefinementZone(
            center_m_xyz=(2.0e-6, 1.5e-6, 1.0e-6),
            radius_m=0.6e-6,
            maximum_tetra_volume_m3=0.08 * voxel_volume,
            name="crack_tip",
        ),
    ),
)

fem = mesh_microstructure(
    volume,
    method="tetgen",
    tetgen_config=config,
    maximum_tetrahedra=2_000_000,
)

assert fem.backend.startswith("tetgen")
assert fem.quality.fem_ready
assert fem.quality.inverted_cells == 0
assert fem.quality.degenerate_cells == 0
```

The resulting mesh is **not** constrained to six identical tetrahedra per voxel. TetGen can refine different phases and local regions at different spatial scales.

---

## Installation

### 1. Core editable installation

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Portable difficult-task dependencies

```bash
python -m pip install -e ".[zynnova]"
```

### 3. ZynMorph + native TetGen

TetGen is a separately licensed third-party component. Before vendoring its source, read `THIRD_PARTY_NOTICES.md` and the upstream license.

Vendor the pinned TetGen 1.6.0 source:

```bash
python scripts/vendor_tetgen.py --accept-agpl
```

Then build/install the pybind11 extension:

```bash
python -m pip install -e ".[zynmorph-tetgen]" -v
```

Check the native backend:

```bash
python -c "from zynnova.zynmorph import tetgen_native_status; print(tetgen_native_status())"
```

A successful install should report `available=True` and a path to `_zynmorph_tetgen_native`.

### Windows / PowerShell

ZynNova uses `scikit-build-core` + CMake + Ninja/your selected generator. A working C++17 compiler must be visible to CMake.

Recommended sequence from the repository root:

```powershell
python scripts/vendor_tetgen.py --accept-agpl

Remove-Item `
  -Recurse `
  -Force `
  .\build\cp312-cp312-win_amd64 `
  -ErrorAction SilentlyContinue

python -m pip install -e ".[zynmorph-tetgen]" -v
```

The exact build directory depends on your Python ABI and platform. Removing a stale build directory is useful after changing CMake or third-party source files.

Verify the compiled modules:

```powershell
python -c "from zynnova._native import _zynmorph_tetgen_native as m; print(m.__file__)"
python -c "from zynnova.zynmorph import tetgen_native_status; print(tetgen_native_status())"
```

---

## Testing the new COMSOL + TetGen functionality

The dedicated notebook is:

```text
notebooks/ZynNova_ZynMorph_TetGen_COMSOL_New_Features_Test.ipynb
```

It covers:

1. Python/package/native-extension preflight.
2. COMSOL Hex8 topology audit at the previously problematic nanometre-scale spacing.
3. Full Hex8 MPHTXT export.
4. Surface-only MPHTXT diagnostic export.
5. A deterministic six-phase porous positive-electrode microstructure.
6. Non-manifold voxel-junction detection and conservative repair.
7. Global conforming multi-material PLC extraction and audit.
8. TetGen C++ adaptive tetrahedralization with phase-specific and local sizing.
9. Tet4 quality, orientation, and volume-distribution checks.
10. MPHTXT/VTK/MSH/INP + boundary PLY/STL export.
11. Final machine-readable validation summary.

Run:

```bash
jupyter lab notebooks/ZynNova_ZynMorph_TetGen_COMSOL_New_Features_Test.ipynb
```

The notebook requires the native TetGen extension by default. For source-only CI checks you can explicitly allow TetGen cells to be skipped:

```bash
# Linux/macOS
ZYNNOVA_NOTEBOOK_REQUIRE_TETGEN=0 jupyter lab
```

```powershell
# PowerShell
$env:ZYNNOVA_NOTEBOOK_REQUIRE_TETGEN = "0"
jupyter lab
```

Production validation should run with the default requirement enabled.

---

## Conditional microstructure generation

A compact ZynMorph conditional-generation example:

```python
from zynnova.zynmorph import GenerationConfig, MicrostructureCondition, run_zynmorph

condition = MicrostructureCondition(
    shape=(24, 48, 48),
    voxel_size_m=100e-9,
    phase_fractions={1: 0.58, 2: 0.28, 5: 0.14},
    correlation_lengths_voxels={
        1: (4.0, 5.0, 5.0),
        2: (3.0, 3.0, 3.0),
        5: (6.0, 2.5, 2.5),
    },
    seed=20260818,
)

config = GenerationConfig(
    backend="spectral-exact",
    mesh_backend="tetgen",
    export_volume_formats=("npz",),
    export_mesh_formats=("vtk", "msh", "inp", "mphtxt"),
)

run = run_zynmorph(condition, config)
print(run.output_directory)
```

---

## 3D generation and reconstruction modules

### ZynVista

Scene-level workflows:

- image/video-conditioned metric reconstruction,
- camera/depth/point-cloud/mesh/3DGS preservation,
- large-world backend adapters,
- style-editing backend contracts,
- DCC and interchange exports.

### ZynForm

Object-level workflows:

- image-to-3D backend adapters,
- high-fidelity geometry/PBR asset preservation,
- surface repair and physical scaling,
- surface-to-volume FEM conversion,
- OBJ/PLY/STL/GLB and engineering mesh exports.

### ZynVox

Speech workflows:

- speech-to-speech voice conversion,
- zero-shot speech synthesis,
- streaming-capable backend contracts,
- consent/provenance controls,
- content/speaker/latency evaluation,
- Python, CLI, and optional Gradio interfaces.

Inspect registered difficult-task backends without loading model weights:

```bash
zynnova status
python -m zynnova status
```

---

## Repository layout

```text
ZynNova/
├── src/zynnova/
│   ├── structure/
│   ├── data/
│   ├── dft/
│   ├── dynamics/
│   ├── ml/
│   ├── physics/
│   ├── visualization/
│   ├── zynsim/
│   ├── core/
│   ├── geometry/
│   ├── zynmorph/
│   ├── zynvista/
│   ├── zynform/
│   └── zynvox/
├── cpp/
│   ├── bindings/
│   ├── include/
│   ├── src/
│   ├── tests/
│   └── third_party/tetgen/
├── notebooks/
├── examples/
├── scripts/
├── tests/
├── docs/
├── CMakeLists.txt
└── pyproject.toml
```

---

## Validation

Python tests:

```bash
python -m pytest -q tests/zynnova
```

TetGen/COMSOL-focused tests:

```bash
python -m pytest -q \
  tests/zynnova/test_comsol_hex_orientation.py \
  tests/zynnova/test_zynmorph_comsol.py \
  tests/zynnova/test_zynmorph_tetgen_surface.py \
  tests/zynnova/test_zynmorph_tetgen_build_contract.py
```

Native C++ tests can be enabled at CMake configure time:

```bash
cmake -S . -B build/native-tests \
  -DZYNNOVA_BUILD_CPP_TESTS=ON
cmake --build build/native-tests --config Release
ctest --test-dir build/native-tests --output-on-failure
```

---

## Troubleshooting

### `tetgen_native_status().available == False`

Check that the source was vendored and rebuild:

```bash
python scripts/vendor_tetgen.py --accept-agpl --force
python -m pip install -e ".[zynmorph-tetgen]" -v
```

Then inspect:

```python
from zynnova.zynmorph import tetgen_native_status
print(tetgen_native_status())
```

### CMake succeeds at compilation but fails during wheel installation

Delete the stale build directory and reinstall. Current CMake rules treat README/vendor-manifest files as optional and only require the actual TetGen source/license files plus `SOURCE_LOCK.json`.

### TetGen reports an invalid PLC

Do not independently mesh every material surface. Use ZynMorph's global multi-material PLC path and inspect:

```python
from zynnova.zynmorph import (
    count_nonmanifold_voxel_edges,
    regularize_nonmanifold_junctions,
    extract_multiphase_plc,
    audit_multiphase_plc,
)
```

A production run should reach a PLC audit with zero degenerate/duplicate faces, no open/non-manifold region edges, and no orientation conflicts.

### COMSOL reports two neighboring Hex8 cells on the same side of a shared face

First run:

```python
from zynnova.zynmorph import audit_comsol_hex8_topology
print(audit_comsol_hex8_topology((2, 2, 2)))
```

A current build must report `valid=True` and `same_side_shared_faces=0`. If a previously installed native voxel module is still being imported, rebuild ZynNova and verify its connectivity convention.

---

## Licensing

Original ZynNova source is distributed under the project license in `LICENSE`.

**TetGen is a separate third-party component licensed under AGPL-3.0-or-later.** A ZynNova build that links/distributes TetGen is subject to the applicable TetGen license terms. The source lock, copied source/license files, and notices are kept under:

```text
cpp/third_party/tetgen/
THIRD_PARTY_NOTICES.md
```

Other external research models and weights retain their own licenses and are not implicitly relicensed by ZynNova.

---

## Design principles

- One public namespace: `zynnova`.
- No silent fallback from production TetGen meshing to structured six-tet voxel splitting.
- Explicit physical units and coordinate conventions.
- Native C++ acceleration behind stable Python APIs.
- Multi-material conformity is checked before volume meshing.
- COMSOL exports are validated structurally before being treated as production artifacts.
- Heavy research backends are isolated behind explicit adapters.
- Third-party source and model licensing remains traceable.
- New modules should extend registries/contracts rather than modify unrelated subsystems.
