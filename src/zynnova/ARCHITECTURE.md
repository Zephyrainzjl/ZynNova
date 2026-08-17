# ZynNova architecture

## Package tree

```text
zynnova/
├── core/          # backend registry, licenses, subprocess, hashing, manifests
├── geometry/      # point/surface/volume types, repair, quality, import/export
├── zynmorph/      # battery microstructure generation → analysis → Tet4 mesh
├── zynvista/      # scene reconstruction/generation → style → scene exports
├── zynform/       # object generation → repair → surface exports → FEM mesh
└── zynvox/        # voice conversion/TTS → normalization → benchmark/provenance
```

The four subframeworks depend only on `core` and, where relevant, `geometry`; none
imports another subframework. This keeps feature additions and dependency changes
local.

## Execution model

```text
validated dataclass request
        │
        ▼
side-effect-free backend discovery ── unavailable reason
        │
        ▼
backend run in-process or isolated argv contract
        │
        ▼
native output validation and preservation
        │
        ├── geometry repair / fusion / exact-count projection
        ├── physical or geometric quality checks
        ├── portable exports
        └── benchmark and provenance
        │
        ▼
manifest.json with hashes, events, artifacts and status
```

### ZynMorph

`MicrostructureCondition` is the canonical condition vector. It supports arbitrary
valid battery phase labels, exact phase fractions, anisotropic correlation lengths,
interface affinities, prescribed percolation axes, manufacturing variables,
descriptor targets, periodic axes, physical voxel spacing, and a reproducible seed.
The pipeline performs exact-count projection, topological constraint enforcement,
descriptor analysis, voxel export, conforming voxel-derived Tet4 meshing, orientation
repair, scale-aware quality checks, and VTK/Gmsh/Abaqus export.

### ZynVista

Reconstruction and world generation have separate registries. Metric dense views can
be fused into a point cloud and surface while native camera/3DGS/mesh outputs remain
available. Style backends operate after reconstruction and cannot silently replace the
engineering mesh. DCC conversion uses native writers where possible and an explicit
Blender bridge for FBX, USD, Alembic and COLLADA.

### ZynForm

The object backend returns its native PBR or surface asset. The pipeline imports it,
repairs topology, applies an explicit unit scale, validates watertightness, exports the
surface, and tetrahedralizes with TetGen, Gmsh, or a deterministic voxel fallback.
FEM quality gates are scale-adaptive and reject inverted or degenerate elements.

### ZynVox

Voice conversion and TTS have separate protocols and registries. Both require a
`ConsentRecord`. The raw backend file is preserved, standardized PCM WAV is emitted,
and an optional benchmark records real-time factor, first-packet latency, clipping,
duration, speaker similarity, and content errors when evaluators are installed.

## Extension pattern

A new backend implements the small protocol in the target submodule and is registered
with `BackendDescriptor`. Heavy dependencies must be imported inside `availability()`
or `run()`, never at module import. Repositories with incompatible environments should
use the versioned external contracts in `BACKEND_CONTRACTS.md`.

## Artifact policy

Each run directory is immutable and receives a unique ID. A successful manifest has
`status=completed`; an exception writes `status=failed` before re-raising. Input hashes
allow reproducing or comparing runs without embedding private input data in reports.
