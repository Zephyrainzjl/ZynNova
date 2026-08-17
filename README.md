# ZynNova

<p align="center">
  <strong>An Extensible Framework for Materials Intelligence, Simulation, 3-D Reconstruction, and Generative Engineering</strong>
</p>


<p align="center">
  <img src="docs/assets/zynnova-overview.png" alt="ZynNova framework overview" width="900">
</p>

## Overview

**ZynNova** is a standalone Python and C++17 framework that retains the complete
materials-science stack while adding isolated workflows for difficult generative and
engineering tasks. Its single public namespace is `zynnova`.

The retained scientific stack covers structures, graphs, datasets, density-functional
workflows, molecular dynamics, materials machine learning, interatomic potentials,
visualization, battery multiphysics, finite elements, inverse problems, and digital-twin workflows. The new difficult-task stack is organized into four independent
subframeworks:

- **ZynMorph** — conditional Li-ion battery microstructure generation,
  reconstruction, descriptors, topology constraints, and Tet4 finite-element meshes.
- **ZynVista** — image/video-conditioned metric scene reconstruction, large-world
  generation, 3DGS/mesh preservation, style transfer, and DCC exports.
- **ZynForm** — high-fidelity image-to-object generation, surface repair, physical
  scaling, multi-format export, and tetrahedral FEM meshing.
- **ZynVox** — consent-aware voice conversion, zero-shot speech synthesis, streaming,
  evaluation, provenance, Python APIs, CLI, and optional Gradio UI.

Heavy public research repositories are kept in isolated environments behind audited
backend contracts, so adding one model does not destabilize unrelated modules.

## Package layout

```text
src/zynnova/
├── structure, data, dft, dynamics, ml, physics, visualization
├── zynsim/                 # retained battery, FEM, multiscale and digital-twin stack
├── core/                   # difficult-task backend registry and provenance
├── geometry/               # shared point/surface/volume geometry
├── zynmorph/               # battery microstructures and FEM meshes
├── zynvista/               # image/video to 3-D scenes
├── zynform/                # image to 3-D objects and FEM
└── zynvox/                 # voice conversion and speech synthesis
```

## Installation

ZynNova requires Python 3.10 or later.

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation from this source tree:

```bash
cd ZynNova
python -m pip install -e ".[dev]"
```

Install the portable difficult-task dependencies:

```bash
python -m pip install -e ".[zynnova]"
```

Large reconstruction, world-generation, object-generation, and voice models should
use the isolated environments documented under `environments/zynnova/`.

## Quick start

```python
import numpy as np

from zynnova import StructureData
from zynnova.structure.molecular import stru2graph

water = StructureData(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.000, 0.000, 0.000],
            [0.958, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]
    ),
)

graph = stru2graph(water, backend="python", neighbor_mode="radius")
print(graph.num_nodes, graph.num_edges)
```

Inspect difficult-task backends without loading model weights:

```bash
zynnova status
python -m zynnova status
```

Run deterministic validation:

```bash
PYTHONPATH=src python scripts/zynnova/verify.py
python scripts/zynnova/static_audit.py
```

## Documentation

- Main documentation: `docs/index.md`
- Standalone framework guide: `docs/ZYNNOVA.md`
- Installation and isolated environments: `docs/ZYNNOVA_INSTALLATION.md`
- Architecture and backend contracts: `src/zynnova/ARCHITECTURE.md` and
  `src/zynnova/BACKEND_CONTRACTS.md`
- Verified public-source inventory: `src/zynnova/SOURCE_LOCK.json`
- Examples: `examples/zynnova/`

## Extending the framework

New datasets, models, simulation backends, structure representations, C++ operators,
3-D generators, meshers, and speech engines can be added without modifying unrelated
subframeworks. Optional heavyweight dependencies must be discovered lazily and should
run through explicit in-process or isolated-process contracts.

## License

ZynNova was created by **Jialiu Zeng** and is distributed under the terms of the
[MIT License](LICENSE). Third-party repositories and model weights retain their own
licenses and are not redistributed by this source tree.
