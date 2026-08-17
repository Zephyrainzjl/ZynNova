# Visualization integration notes

## New package tree

```text
src/zynnova/visualization/
├── __init__.py
├── results/
│   └── __init__.py
└── structure/
    ├── __init__.py
    ├── api.py
    ├── adapters.py
    ├── backend.py
    ├── formats.py
    ├── molecule.py
    ├── polymer.py
    ├── crystal.py
    └── types.py
```

`results` is intentionally empty apart from a namespace placeholder.

## Public API

```python
from zynnova.visualization import visualize
from zynnova.visualization.structure import (
    visualize_structure,
    visualize_molecule,
    visualize_polymer,
    visualize_crystal,
)
```

## Backend policy

- `py3dmol`: default for static structures, unit cells, supercells, atom labels,
  and repeat-unit/coarse-grained polymer diagrams.
- `nglview`: optional trajectory backend when installed.
- Optional dependencies are imported lazily, so importing `zynnova` does not
  require a notebook visualization package.

## Validated inputs

- `StructureData`
- ASE `Atoms`
- supported structure files through ASE
- RDKit `Mol` with a 3D conformer
- `PolymerRecord`
- `MolecularGraph`
- `.zpoly` files
- sequences of atomistic structures for trajectories

## Validation

- Full tests: 16 passed, 2 skipped in the source environment.
- Ruff checks: passed.
- Wheel build: passed, including the C++ extension and visualization package.
- Installed-wheel import and py3Dmol rendering command generation: passed.
