# Data module integration notes

## Release

- Project version: `0.4.0`
- New package: `src/zynnova/data`
- Existing structure, dynamics, and visualization namespaces were retained.

## Implemented subsystems

- Unified `MaterialSample`
- Arbitrary `FieldSpec` / `TaskSpec`
- Prediction, generation, representation, and neural-potential task compilation
- Crystal, molecule, polymer, and special dataset namespaces
- Dataset plugin registry and catalog
- Existing `zynnova.structure` graph/polymer integration
- PyG, dense graph, Transformer, and potential encoders
- Map-style and streaming PyTorch datasets
- Variable-atom neural-potential collation
- Train/validation/test data module
- Random, grouped, and molecular-scaffold splits
- Dataset transforms and field statistics
- Dataset validation reports
- Directory, JSONL, CSV, HDF5, and NPZ storage
- Resumable/checksummed/safe dataset downloader

## Built-in plugins

Crystal:
- Materials Project
- Matbench
- JARVIS-DFT
- NOMAD archive API
- local crystal files

Molecular:
- QM9
- PCQM4Mv2
- revised MD17
- ANI-1x
- local molecule files

Polymer:
- TransPolymer
- configurable PSMILES table
- local `.zpoly` / JSON records

Special:
- OC20 LMDB
- ASE trajectory potential data
- configurable tabular data
- arbitrary record converter

## Verification in the delivery environment

```text
Python compileall: passed
Full pytest suite: 31 passed, 2 skipped
Data-specific tests: 17 passed
```

Skipped tests require optional `torch-geometric` or a prebuilt native C++
extension. The environment did not provide `scikit-build-core` and `pybind11`
through its package index, so a fresh wheel build could not be repeated there.
The existing CMake source was not changed by the data module.

Network-backed plugins are implemented against their official APIs/files but
were not used to download multi-gigabyte datasets during validation. Their
parsers are dependency-gated and can be tested locally with small limits or
existing raw caches.
