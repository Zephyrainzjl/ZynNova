# Validation summary

## Test suite

- `PYTHONPATH=src pytest -q tests/zynnova`: **115 passed**
- `python -m compileall -q src/zynnova`: **passed**
- top-level unified API import: **passed**
- clean notebook JSON: **valid, outputs cleared**
- notebook execution with native TetGen disabled in the artifact runtime: **passed**

## COMSOL reversible mapping observed in notebook

```text
Writer region -> COMSOL entity: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
Original regions: [0 1 2 3 4]
Read-back regions: [0 1 2 3 4]
restored_original_regions: True
COMSOL reversible region mapping: PASS
```

## Integrated electrode generation smoke

The integration notebook exercised mixed sphere/ellipsoid particles, padded
packing, CBD, nanoporosity, a transport channel, per-particle tracking, PSD
validation and HDF5/VTK/NPZ outputs.

## Descriptor/reconstruction smoke

The notebook exercised:

- registered descriptor/loss/optimizer discovery;
- 3-D directional characterization;
- multigrid descriptors;
- binary single-phase mode;
- one/two/three-source directional merge;
- coarse-to-fine 2-D -> 3-D multigrid reconstruction;
- interpolation and match;
- periodic microstructure translation.

Every reconstructed probability field uses softmax and therefore satisfies
`sum_phase probabilities == 1` numerically rather than via a phase-sum penalty.
