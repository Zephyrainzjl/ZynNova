# ZynNova framework

ZynNova is the top-level framework and Python package. All retained scientific modules
and the four difficult-task subframeworks live directly under the `zynnova` namespace;
there is no wrapper package or duplicated extension namespace.

## Major scientific modules

- `zynnova.structure`, `zynnova.data`, `zynnova.dft`, `zynnova.dynamics`
- `zynnova.ml`, `zynnova.physics`, `zynnova.visualization`
- `zynnova.zynsim` for battery, finite-element, multiphysics, inverse, multiscale,
  phase-field, digital-twin, and workflow capabilities

## Difficult-task subframeworks

- **ZynMorph**: conditional Li-ion battery microstructure generation and
  reconstruction, exact phase fractions, descriptors, topology constraints, voxel
  exchange, Tet4 meshes, and VTK/Gmsh/Abaqus export.
- **ZynVista**: metric image/video scene reconstruction, large 3-D world generation,
  point/mesh/3DGS preservation, style editing, COLMAP and DCC exports.
- **ZynForm**: image-to-object backends, PBR asset preservation, surface repair,
  physical scaling, and tetrahedral finite-element meshing.
- **ZynVox**: authorized voice conversion, controllable zero-shot speech synthesis,
  streaming, UI, comparison tools, provenance, and consent gates.

## Commands

```bash
python -m pip install -e ".[zynnova]"
zynnova status
python -m zynnova status
python -m pytest -q tests/zynnova
```

Large learned backends use separate environments. Follow
`environments/zynnova/README.md`, lock official sources with
`scripts/zynnova/source_bootstrap.py`, and review `src/zynnova/SOURCE_LOCK.json` before
installing any external repository or model weight.

Validation scope and machine-readable results are documented under `validation/` and
`docs/ZYNNOVA_VALIDATION.md`.
