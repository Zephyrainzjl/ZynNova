# ZynNova framework

ZynNova is the top-level Python package and namespace. Retained scientific modules
plus four difficult-task subframeworks are available directly under `zynnova`.

## Major scientific modules

- `zynnova.structure`, `zynnova.data`, `zynnova.dynamics`
- `zynnova.ml`, `zynnova.geometry`, `zynnova.core`
- `zynnova.zynsim` for finite-element, multiphysics, inverse, and workflow capabilities

## Difficult-task subframeworks

- **ZynMorph**: conditional Li-ion battery microstructure generation, reconstruction,
  exact phase-fraction control, topology constraints, Tet4 meshing, and export.
- **ZynVista**: image/video-conditioned scene reconstruction, large 3-D world generation,
  3DGS and mesh preservation, style editing, and DCC/COLMAP exports.
- **ZynForm**: image-to-object generation, physical scaling, surface repair,
  and tetrahedral FEM meshing.
- **ZynVox**: consent-aware voice conversion, zero-shot text-to-speech, streaming,
  evaluation, provenance, and optional Gradio UI.

## Common commands

```bash
python -m pip install -e ".[zynnova]"
zynnova status
python -m zynnova status
python -m pytest -q tests
```

Validation results and machine-readable artifacts are in `validation/`.
