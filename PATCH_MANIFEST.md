# ZynNova Free-form Unstructured Mesh Patch

This patch extends ZynMorph beyond rectangular voxel envelopes.

## New files

- `src/zynnova/zynmorph/freeform.py`
  - arbitrary closed triangular shells
  - multiple material inclusions / nested domains / true holes
  - shell orientation and topology gates
  - TetGen free-form PLC execution
  - existing Tet4 -> interface PLC diagnostics
  - automatic seeds for disconnected Tet4 material components
- `src/zynnova/zynmorph/reference_mesh.py`
  - COMSOL Tet4 MPHTXT reader
  - global/per-domain edge and tetra-volume profiles
  - TetGen sizing transfer from a reference mesh
- `tests/zynnova/test_zynmorph_freeform.py`
- `docs/FREEFORM_UNSTRUCTURED_MESHING.md`
- `notebooks/ZynNova_Freeform_Unstructured_TetGen_Test.ipynb`
- `validation/reference_mesh_profile_TetMesh-cell_nmc_grp.json`

## Modified files

- `src/zynnova/zynmorph/meshing.py`
  - `mesh_freeform_geometry(...)`
  - `mesh_freeform_like_reference(...)`
- `src/zynnova/zynmorph/__init__.py`
  - exports the new public API
- `README.md`
  - documents the free-form production path

## Important

This patch assumes the previously integrated native TetGen 1.6 C++ extension is
present.  `method="tetgen"` never silently falls back to structured voxel tets.
