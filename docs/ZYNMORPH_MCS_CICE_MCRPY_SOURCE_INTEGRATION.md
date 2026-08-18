# ZynMorph source-level microstructure integration audit

## Scope

This integration makes ZynMorph the single public API for two previously
separate classes of workflow:

1. stochastic battery-electrode synthesis and particle/CBD statistics;
2. descriptor-space microstructure characterization and reconstruction.

The upstream projects were audited as algorithmic references. ZynNova does not
create nested `mcs_cice` or `mcrpy` public namespaces.

## MCS-CICE ElectrodeGenerationAlgorithm mapping

| Audited upstream behavior | ZynNova API |
|---|---|
| spheres / ellipsoids / mixed | `ParticleDistribution.geometry` |
| RSA | `PackingSettings(method="rsa")` |
| gravity packing | `PackingSettings(method="gravity")` |
| pseudo gravity | `PackingSettings(method="pseudo_gravity")` |
| pseudo electrostatic | `PackingSettings(method="electrostatic")` |
| contained / extended / periodic | `PackingSettings.boundary_mode` |
| overlap | `PackingSettings.overlap_fraction` |
| PSD / SEM-lognormal validation | `validate_particle_size_distribution` |
| angle tolerance / ranges | `ParticleDistribution.angle_*` |
| individual particle labels | `ElectrodeSynthesisConfig.individual_particle_labels` |
| bridge CBD | `CBDSettings(method="bridge")` |
| Mistry-style CBD | `CBDSettings(method="mistry")` |
| random CBD | `CBDSettings(method="random")` |
| blob CBD | `CBDSettings(method="blob")` |
| mixed CBD | `CBDSettings(method="mixed")` |
| CBD nanoporosity | `CBDSettings.nanoporosity` |
| variable voxel size | `ElectrodeSynthesisConfig.voxel_size_m` |
| packing guard / crop | `padding_voxels`, `crop_after_generation` |
| content padding | `pad_structure_to_content` |
| cutting | `cut_electrode_empty_tail` |
| X channel | `ChannelSettings` |
| AM/binder/carbon composition | `ElectrodeComposition` |
| porosity/loading/capacity | `ElectrodeStatistics` |
| HDF5 / VTK | `ElectrodeSynthesisResult.export` |
| Pickle persistence | replaced by safe NPZ/NPY |

The audited repository did not expose a LICENSE file in its root at audit time.
For that reason no upstream Python source is vendored into ZynNova; this module
is a clean-room implementation of the public/source-audited behavior.

## MCRpy mapping

### Descriptors

ZynNova registers the full audited plugin surface:

`VolumeFractions`, `Variation`, `FFTCorrelations`, `TwoPointCorrelations`,
`Correlations`, `FFTCrossCorrelations`, `CrossCorrelations`,
`LineCorrelations`, `LinealPath`, `LinealPathApproximation`,
`LineLinealPathApproximation`, `GramMatrices`, `MultiPhaseGramMatrices`, and
`OrientationDescriptor`.

The Gram plugins deliberately use ZynNova's differentiable multi-resolution
local/gradient/Laplacian feature bank instead of requiring the legacy bundled
VGG pickle. This keeps the descriptor differentiable, works natively in 2D and
3D, and removes unsafe pickle loading.

### Losses

`MSE`, `SSE`, `RMS`, `L1`, and `L2` are registered plugins.

### Optimizers

`Adam`, `Adamax`, `Nadam`, `RMSprop`, `SGD`, `Adagrad`, `Adadelta`, `LBFGSB`,
`TNC`, `SimulatedAnnealing` / Yeong-Torquato, and `YTPost` are registered.

### Workflows

| Workflow | ZynNova API |
|---|---|
| characterize | `characterize` |
| reconstruct | `reconstruct` |
| match | `match` |
| merge descriptor values | `merge` |
| 2D→3D directional merge | `merge_directional` |
| interpolate | `interpolate` |
| smooth | `smooth_microstructure` |
| view | `view_microstructure`, `view_characterization` |
| save/load | safe NPZ/NPY/HDF5/TIFF/VTK APIs |

For 3D characterization, `slice_mode="average"`, `"sample"`, and
`"sample_surface"` preserve the source framework's orthogonal-slice behavior.
`isotropic=True` averages the three directional descriptors; otherwise the x/y/z
descriptors remain separate.

Multigrid reconstruction is genuinely coarse-to-fine: each coarse result is
nearest-neighbor expanded and becomes the next level's phase-preserving
initialization.

## Improvements rather than duplicated weak paths

Where ZynMorph already had the stronger production implementation, the upstream
concept is routed into that implementation instead of adding a second branch:

- exact multi-phase sum-to-one uses softmax rather than a penalty;
- all reconstructed or generated voxel fields can enter the same
  `mesh_complex_regions` free-form PLC→TetGen path;
- material tracking labels can be collapsed into final FEM material IDs before
  meshing;
- final Tet4 meshes use the same COMSOL reversible region/entity mapping;
- unsafe Pickle persistence is not used for ZynNova-owned outputs.


### Periodic translation

`translate_microstructure(...)` provides the periodic shift/translation operation used by MCR-style workflows. It preserves phase counts exactly and supports 2-D and 3-D arrays. The audited upstream `Symmetry` module is only a placeholder and therefore has no algorithmic behavior to port.
