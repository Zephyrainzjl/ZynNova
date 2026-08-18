# Third-party notices

## TetGen 1.6.0

- Upstream algorithm: TetGen by Hang Si / Weierstrass Institute.
- Pinned source mirror: `pyvista/tetgen`, revision
  `c039698cf4cce5c671b281c003dbc6cd8e58acc3`.
- Upstream files: `tetgen.cxx`, `tetgen.h`, `predicates.cxx`, `tetgen-license`.
- License: **GNU Affero General Public License, version 3 or later (AGPL-3.0-or-later)**.
- ZynNova pybind11 adapter: MIT.

A wheel compiled with `_zynmorph_tetgen_native` links to and redistributes
TetGen.  The resulting covered distribution must comply with TetGen's AGPL
terms.  Build with `-DZYNNOVA_BUILD_TETGEN=OFF` when that licensing boundary is
not acceptable.  This notice is informational and not legal advice.

## MCRpy source-audited integration

- Upstream: `NEFM-TUDresden/MCRpy`.
- License: Apache License 2.0.
- ZynNova uses a source-audited native NumPy/PyTorch/SciPy implementation of the
  public descriptor/loss/optimizer/workflow surface. No MCRpy package is
  installed at runtime and no upstream pickle weight file is redistributed.

## MCS-CICE ElectrodeGenerationAlgorithm source audit

- Upstream: `mcs-cice/ElectrodeGenerationAlgorithm`.
- At the 2026-08-18 audit, no LICENSE file was visible in the repository root.
- ZynNova therefore does not vendor or copy those Python files. The electrode
  generator is a clean-room implementation of the behavior and public/source
  interfaces observed in the repository.
