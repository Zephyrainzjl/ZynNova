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
