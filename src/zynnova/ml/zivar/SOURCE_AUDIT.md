# ZIVAR 0.2 source and design provenance

This audit applies to architecture `zivar-variational-electrospin-2` and
numerics `variational-scf-pme.1`. ZIVAR's implementation is independent: no
source code, trained weights, or datasets from the projects below are copied
or redistributed in this directory. A pinned project can be a compatibility
dependency or a design/validation reference; it is not evidence that ZIVAR
contains that project's algorithm or has reached the same maturity.

The machine-readable identities, tags, commits, licenses, and upstream/local
algorithm locations are in `SOURCE_LOCK.json`.

## Implemented architecture and provenance boundary

The single default path is:

```text
structure/cell/charge/external axial spin
  -> registered O(3) backbone
  -> local coefficients
  -> constrained variational x=(q,p,Q_STF,m) functional
  -> fail-closed matrix-free SCF
  -> one scalar short-range + electrostatic + polar + magnetic energy
  -> force, stress, effective field, and torque derivatives
```

The local implementation is divided as follows:

- `functional.py`, `types.py`, and `variational.py` define the ZIVAR state,
  scalar functional, constraints, and production coupling. These are original
  ZynNova implementations; they are not MACE-POLAR, NequIP, or Allegro code.
- `operators.py`, `scf.py`, and `implicit.py` implement projected matrix-free
  PCG, preconditioning, fail-closed convergence reporting, and the adjoint/KKT
  differentiation checks. They do not call an upstream QEq solver.
- `ewald_reference.py`, `mesh.py`, and `pme.py` implement ZIVAR's point-charge
  direct Ewald reference and B-spline/FFT monopole PME. `torch-pme` is used only
  as an official comparison point; it is neither imported nor vendored.
- `cpp/include/zynnova/zivar/matrix_free.hpp` and
  `cpp/src/zivar/matrix_free.cpp` implement an optional Kokkos matrix-free
  sparse operator, reductions, constraint primitives, Jacobi preconditioning,
  and PCG. This target uses Kokkos APIs but contains no LAMMPS implementation.
  The audited CPU build used official Kokkos tag `4.3.01`
  (`6ecdf605e0f7639adec599d25cf0e206d7b8f9f5`); consumer builds still resolve
  Kokkos externally and must record their actual version.
- `lammps.py` implements a reference/development `fix external` callback and
  binary-bound export manifest. It does not embed LAMMPS source and must not be
  described as a native pair style.

Legacy fixed-depth `polar`, dense `qeq`, `direct`, and `fukui_auxiliary`
configurations remain explicit compatibility choices. They are not constructed
in parallel with the default variational core and are not implicit fallbacks.

## Frozen official references

| Project | Frozen identity | License | Consulted algorithm/API location | Relationship to ZIVAR |
|---|---|---|---|---|
| MACE | `v0.3.16`, `4d2da09413ac1407...` | MIT | `mace/modules/models.py`, `blocks.py`, `extensions.py::PolarMACE` | Optional public-API backbone and a design comparison; no MACE weights or source are shipped. |
| e3nn | `0.4.4`, `5c0d2fdcc719190e...` | MIT | `e3nn.o3.Irreps`, `TensorProduct`, `spherical_harmonics` | Direct pinned dependency for O(3) representation operations. |
| torch-pme | `v0.4.0`, `bc655e4d00579523...` | BSD-3-Clause | `EwaldCalculator`, `PMECalculator`, `P3MCalculator` | Reference-only error/decomposition comparison; ZIVAR implements its own Ewald/PME code. |
| NequIP | `v0.7.0`, `b513fdf19f4f7665...` | MIT | `nequip.model`, `nequip.nn`, `AtomicDataDict` | Reference-only backbone protocol/deployment comparison. |
| Allegro | `v0.4.0`, `0ffb66c8de751a50...` | MIT | `AllegroModel`, `Allegro_Module` | Reference-only scalable local-backbone comparison. |
| LAMMPS | `patch_30Mar2026`, `fc6a61720c426044...` | GPL-2.0 | `fix_external.cpp`, `src/ML-IAP`, `src/QEQ`, `src/KOKKOS` | Official integration-contract reference; no LAMMPS code is vendored. |

Official repositories:

- MACE: https://github.com/ACEsuit/mace
- e3nn: https://github.com/e3nn/e3nn
- torch-pme: https://github.com/lab-cosmo/torch-pme
- NequIP: https://github.com/mir-group/nequip
- Allegro: https://github.com/mir-group/allegro
- LAMMPS: https://github.com/lammps/lammps

MACE and e3nn are the only projects in this table that are runtime
compatibility dependencies of the standard MACE-backed ZIVAR installation.
The other entries are reference-only unless a user independently installs or
builds them for comparison. In particular, `torch-pme` is not silently used as
a fallback.

## What the current implementation establishes

- A constrained total charge is enforced by the default SCF path, with
  observable absolute/relative residuals, iteration count, termination reason,
  and hard errors for non-convergence, non-finite values, or non-positive
  curvature.
- Energy, force, stress, spin effective field, and torque derive from the same
  scalar functional. State-observable gradients use a matrix-free adjoint; the
  dense KKT implementation is a small-system validation oracle, not the
  production solver.
- The direct float64 reference supports triclinic three-dimensional periodic
  cells and includes real-space, reciprocal-space, self, and neutralizing
  background terms under conducting boundary conditions. The production PME
  path uses a three-dimensional `torch.fft` mesh and configurable error target.
- External non-collinear spins and induced moments are axial, time-odd vectors;
  the functional tests proper/improper O(3) behavior and joint time reversal.
- The optional C++ target launches real Kokkos kernels for its numerical
  primitives. This fact establishes only those primitives, not a full GPU
  ZIVAR or LAMMPS implementation.

These statements still require the numerical and deployment gates in
`VALIDATION.md` for any particular release artifact, device, dtype, LAMMPS
binary, or trained checkpoint.

## Explicit non-claims and remaining blockers

- PME currently accelerates monopole charge only. Long-range dipole and
  quadrupole mesh operators are not implemented. Partial periodicity is
  rejected, and open-boundary electrostatics has no FMM accelerator.
- The Kokkos target is a matrix-free numerical core. It has no ZIVAR backbone,
  PME mesh, LAMMPS neighbor-list contract, halo exchange, pair/fix registration,
  restart state, domain decomposition, or multi-GPU communication.
- The Python LAMMPS bridge globally gathers atoms and constructs the graph on
  the CPU. It is a correctness/reference callback, not native Kokkos and not a
  production domain-decomposed deployment.
- MLIAP export remains local-backbone-only and excludes the global SCF/PME and
  induced `q/p/Q/m` state.
- Interface existence, a skipped test, a Python fallback, or a compiled kernel
  is not recorded as a passed maturity gate. Release evidence must be generated
  by the fixed gate runner and bound to the relevant source and executable.

The configured local executable
`/home/zephyrain/software/lammps-mliap-gpu-nompi/bin/lmp` reports a development
build (`develop / patch_30Mar2026-1211-g5af1318e9e`) with MPI stubs and Kokkos
CUDA/OpenMP. That observation is deliberately separate from the frozen
`patch_30Mar2026`/`fc6a61720c42604466e626763af66feefde23646` reference. Its
reported packages include KOKKOS, ML-IAP, ML-PACE, ML-SNAP, and PYTHON but not
SPIN, so it cannot certify native coupled
spin-lattice deployment. An exported callback bundle additionally binds the
exact executable by SHA-256; neither the path nor a matching version string is
enough by itself. On this audited host a non-Kokkos `run 0` succeeds, while the
documented `-k on g 1 -sf kk` launch fails before model evaluation with
`cudaErrorInsufficientDriver`; no GPU LAMMPS result is therefore claimed.

## License and citation rule

Use the upstream license and citation associated with every installed
dependency, external executable, model weight, and dataset. `SOURCE_LOCK.json`
records source provenance, not permission to redistribute those artifacts.
Any future source-derived implementation must receive a new license review and
must update this audit before merging.
