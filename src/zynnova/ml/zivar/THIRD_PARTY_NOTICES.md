# Third-party notices

ZIVAR 0.2 is an independent ZynNova implementation. This directory does not
vendor source code, trained weights, or datasets from the projects listed
below. Installation, linking, execution, or redistribution of an external
component remains subject to that component's upstream license and notices.
The pinned identities and algorithm/API locations are recorded in
`SOURCE_LOCK.json`.

## Runtime and build integrations

- **MACE 0.3.16** — MIT License,
  https://github.com/ACEsuit/mace. ZIVAR can use MACE through its public Python
  API as a replaceable local equivariant backbone. MACE source and weights are
  not redistributed here.
- **e3nn 0.4.4** — MIT License,
  https://github.com/e3nn/e3nn. ZIVAR uses its installed public O(3) APIs;
  e3nn source is not redistributed here.
- **Kokkos 4.3.1** (validated CPU build; consumer resolution remains external)
  — BSD-3-Clause, externally supplied build dependency,
  https://github.com/kokkos/kokkos. The optional
  `ZynNova::zivar_kokkos_core` target uses public Kokkos APIs. No Kokkos source
  is vendored, and this numerical target is not a LAMMPS pair style.
- **LAMMPS patch_30Mar2026 reference** — GPL-2.0,
  https://github.com/lammps/lammps. ZIVAR's development bridge uses the public
  `fix external` callback contract and can interrogate/hash a separately
  installed executable. No LAMMPS source is included in ZIVAR. Distributing a
  LAMMPS build or a future linked/native integration requires an independent
  GPL compliance review and preservation of LAMMPS's notices.

PyTorch, ASE, NumPy, mpi4py, cuEquivariance, and OpenEquivariance may also be
installed by selected ZynNova extras. They are external packages, are not
vendored in this directory, and retain their upstream licenses. Optional
accelerators do not change ZIVAR's checkpoint architecture or prove numerical
parity without the registered gates.

## Reference-only projects

- **torch-pme 0.4.0** — BSD-3-Clause,
  https://github.com/lab-cosmo/torch-pme. Consulted as an official comparison
  for differentiable Ewald/PME/P3M interfaces. ZIVAR does not import it and its
  Ewald/PME implementation was written independently.
- **NequIP 0.7.0** — MIT License,
  https://github.com/mir-group/nequip. Consulted for equivariant model protocol
  and deployment comparisons; no source or weights are copied.
- **Allegro 0.4.0** — MIT License,
  https://github.com/mir-group/allegro. Consulted for scalable equivariant
  local-backbone comparison; no source or weights are copied.

Reference-only status means these packages are not implementation backends or
fallbacks for the ZIVAR 0.2 variational SCF/PME path. Similar behavior or use of
published equations does not imply source reuse.

## Models and data

No third-party checkpoint or dataset is distributed by these ZIVAR sources.
Users must separately verify the license, citation, provenance, and permitted
use of every model or dataset they supply. Benchmark-only packages and weights,
including CHGNet when independently installed, retain their own terms.

See `SOURCE_AUDIT.md` for the capability boundary and the distinction between
implemented code, reference comparisons, and unresolved production blockers.
