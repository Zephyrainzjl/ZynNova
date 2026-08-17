# ZIVAR variational electro-spin architecture

## One production path

The default architecture revision is `zivar-variational-electrospin-2`:

```text
structure, cell, total charge, external axial spins S
  -> one neighbour graph and optional reciprocal mesh
  -> registered O(3) backbone
  -> local functional coefficients
  -> constrained minimisation of x=(q,p,Q_STF,m)
  -> short + Coulomb + polar + quadrupolar + induced-magnetic + E(R,S)
  -> one scalar total energy
  -> force, stress, effective field and torque derivatives
```

`model.energy_forces_stress` never calls an independent force or stress head.
The external non-collinear spin `S` and induced moment `m` are different
objects: `S` is an axial, time-odd dynamical condition; `m` is an axial,
time-odd variational response.

The old fixed-depth `polar`, dense `qeq`, `direct`, and `fukui_auxiliary`
implementations remain explicit legacy configurations. The default model does
not instantiate them and there is no implicit fallback.

## Functional and SCF

For fixed geometry and `S`, `functional.py` evaluates

\[
E(x)=E_{short}(R)+E_{spin}(R,S)+l(R,S)^T x
     +\tfrac12 x^T[D(R)+K_{Coulomb}(R)]x+E_{external}.
\]

`D` contains positive charge hardness, inverse polarizability, inverse
quadrupole polarizability, and inverse magnetic susceptibility. `K` is
matrix-free. The total Hessian—not the self-subtracted Coulomb block in
isolation—must be positive definite on the equality-constrained subspace.
Open-boundary charge transfer uses blocked pair matvecs and an atomic self
hardness; fully periodic charge transfer uses Ewald real space plus a genuine
3-D FFT PME reciprocal term. Partial PBC is rejected.

`scf.py` uses projected preconditioned CG with exact affine feasibility,
`atol`, `rtol`, a directional stationary-energy tolerance, `max_iter`,
warm-start input, negative-curvature detection, and an observable `SCFReport`.
Non-convergence, non-finite values, curvature breakdown, a violated charge
constraint, or an excessive energy-error estimate raises
`SCFConvergenceError`; the last iterate is never returned as a prediction.

Energy, force, and stress use the converged-state envelope theorem. Supervised
state observables use an adjoint matrix-free solve, so charge/moment losses
differentiate through the stationary solution without copying labels into
conditions. `implicit.py` supplies a dense KKT oracle used by gradcheck tests.

Current limitation: PME accelerates monopole charge only. Long-range dipole and
quadrupole mesh operators, open-boundary FMM, multi-GPU mesh decomposition,
and domain decomposition remain release blockers rather than hidden fallbacks.

## O(3), inversion, and time reversal

`p` is a polar vector, `Q` a symmetric-traceless polar rank-two tensor, and
`S,m` axial vectors. The scalar functional is invariant under proper and
improper O(3) transforms. In zero magnetic field it is invariant under joint
time reversal `(S,m)->(-S,-m)`. The spin Hamiltonian adds exchange,
biquadratic exchange, SOC-gated anisotropy and DMI, higher-order invariant
couplings, Landau energy, and Zeeman energy. Effective fields and torques are
derivatives of the same scalar.

## Typed public contract

`types.py` defines `ZIVARBatch`, `Conditions`, `Targets`, `ElectronicState`,
`EnergyBreakdown`, and `ZIVARPrediction`. `ElectronicState.pack()` maps each
atom to twelve independent values in the order `q(1),p(3),Q_STF(5),m(3)`.
ASE, training, and new deployment integrations adapt to this contract; legacy
dictionary inputs remain a storage-compatible surface.

The backbone protocol/registry remains sealed and replaceable. A checkpoint
records the adapter fingerprint, architecture/numerics revisions, tensor
dtypes, RNG state, and—when saved from a trainer—the optimizer, scheduler,
scaler, and counters. Version-0.1 fixed-depth electronic
weights are not physically compatible: only matching backbone weights may be
migrated, followed by retraining of the variational core.

## Deployment truth table

- PyTorch CPU/CUDA: the complete Python model and PME path run on the selected
  tensor device; ASE graph construction is currently CPU-side.
- Kokkos: `ZynNova::zivar_kokkos_core` contains real matrix-free matvec,
  reductions, Jacobi, constraint primitives, and PCG kernels behind an
  opt-in CMake flag. It is a numerical foundation, not a complete ZIVAR pair
  style.
- LAMMPS full physics: the existing `fix external` Python callback is a
  correctness/reference bridge with global gather and CPU graph construction.
  It is not native Kokkos, not domain decomposed, and not multi-GPU production.
- MLIAP export: explicitly local-backbone-only; it excludes SCF, PME, induced
  moments, and the global energy functional.

No release may call the Python callback or the numerical Kokkos primitive a
native production pair style. A real LAMMPS/Kokkos integration still needs
pair/fix sources, neighbour/domain contracts, halo communication, restart
state, CPU/GPU parity, and serial/MPI execution gates.
