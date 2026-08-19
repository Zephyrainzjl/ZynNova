# ZIVAR validation and release gates

Source completeness and production maturity are separate. A passing unit suite
does not imply a validated CUDA, Kokkos, MPI, or LAMMPS deployment.

## Core numerical gates

The release registry in `maturity.py` requires source-hash-bound JSON
artifacts. A bare `{"status":"pass"}` file is invalid. Each artifact records
the fixed runner command and measured metrics.

| capability | required threshold |
|---|---:|
| charge constraint | max residual `1e-10 e` |
| SCF stationarity | projected gradient RMS `1e-9`, energy error `1e-12 eV` |
| matrix-free QEq vs dense KKT | max charge error `1e-9 e` |
| tight direct Ewald | energy error `1e-8 eV` |
| PME vs tight Ewald | energy `1e-5 eV`, force `1e-4 eV/Å` |
| force finite difference | max error `1e-5 eV/Å` |
| strain/stress finite difference | max error `1e-6 eV/Å³` |
| spin-field finite difference | max error `1e-6 eV/μB` |
| O(3), including inversion | max equivariance error `1e-9` in float64 |
| zero-field time reversal | max energy error `1e-9 eV` |
| batch vs single | max float64 error `1e-9` |
| exact CPU resume | next-step parameter error `0` |

Thresholds reflect float64 analytic/central-difference checks on small,
well-conditioned systems. Float32 CPU/GPU parity is separately capped at
`1e-4 eV/Å` for forces and must record model/structure scale. Tests must also
show error reduction when finite-difference step, Ewald cutoff, or PME mesh is
tightened; a single coincidental point is insufficient.

SCF convergence requires the projected KKT residual, every equality
constraint, and a preconditioned directional energy-error estimate below
`energy_atol_eV_per_atom * N`. `SCFReport.energy_change` separately records the
actual last Krylov-step energy change; it is not confused with the stationary
energy-error estimate.
Max-iteration, negative-curvature, non-finite, and inconsistent-constraint
cases must raise and must not return a last iterate.

## Deployment gates

A production deployment additionally requires:

- exact Python/LAMMPS energy, force, virial, charge, and induced-moment parity;
- run-0, neighbour rebuild, 10--100 step MD, and restart continuity;
- serial and the actual supported MPI/domain-decomposition configurations;
- CPU/GPU comparison on the same checkpoint and neighbour list;
- size sweeps reporting wall time, peak host memory, peak device memory, and
  fitted complexity exponents;
- multi-GPU halo/domain tests when that capability is claimed.

The present Python callback and local-backbone MLIAP artifact cannot satisfy
the native full-model gate. The optional Kokkos matrix-free core proves only
its numerical primitives until a real LAMMPS pair/fix integration exists.

## Checkpoints, data, and training

Release evidence binds the source hash, checkpoint SHA256, dataset and split
hashes, preprocessing/unit schema, Conda software record, GPU/driver, and exact
LAMMPS binary/build packages. Checkpoints save model, optimizer, scheduler,
scaler, counters, and RNG state. An interrupted deterministic CPU run must
produce bit-identical parameters on the next step. Exact resume also binds
model weights, trainer/loss configuration, optimizer class and named parameter
group order, and scheduler/optimizer identity. Callable schedulers such as
`LambdaLR` are rejected because their behaviour is not recoverable from a
PyTorch `state_dict`; only the audited serializable scheduler whitelist is
accepted.

Energy losses must state atom/graph normalization; sparse labels use explicit
masks, and an all-false mask is an error. Charge labels, formal oxidation
states, external spins, induced moments, and effective fields have distinct
semantics and provenance. No target may seed an SCF condition.

Use only the fixed runner:

```bash
conda run -n zynnova zivar audit-release --template evidence.json
conda run -n zynnova zivar run-gate unit_cpu_float64 --artifact artifacts/unit.json
conda run -n zynnova zivar audit-release --evidence evidence.json
```

Unavailable hardware gates fail explicitly. They are never converted to pass
by a skip, TODO, interface presence, or Python fallback.

## Current host observations (not release evidence)

On the 2026-08-17 audited RTX 5080 / driver 580.88 / PyTorch 2.13.0+cu130 host,
the
float64 complete variational model matched CPU at `1.11e-16 eV` in energy,
`1.73e-18 eV/Å` in force, and `6.94e-18 e` in charge for the deterministic
three-atom diagnostic. A 64³, order-six PME calculation matched CPU in float64
to `1.78e-14 eV` and `6.57e-13 eV/Å`. The float32 PME force discrepancy was
`2.97e-4 eV/Å`, which **fails** the registered `1e-4 eV/Å` parity threshold;
the CUDA float32 gate therefore remains unresolved.

The configured LAMMPS development binary passes a non-Kokkos `run 0`, but its
requested `-k on g 1 -sf kk` launch currently stops in Kokkos initialization
with `cudaErrorInsufficientDriver`. These manual observations are diagnostics,
not hash-bound passing gate artifacts, and do not change `production_ready`.
