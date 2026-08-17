# ZIVAR 0.2 variational electro-spin potential

Architecture: `zivar-variational-electrospin-2`

Numerics: `variational-scf-pme.1`

ZIVAR now has one default physical core: a replaceable O(3) backbone
parameterises a constrained `q/p/Q/m` energy functional, a fail-closed SCF
solve finds its stationary state, and one scalar total energy supplies forces,
stress, spin effective fields, and torque.

```python
from zynnova.ml.zivar import ZIVARConfig, build_zivar, zivar_calculator

config = ZIVARConfig.production(
    dft_level="PBE+U+SOC",
    backbone__atomic_numbers=(3, 8, 26),
    scf__atol=1e-10,
    scf__rtol=1e-8,
    electrostatics__error_target=1e-6,
)
model = build_zivar(config, device="cuda").double()

# A non-collinear production evaluation requires explicit axial spins [N,3].
atoms.arrays["spin_vectors"] = spin_vectors_muB
atoms.info["total_charge"] = 0.0
atoms.calc = zivar_calculator(model, device="cuda", dtype="float64")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The default route is `ElectronicConfig(method="variational")`. The former
fixed-depth `polar`, dense `qeq`, direct, Fukui, scalar-moment, and formal
oxidation heads remain opt-in compatibility or auxiliary routes. They are not
constructed beside the variational model and are not fallback paths.

Implemented and directly tested:

- typed `q`, polar `p`, STF `Q`, and induced axial `m` state;
- exact graph total-charge constraints;
- matrix-free projected PCG, onsite preconditioning, warm starts, convergence
  reports, negative-curvature detection, and fail-closed errors;
- envelope force/stress derivatives and adjoint implicit state gradients;
- triclinic direct Ewald reference with real/reciprocal/self/background terms;
- B-spline charge assignment, 3-D `torch.fft` PME, window deconvolution, and
  error-target mesh planning;
- non-collinear external spins, inversion/time-reversal rules, exchange,
  anisotropy/DMI/SOC conditions, effective fields, and torque;
- complete trainer/optimizer/scaler/counter/RNG checkpoint resume;
- a real optional Kokkos matrix-free numerical core.

Not yet production-complete:

- PME dipoles/quadrupoles and isolated FMM;
- cached/batched ASE neighbour construction;
- a full native LAMMPS pair/fix, Kokkos neighbour kernels, restart state,
  domain decomposition, and multi-GPU communication;
- hardware-bound CUDA/LAMMPS/long-trajectory benchmark evidence.

The current full-physics LAMMPS `fix external` callback is a reference bridge,
and MLIAP export contains the local backbone only. Neither is labelled a native
full-ZIVAR deployment.

Run the local suite and generate a fail-closed evidence template with:

```bash
conda run -n zynnova pytest -q src/zynnova/ml/zivar/tests
conda run -n zynnova zivar audit-release --template evidence.json
```

See `ARCHITECTURE.md` for equations and deployment boundaries and
`VALIDATION.md` for numerical thresholds. `production_ready` remains false
until every registered target gate has a source-bound artifact produced by the
fixed gate runner.
