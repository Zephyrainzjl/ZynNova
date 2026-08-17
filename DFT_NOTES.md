# ZynNova quantum mechanics, DFT, and AIMD

## Scope

`zynnova.dft` contains three deliberately separated capabilities:

1. A self-contained C++ numerical core for one-dimensional nonrelativistic
   quantum mechanics:
   - low-lying bound states of the time-independent Schrödinger equation;
   - reduced radial states for arbitrary central potentials and angular momentum;
   - real/complex wavefunctions and residual norms;
   - Crank–Nicolson propagation of the time-dependent Schrödinger equation.
2. A uniform electronic-structure interface:
   - a native ASE-compatible molecular PySCF calculator with analytic forces;
   - configured GPAW construction;
   - transparent construction of Quantum ESPRESSO, VASP, CP2K, ABINIT,
     FHI-aims, SIESTA, CASTEP, ELK, NWChem, ORCA, and Psi4 ASE calculators;
   - any already-created ASE calculator can be supplied directly.
3. Born–Oppenheimer ab-initio molecular dynamics:
   - C++ velocity-Verlet for NVE;
   - C++ BAOAB Langevin splitting for NVT;
   - exact restart, SCF error propagation, trajectory/thermodynamic output,
     fixed atoms, and numerical safety checks.

The one-dimensional solver is a conventional quantum-mechanics tool and
benchmark engine. It is not presented as a replacement for a many-electron DFT
code. Research DFT energies and forces are obtained from established engines.

## Installation

Build the native extension and install the common DFT workflow:

```bash
pip install -v ".[dft]"
```

For isolated molecular DFT with PySCF:

```bash
pip install -v ".[dft-pyscf]"
```

For GPAW:

```bash
pip install -v ".[dft-gpaw]"
```

GPAW setup data and executable-based backends still require their normal
system-level installation. Quantum ESPRESSO, VASP, CP2K, ABINIT, FHI-aims, and
similar engines also require a valid executable/profile and, where applicable,
pseudopotentials or a license.

## Conventional quantum mechanics

Atomic units use Bohr, Hartree, electron masses, and atomic time. The
`ev_angstrom` mode uses Å, eV, electron masses, and fs.

```python
import numpy as np

from zynnova.dft import harmonic, solve_schrodinger_1d

x = np.linspace(-8.0, 8.0, 1201)
solution = solve_schrodinger_1d(
    x,
    harmonic(x, force_constant=1.0),
    mass=1.0,
    num_states=6,
    units="atomic",
    backend="cpp",
)

print(solution.energies)
print(solution.residual_norms)
print(solution.norm(0))
```

The grid endpoints are Dirichlet boundaries. Enlarge the domain until boundary
amplitudes are negligible, then refine the grid until energies and observables
stop changing at the required precision.

Time propagation uses the norm-preserving Crank–Nicolson method:

```python
from zynnova.dft import propagate_wavefunction_1d

trajectory = propagate_wavefunction_1d(
    x,
    harmonic(x),
    solution.wavefunctions[0].astype(complex),
    timestep=0.01,
    steps=1000,
    save_every=10,
    units="atomic",
    backend="cpp",
)
```

Central-potential problems use the same C++ eigensolver:

```python
from zynnova.dft import solve_radial_schrodinger

radius = np.linspace(0.0, 40.0, 3001)
coulomb = np.zeros_like(radius)
coulomb[1:] = -1.0 / radius[1:]
hydrogen = solve_radial_schrodinger(
    radius,
    coulomb,
    angular_momentum=0,
    num_states=3,
)
```

These states use the reduced radial convention `u(r) = r R(r)`.

## Molecular DFT single point

```python
from zynnova.dft import ElectronicConfig, single_point
from zynnova.structure import StructureData

water = StructureData(
    atomic_numbers=[8, 1, 1],
    positions=[
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ],
)

electronic = ElectronicConfig(
    backend="pyscf",
    xc="PBE0",
    basis="def2-tzvp",
    grid_level=4,
    scf_tolerance=1.0e-10,
    density_fit=True,
)
result = single_point(water, electronic=electronic)
print(result.energy_eV)
print(result.forces_eV_per_A)
```

`spin` is PySCF's number of alpha electrons minus beta electrons (`2S`), not
the multiplicity. Charged/open-shell systems must use a charge and spin
consistent with the electron count.

## Born–Oppenheimer AIMD

```python
from zynnova.dft import (
    AIMDConfig,
    AIMDEnsemble,
    AIMDOutputConfig,
    ElectronicConfig,
    run_aimd,
)

electronic = ElectronicConfig(
    backend="pyscf",
    xc="PBE",
    basis="def2-svp",
    scf_tolerance=1.0e-9,
    density_fit=True,
)
config = AIMDConfig(
    ensemble=AIMDEnsemble.NVT_LANGEVIN,
    steps=1000,
    timestep_fs=0.25,
    temperature_K=300.0,
    friction_per_fs=0.01,
    integrator_backend="cpp",
    output=AIMDOutputConfig(
        directory="water-aimd",
        trajectory_interval=5,
        log_interval=1,
        checkpoint_interval=10,
    ),
)

result = run_aimd(water, electronic=electronic, config=config)
```

Resume to a new total step count with the same physical and output
configuration:

```python
config.steps = 2000
result = run_aimd(
    water,
    electronic=electronic,
    config=config,
    resume=True,
)
```

The native checkpoint stores coordinates, Å/fs velocities, fixed/mobile atoms,
the completed step, and the exact random-number-generator state. NVT restarts
therefore continue the same stochastic sequence.

## Periodic DFT

For GPAW, the typed configuration maps common plane-wave settings:

```python
electronic = ElectronicConfig(
    backend="gpaw",
    mode="pw",
    xc="PBE",
    plane_wave_cutoff_eV=600.0,
    kpoints=(4, 4, 4),
    scf_tolerance=1.0e-8,
    txt="gpaw.log",
    backend_kwargs={
        "occupations": {"name": "fermi-dirac", "width": 0.05},
    },
)
result = single_point(periodic_structure, electronic=electronic)
```

For another ASE calculator, pass its exact backend-specific arguments:

```python
from zynnova.dft import create_dft_calculator, run_aimd

calculator = create_dft_calculator(
    "espresso",
    profile=espresso_profile,
    pseudopotentials=pseudopotentials,
    input_data=input_data,
    kpts=(4, 4, 4),
)
result = run_aimd(periodic_structure, calculator, config)
```

## Accuracy and speed checklist

- Converge basis size or plane-wave cutoff, k-point mesh, real-space/grid
  accuracy, smearing, SCF tolerance, pseudopotential choice, and cell size for
  the target observable.
- Converge AIMD timestep separately. Hydrogen-rich systems commonly need a
  smaller timestep than heavy-atom solids.
- Run a short NVE trajectory and inspect total-energy drift before production
  NVT sampling.
- Select the exchange-correlation functional for the chemistry. Dispersion,
  strong correlation, charge transfer, excited states, and bond breaking can
  require methods beyond a default GGA.
- PySCF reuses the previous density matrix, supports density fitting, and can
  use GPU4PySCF with `use_gpu=True`.
- Plane-wave engines should use their native MPI/domain parallelism. The Python
  orchestration overhead is normally negligible relative to an SCF force call.
- Do not enable unsafe compiler `fast-math`: the native targets retain strict
  floating-point behavior and release the Python GIL during heavy kernels.

## Native implementation

| Layer | Files | Responsibility |
| --- | --- | --- |
| C++ quantum core | `cpp/include/zynnova/dft/quantum.hpp`, `cpp/src/dft/quantum.cpp` | Sturm-bisection/inverse-iteration bound states and complex Crank–Nicolson propagation |
| C++ AIMD core | `cpp/include/zynnova/dft/aimd.hpp`, `cpp/src/dft/aimd.cpp` | Unit-safe NVE/NVT integration, temperature, kinetic energy, reproducible RNG |
| Python binding | `cpp/bindings/dft_module.cpp` | NumPy/pybind11 boundary and GIL release |
| Python API | `src/zynnova/dft` | Units, validation, backend construction, electronic calculations, AIMD I/O/restart |
