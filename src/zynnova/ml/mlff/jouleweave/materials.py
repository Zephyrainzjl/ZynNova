from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ....dynamics.adapters import to_ase_atoms
from ....dynamics.config import CellMode, RelaxationConfig
from ....dynamics.exceptions import PotentialError
from ....dynamics.relaxation import relax
from .calculator import jouleweave_calculator, load_jouleweave_calculator
from .model import JouleWeave


def _resolve_calculator(
    potential: Any,
    *,
    require_stress: bool,
    device: str,
    dtype: str,
    compile_model: bool,
) -> Any:
    if isinstance(potential, JouleWeave):
        calculator = jouleweave_calculator(
            potential,
            device=device,
            dtype=dtype,
            analytic_stress=require_stress,
            compile_model=compile_model,
        )
    elif isinstance(potential, (str, Path)):
        calculator = load_jouleweave_calculator(
            potential,
            device=device,
            dtype=dtype,
            analytic_stress=require_stress,
        )
    elif hasattr(potential, "calculate") and hasattr(potential, "implemented_properties"):
        calculator = potential
    else:
        raise TypeError(
            "potential must be a JouleWeave model, checkpoint path, or ASE calculator"
        )
    if require_stress and "stress" not in set(calculator.implemented_properties):
        raise PotentialError(
            "This workflow requires stress. Pass a JouleWeave model/checkpoint so an "
            "analytic-stress calculator can be built, or provide an ASE calculator "
            "that implements stress."
        )
    return calculator


def _periodic_atoms(structure: Any) -> Any:
    atoms = to_ase_atoms(structure)
    if not bool(np.all(np.asarray(atoms.pbc, dtype=bool))):
        raise ValueError("crystal workflows require periodic boundary conditions in all axes")
    if float(atoms.get_volume()) <= 0:
        raise ValueError("crystal workflows require a positive cell volume")
    return atoms


def _fixed_cell_config(config: RelaxationConfig | None) -> RelaxationConfig:
    if config is None:
        return RelaxationConfig(
            cell_mode=CellMode.FIXED,
            fmax_eV_per_A=0.03,
            max_steps=300,
            trajectory_filename=None,
            logfile=None,
        )
    return replace(config, cell_mode=CellMode.FIXED)


def optimize_jouleweave_structure(
    structure: Any,
    potential: Any,
    *,
    relax_cell: bool = True,
    config: RelaxationConfig | None = None,
    output_directory: str | Path = "jouleweave-relax",
    device: str = "auto",
    dtype: str = "float32",
    compile_model: bool = False,
) -> Any:
    """Relax atomic coordinates and, by default, all six cell degrees of freedom."""

    if config is None:
        config = RelaxationConfig(
            cell_mode=CellMode.FRECHET if relax_cell else CellMode.FIXED,
        )
    elif relax_cell and CellMode(config.cell_mode) is CellMode.FIXED:
        config = replace(config, cell_mode=CellMode.FRECHET)
    elif not relax_cell:
        config = replace(config, cell_mode=CellMode.FIXED)
    if CellMode(config.cell_mode) is not CellMode.FIXED:
        structure = _periodic_atoms(structure)
    calculator = _resolve_calculator(
        potential,
        require_stress=CellMode(config.cell_mode) is not CellMode.FIXED,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
    )
    return relax(
        structure,
        calculator,
        config,
        output_directory=output_directory,
    )


@dataclass(slots=True)
class EquationOfStateResult:
    volumes_A3: np.ndarray
    energies_eV: np.ndarray
    volume_factors: np.ndarray
    equilibrium_volume_A3: float
    equilibrium_energy_eV: float
    bulk_modulus_eV_per_A3: float
    bulk_modulus_GPa: float
    eos_model: str
    fit: Any

    def plot(self, filename: str | Path | None = None, **kwargs: Any) -> Any:
        kwargs.setdefault("show", False)
        return self.fit.plot(
            filename=None if filename is None else str(filename),
            **kwargs,
        )


def fit_jouleweave_eos(
    structure: Any,
    potential: Any,
    *,
    n_points: int = 7,
    volume_strain: float = 0.06,
    eos_model: str = "birchmurnaghan",
    relax_internal: bool = True,
    relaxation_config: RelaxationConfig | None = None,
    output_directory: str | Path = "jouleweave-eos",
    device: str = "auto",
    dtype: str = "float32",
    compile_model: bool = False,
) -> EquationOfStateResult:
    """Fit an energy-volume curve and return equilibrium volume and bulk modulus."""

    if n_points < 5:
        raise ValueError("n_points must be at least 5 for a stable EOS fit")
    if not 0 < volume_strain < 0.5:
        raise ValueError("volume_strain must lie in (0, 0.5)")
    atoms = _periodic_atoms(structure)
    calculator = _resolve_calculator(
        potential,
        require_stress=False,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
    )
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cell = np.asarray(atoms.cell.array, dtype=float)
    volume_factors = np.linspace(1.0 - volume_strain, 1.0 + volume_strain, n_points)
    volumes: list[float] = []
    energies: list[float] = []
    fixed_config = _fixed_cell_config(relaxation_config)
    for index, volume_factor in enumerate(volume_factors):
        trial = atoms.copy()
        linear_scale = float(volume_factor) ** (1.0 / 3.0)
        trial.set_cell(base_cell * linear_scale, scale_atoms=True)
        if relax_internal:
            relaxed = relax(
                trial,
                calculator,
                fixed_config,
                output_directory=output_root / f"volume-{index:02d}",
            )
            trial = to_ase_atoms(relaxed.final_structure)
        trial.calc = calculator
        volumes.append(float(trial.get_volume()))
        energies.append(float(trial.get_potential_energy()))

    try:
        from ase import units
        from ase.eos import EquationOfState
    except ImportError as exc:
        raise ImportError("EOS fitting requires ASE and SciPy; install zynnova[mlff]") from exc
    fit = EquationOfState(volumes, energies, eos=eos_model)
    equilibrium_volume, equilibrium_energy, bulk_modulus = fit.fit()
    return EquationOfStateResult(
        volumes_A3=np.asarray(volumes, dtype=float),
        energies_eV=np.asarray(energies, dtype=float),
        volume_factors=np.asarray(volume_factors, dtype=float),
        equilibrium_volume_A3=float(equilibrium_volume),
        equilibrium_energy_eV=float(equilibrium_energy),
        bulk_modulus_eV_per_A3=float(bulk_modulus),
        bulk_modulus_GPa=float(bulk_modulus / units.GPa),
        eos_model=eos_model,
        fit=fit,
    )


def _strain_tensor(component: int, value: float) -> np.ndarray:
    strain = np.zeros((3, 3), dtype=float)
    if component < 3:
        strain[component, component] = value
    else:
        i, j = ((1, 2), (0, 2), (0, 1))[component - 3]
        strain[i, j] = 0.5 * value
        strain[j, i] = 0.5 * value
    return strain


def _safe_inverse(value: float, *, threshold: float = 1.0e-14) -> float:
    return float("nan") if abs(value) < threshold else 1.0 / value


@dataclass(slots=True)
class ElasticityResult:
    stiffness_eV_per_A3: np.ndarray
    stiffness_GPa: np.ndarray
    compliance_A3_per_eV: np.ndarray
    strain_amplitude: float
    applied_strains: np.ndarray
    measured_stresses_eV_per_A3: np.ndarray
    bulk_modulus_voigt_GPa: float
    bulk_modulus_reuss_GPa: float
    bulk_modulus_hill_GPa: float
    shear_modulus_voigt_GPa: float
    shear_modulus_reuss_GPa: float
    shear_modulus_hill_GPa: float
    young_modulus_hill_GPa: float
    poisson_ratio_hill: float


def calculate_jouleweave_elasticity(
    structure: Any,
    potential: Any,
    *,
    strain: float = 0.005,
    relax_internal: bool = True,
    symmetrize: bool = True,
    relaxation_config: RelaxationConfig | None = None,
    output_directory: str | Path = "jouleweave-elastic",
    device: str = "auto",
    dtype: str = "float32",
    compile_model: bool = False,
) -> ElasticityResult:
    """Calculate the full 6x6 elastic tensor by central stress differences."""

    if not 0 < strain < 0.1:
        raise ValueError("strain must lie in (0, 0.1)")
    atoms = _periodic_atoms(structure)
    calculator = _resolve_calculator(
        potential,
        require_stress=True,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
    )
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cell = np.asarray(atoms.cell.array, dtype=float)
    fixed_config = _fixed_cell_config(relaxation_config)
    stiffness = np.zeros((6, 6), dtype=float)
    applied_strains: list[np.ndarray] = []
    measured_stresses: list[np.ndarray] = []

    for component in range(6):
        stresses: list[np.ndarray] = []
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            value = sign * strain
            voigt_strain = np.zeros(6, dtype=float)
            voigt_strain[component] = value
            trial = atoms.copy()
            deformation = np.eye(3) + _strain_tensor(component, value)
            trial.set_cell(base_cell @ deformation, scale_atoms=True)
            if relax_internal:
                relaxed = relax(
                    trial,
                    calculator,
                    fixed_config,
                    output_directory=output_root / f"c{component + 1}-{label}",
                )
                trial = to_ase_atoms(relaxed.final_structure)
            trial.calc = calculator
            stress_value = np.asarray(trial.get_stress(voigt=True), dtype=float).reshape(6)
            applied_strains.append(voigt_strain)
            measured_stresses.append(stress_value)
            stresses.append(stress_value)
        stiffness[:, component] = (stresses[1] - stresses[0]) / (2.0 * strain)

    if symmetrize:
        stiffness = 0.5 * (stiffness + stiffness.T)
    compliance = np.linalg.pinv(stiffness, hermitian=symmetrize)
    c = stiffness
    s = compliance
    bulk_voigt = (
        c[0, 0]
        + c[1, 1]
        + c[2, 2]
        + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])
    ) / 9.0
    shear_voigt = (
        c[0, 0]
        + c[1, 1]
        + c[2, 2]
        - c[0, 1]
        - c[0, 2]
        - c[1, 2]
        + 3.0 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0
    bulk_reuss = _safe_inverse(
        s[0, 0]
        + s[1, 1]
        + s[2, 2]
        + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2])
    )
    shear_reuss = 15.0 * _safe_inverse(
        4.0
        * (
            s[0, 0]
            + s[1, 1]
            + s[2, 2]
            - s[0, 1]
            - s[0, 2]
            - s[1, 2]
        )
        + 3.0 * (s[3, 3] + s[4, 4] + s[5, 5])
    )
    bulk_hill = 0.5 * (bulk_voigt + bulk_reuss)
    shear_hill = 0.5 * (shear_voigt + shear_reuss)
    denominator = 3.0 * bulk_hill + shear_hill
    if abs(denominator) < 1.0e-14:
        young_hill = float("nan")
        poisson_hill = float("nan")
    else:
        young_hill = 9.0 * bulk_hill * shear_hill / denominator
        poisson_hill = (3.0 * bulk_hill - 2.0 * shear_hill) / (2.0 * denominator)

    try:
        from ase import units
    except ImportError as exc:
        raise ImportError("elasticity workflows require ASE; install zynnova[mlff]") from exc
    conversion = 1.0 / units.GPa
    return ElasticityResult(
        stiffness_eV_per_A3=stiffness,
        stiffness_GPa=stiffness * conversion,
        compliance_A3_per_eV=compliance,
        strain_amplitude=float(strain),
        applied_strains=np.asarray(applied_strains),
        measured_stresses_eV_per_A3=np.asarray(measured_stresses),
        bulk_modulus_voigt_GPa=float(bulk_voigt * conversion),
        bulk_modulus_reuss_GPa=float(bulk_reuss * conversion),
        bulk_modulus_hill_GPa=float(bulk_hill * conversion),
        shear_modulus_voigt_GPa=float(shear_voigt * conversion),
        shear_modulus_reuss_GPa=float(shear_reuss * conversion),
        shear_modulus_hill_GPa=float(shear_hill * conversion),
        young_modulus_hill_GPa=float(young_hill * conversion),
        poisson_ratio_hill=float(poisson_hill),
    )


@dataclass(slots=True)
class PhononResult:
    phonons: Any
    force_constants_eV_per_A2: np.ndarray
    supercell: tuple[int, int, int]
    displacement_A: float
    equilibrium_fmax_eV_per_A: float
    cache_prefix: Path

    def band_structure(self, path: Any, *, modes: bool = False, **kwargs: Any) -> Any:
        return self.phonons.get_band_structure(path, modes=modes, **kwargs)

    def dos(
        self,
        kpts: tuple[int, int, int] = (20, 20, 20),
        *,
        indices: list[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.phonons.get_dos(kpts=kpts, indices=indices, **kwargs)

    def write_modes(self, q_c: Any, **kwargs: Any) -> None:
        self.phonons.write_modes(q_c, **kwargs)

    def clean(self) -> Any:
        return self.phonons.clean()


def calculate_jouleweave_phonons(
    structure: Any,
    potential: Any,
    *,
    supercell: tuple[int, int, int] = (3, 3, 3),
    displacement_A: float = 0.01,
    method: str = "Frederiksen",
    symmetrize: int = 3,
    acoustic: bool = True,
    cutoff_A: float | None = None,
    born: bool = False,
    cleanup: bool = False,
    output_directory: str | Path = "jouleweave-phonons",
    device: str = "auto",
    dtype: str = "float32",
    compile_model: bool = False,
    read_kwargs: dict[str, Any] | None = None,
) -> PhononResult:
    """Run finite-displacement phonons and expose bands, DOS, and mode writing."""

    if len(supercell) != 3 or any(int(value) < 1 for value in supercell):
        raise ValueError("supercell must contain three positive integers")
    if displacement_A <= 0:
        raise ValueError("displacement_A must be positive")
    atoms = _periodic_atoms(structure)
    calculator = _resolve_calculator(
        potential,
        require_stress=False,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
    )
    atoms.calc = calculator
    equilibrium_forces = np.asarray(atoms.get_forces(), dtype=float)
    equilibrium_fmax = float(np.linalg.norm(equilibrium_forces, axis=1).max(initial=0.0))
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_prefix = output_root / "phonon"
    try:
        from ase.phonons import Phonons
    except ImportError as exc:
        raise ImportError("phonon workflows require ASE; install zynnova[mlff]") from exc
    phonons = Phonons(
        atoms,
        calculator,
        supercell=tuple(int(value) for value in supercell),
        delta=float(displacement_A),
        name=str(cache_prefix),
    )
    phonons.run()
    options = dict(read_kwargs or {})
    options.setdefault("method", method)
    options.setdefault("symmetrize", symmetrize)
    options.setdefault("acoustic", acoustic)
    options.setdefault("cutoff", cutoff_A)
    options.setdefault("born", born)
    phonons.read(**options)
    force_constants = np.asarray(phonons.get_force_constant(), dtype=float)
    if cleanup:
        phonons.clean()
    return PhononResult(
        phonons=phonons,
        force_constants_eV_per_A2=force_constants,
        supercell=tuple(int(value) for value in supercell),
        displacement_A=float(displacement_A),
        equilibrium_fmax_eV_per_A=equilibrium_fmax,
        cache_prefix=cache_prefix,
    )


class JouleWeaveMaterials:
    """CHGNet-style high-level facade for structure and crystal-property workflows."""

    def __init__(
        self,
        potential: Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
    ) -> None:
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.compile_model = bool(compile_model)

    def _options(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "compile_model": self.compile_model,
        }

    def relax(self, structure: Any, **kwargs: Any) -> Any:
        options = {**self._options(), **kwargs}
        return optimize_jouleweave_structure(
            structure,
            self.potential,
            **options,
        )

    def eos(self, structure: Any, **kwargs: Any) -> EquationOfStateResult:
        options = {**self._options(), **kwargs}
        return fit_jouleweave_eos(
            structure,
            self.potential,
            **options,
        )

    def elastic(self, structure: Any, **kwargs: Any) -> ElasticityResult:
        options = {**self._options(), **kwargs}
        return calculate_jouleweave_elasticity(
            structure,
            self.potential,
            **options,
        )

    def phonons(self, structure: Any, **kwargs: Any) -> PhononResult:
        options = {**self._options(), **kwargs}
        return calculate_jouleweave_phonons(
            structure,
            self.potential,
            **options,
        )

    def neb(self, initial: Any, final: Any, **kwargs: Any) -> Any:
        from .rare_events import JouleWeaveNEB

        workflow = JouleWeaveNEB(
            self.potential,
            **self._options(),
        )
        return workflow.run(initial, final, **kwargs)

    def dimer(self, structure: Any, **kwargs: Any) -> Any:
        from .rare_events import JouleWeaveDimer

        workflow = JouleWeaveDimer(
            self.potential,
            **self._options(),
        )
        return workflow.run(structure, **kwargs)

    def metadynamics(self, structure: Any, **kwargs: Any) -> Any:
        from .rare_events import JouleWeaveMetadynamics

        workflow = JouleWeaveMetadynamics(
            self.potential,
            **self._options(),
        )
        return workflow.run(structure, **kwargs)

    def cathode_cycling(self, parent_structure: Any, **kwargs: Any) -> Any:
        from .cathode import CathodeCyclingWorkflow

        workflow = CathodeCyclingWorkflow(
            self.potential,
            **self._options(),
        )
        return workflow.run(parent_structure, **kwargs)


__all__ = [
    "ElasticityResult",
    "EquationOfStateResult",
    "JouleWeaveMaterials",
    "PhononResult",
    "calculate_jouleweave_elasticity",
    "calculate_jouleweave_phonons",
    "fit_jouleweave_eos",
    "optimize_jouleweave_structure",
]
