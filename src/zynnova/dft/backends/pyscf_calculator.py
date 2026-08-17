from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError as exc:  # imported lazily by the backend factory
    raise ImportError("PySCFCalculator requires ASE; install zynnova[dft-pyscf]") from exc

from ..exceptions import MissingElectronicBackendError, SCFConvergenceError


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


class PySCFCalculator(Calculator):
    """All-electron molecular Kohn–Sham DFT calculator with analytic forces.

    The previous density matrix is reused as the next Born–Oppenheimer MD SCF
    guess. Density fitting and GPU4PySCF are optional acceleration paths.
    Periodic cells are intentionally rejected; use GPAW, Quantum ESPRESSO,
    VASP, CP2K, or another periodic ASE calculator for those systems.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        xc: str = "PBE",
        basis: str | dict[str, Any] = "def2-svp",
        charge: int = 0,
        spin: int = 0,
        conv_tol: float = 1.0e-9,
        max_cycle: int = 100,
        grid_level: int = 4,
        density_fit: bool = True,
        auxbasis: str | dict[str, Any] | None = None,
        use_gpu: bool = False,
        newton_on_failure: bool = True,
        verbose: int = 0,
        max_memory_mb: float | None = None,
        scf_options: dict[str, Any] | None = None,
        label: str = "pyscf",
        **calculator_kwargs: Any,
    ) -> None:
        super().__init__(label=label, **calculator_kwargs)
        self.xc = xc
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.grid_level = int(grid_level)
        self.density_fit = bool(density_fit)
        self.auxbasis = auxbasis
        self.use_gpu = bool(use_gpu)
        self.newton_on_failure = bool(newton_on_failure)
        self.verbose = int(verbose)
        self.max_memory_mb = max_memory_mb
        self.scf_options = dict(scf_options or {})
        self._density_matrix: Any | None = None
        self.scf_object: Any | None = None
        self.converged = False

    def _build_scf(self, atoms: Any):
        try:
            from pyscf import dft, gto
        except ImportError as exc:
            raise MissingElectronicBackendError(
                "PySCF is required; install zynnova[dft-pyscf]"
            ) from exc
        if np.any(atoms.pbc):
            raise ValueError(
                "PySCFCalculator supports isolated molecules only; use a periodic "
                "DFT backend for structures with PBC"
            )
        molecule = gto.Mole()
        molecule.atom = [
            (symbol, tuple(float(value) for value in position))
            for symbol, position in zip(
                atoms.get_chemical_symbols(),
                atoms.get_positions(),
                strict=True,
            )
        ]
        molecule.unit = "Angstrom"
        molecule.basis = self.basis
        molecule.charge = self.charge
        molecule.spin = self.spin
        molecule.verbose = self.verbose
        if self.max_memory_mb is not None:
            molecule.max_memory = float(self.max_memory_mb)
        molecule.build()

        mean_field = dft.UKS(molecule) if self.spin != 0 else dft.RKS(molecule)
        mean_field.xc = self.xc
        mean_field.conv_tol = self.conv_tol
        mean_field.max_cycle = self.max_cycle
        mean_field.grids.level = self.grid_level
        if self.density_fit:
            mean_field = mean_field.density_fit(auxbasis=self.auxbasis)
        for name, value in self.scf_options.items():
            if not hasattr(mean_field, name):
                raise ValueError(f"Unknown PySCF SCF option: {name!r}")
            setattr(mean_field, name, value)
        if self.use_gpu:
            try:
                mean_field = mean_field.to_gpu()
            except (AttributeError, ImportError) as exc:
                raise MissingElectronicBackendError(
                    "use_gpu=True requires a compatible GPU4PySCF installation"
                ) from exc
        return mean_field

    def calculate(
        self,
        atoms: Any = None,
        properties: tuple[str, ...] | list[str] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        mean_field = self._build_scf(self.atoms)
        density_guess = self._density_matrix
        if density_guess is not None:
            try:
                expected = mean_field.get_ovlp().shape[-1]
                if _as_numpy(density_guess).shape[-1] != expected:
                    density_guess = None
            except Exception:
                density_guess = None
        energy_hartree = mean_field.kernel(dm0=density_guess)
        if not bool(mean_field.converged) and self.newton_on_failure:
            density_guess = mean_field.make_rdm1()
            mean_field = mean_field.newton()
            mean_field.conv_tol = self.conv_tol
            mean_field.max_cycle = self.max_cycle
            energy_hartree = mean_field.kernel(dm0=density_guess)
        self.converged = bool(mean_field.converged)
        if not self.converged or not np.isfinite(float(energy_hartree)):
            raise SCFConvergenceError(
                f"PySCF {self.xc}/{self.basis} did not converge in {self.max_cycle} cycles"
            )
        try:
            gradient = _as_numpy(mean_field.nuc_grad_method().kernel())
        except Exception as exc:
            raise SCFConvergenceError(
                "PySCF energy converged, but the analytic nuclear gradient failed"
            ) from exc
        try:
            from ase.units import Bohr, Hartree
        except ImportError as exc:
            raise MissingElectronicBackendError("ASE units are unavailable") from exc
        energy_eV = float(energy_hartree) * Hartree
        forces_eV_per_A = -np.asarray(gradient, dtype=np.float64) * Hartree / Bohr
        if forces_eV_per_A.shape != (len(self.atoms), 3):
            raise SCFConvergenceError("PySCF returned an invalid gradient shape")
        self.results = {
            "energy": energy_eV,
            "free_energy": energy_eV,
            "forces": forces_eV_per_A,
        }
        self._density_matrix = mean_field.make_rdm1()
        self.scf_object = mean_field
