from __future__ import annotations

import time
from typing import Any

import numpy as np

from .adapters import to_ase_atoms, to_structure_data
from .backends import calculator_from_config, create_dft_calculator
from .config import ElectronicConfig
from .results import ElectronicStructureResult


def _optional_property(atoms: Any, method_name: str, *args: Any, **kwargs: Any):
    method = getattr(atoms, method_name, None)
    if method is None:
        return None
    try:
        return np.asarray(method(*args, **kwargs), dtype=np.float64)
    except Exception:
        return None


def single_point(
    structure: Any,
    calculator: Any | None = None,
    *,
    electronic: ElectronicConfig | None = None,
    backend: str | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> ElectronicStructureResult:
    """Calculate electronic energy, analytic forces, and available observables."""
    atoms = to_ase_atoms(structure)
    if calculator is None:
        if electronic is not None and backend is not None:
            raise ValueError("provide either electronic= or backend=, not both")
        if electronic is not None:
            calculator = calculator_from_config(electronic, structure=atoms)
            backend_name = electronic.backend
        else:
            backend_name = backend or "pyscf"
            calculator = create_dft_calculator(backend_name, **dict(backend_kwargs or {}))
    else:
        backend_name = type(calculator).__name__
    atoms.calc = calculator
    started = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    wall_time = time.perf_counter() - started
    stress = _optional_property(atoms, "get_stress", voigt=False) if np.any(atoms.pbc) else None
    dipole = _optional_property(atoms, "get_dipole_moment")
    charges = _optional_property(atoms, "get_charges")
    magnetic_moments = _optional_property(atoms, "get_magnetic_moments")
    metadata = {
        "calculator": type(calculator).__qualname__,
        "implemented_properties": sorted(getattr(calculator, "implemented_properties", ())),
    }
    if hasattr(calculator, "converged"):
        metadata["converged"] = bool(calculator.converged)
    return ElectronicStructureResult(
        structure=to_structure_data(atoms),
        energy_eV=energy,
        forces_eV_per_A=forces,
        stress_eV_per_A3=stress,
        dipole_eA=dipole,
        charges_e=charges,
        magnetic_moments=magnetic_moments,
        wall_time_s=wall_time,
        backend=str(backend_name),
        metadata=metadata,
    )
