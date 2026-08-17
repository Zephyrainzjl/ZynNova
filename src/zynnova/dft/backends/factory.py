from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from ..config import ElectronicConfig
from ..exceptions import DFTConfigurationError, MissingElectronicBackendError

_ASE_BACKENDS: dict[str, tuple[str, str]] = {
    "abinit": ("ase.calculators.abinit", "Abinit"),
    "aims": ("ase.calculators.aims", "Aims"),
    "castep": ("ase.calculators.castep", "Castep"),
    "cp2k": ("ase.calculators.cp2k", "CP2K"),
    "elk": ("ase.calculators.elk", "ELK"),
    "espresso": ("ase.calculators.espresso", "Espresso"),
    "nwchem": ("ase.calculators.nwchem", "NWChem"),
    "orca": ("ase.calculators.orca", "ORCA"),
    "psi4": ("ase.calculators.psi4", "Psi4"),
    "siesta": ("ase.calculators.siesta", "Siesta"),
    "vasp": ("ase.calculators.vasp", "Vasp"),
}

_ALIASES = {
    "qe": "espresso",
    "quantum_espresso": "espresso",
    "quantumespresso": "espresso",
    "fhi_aims": "aims",
    "pyscf_dft": "pyscf",
}


def _normalize_backend(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, key)


def _module_importable(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def available_dft_backends(*, installed_only: bool = False) -> dict[str, bool]:
    """Return whether each Python calculator interface can be imported.

    For executable-based engines, ``True`` confirms the Python interface only;
    pseudopotentials, profiles, licenses, and external executables still need to
    be configured by the user.
    """
    status = {
        "pyscf": _module_importable("ase") and _module_importable("pyscf"),
        "gpaw": _module_importable("ase") and _module_importable("gpaw"),
    }
    for name, (module_name, _) in _ASE_BACKENDS.items():
        status[name] = _module_importable(module_name)
    if installed_only:
        return {name: available for name, available in status.items() if available}
    return status


def create_dft_calculator(name: str, /, **kwargs: Any):
    """Create a DFT-capable ASE calculator by a stable ZynNova backend name.

    Backend-specific keyword arguments are deliberately forwarded without lossy
    translation. This keeps advanced settings such as pseudopotentials,
    parallel profiles, smearing, and convergence dictionaries fully accessible.
    """
    key = _normalize_backend(name)
    if key == "external":
        calculator = kwargs.get("calculator")
        if calculator is None:
            raise DFTConfigurationError("name='external' requires calculator=<ASE calculator>")
        return calculator
    try:
        if key == "pyscf":
            from .pyscf_calculator import PySCFCalculator

            return PySCFCalculator(**kwargs)
        if key == "gpaw":
            gpaw = import_module("gpaw")
            mode = kwargs.pop("mode", None)
            cutoff = kwargs.pop("plane_wave_cutoff_eV", None)
            if isinstance(mode, str) and mode.lower() == "pw":
                mode = gpaw.PW(float(cutoff or 500.0))
            elif mode is None and cutoff is not None:
                mode = gpaw.PW(float(cutoff))
            if mode is not None:
                kwargs["mode"] = mode
            return gpaw.GPAW(**kwargs)
        if key in _ASE_BACKENDS:
            module_name, class_name = _ASE_BACKENDS[key]
            calculator_class = getattr(import_module(module_name), class_name)
            return calculator_class(**kwargs)
    except ImportError as exc:
        raise MissingElectronicBackendError(
            f"Electronic backend {key!r} is not installed or cannot be imported"
        ) from exc
    known = sorted({"pyscf", "gpaw", "external", *_ASE_BACKENDS})
    raise ValueError(f"Unknown DFT backend {name!r}; choose one of {known}")


def calculator_from_config(
    config: ElectronicConfig,
    *,
    structure: Any | None = None,
):
    """Build a calculator from validated common electronic settings."""
    config.validate()
    key = _normalize_backend(config.backend)
    kwargs = dict(config.backend_kwargs)
    if key == "pyscf":
        common = {
            "xc": config.xc,
            "basis": config.basis,
            "charge": config.charge,
            "spin": config.spin,
            "conv_tol": config.scf_tolerance,
            "max_cycle": config.max_scf_cycles,
            "grid_level": config.grid_level,
            "density_fit": config.density_fit,
            "auxbasis": config.auxiliary_basis,
            "use_gpu": config.use_gpu,
            "newton_on_failure": config.newton_on_failure,
        }
        common.update(kwargs)
        return create_dft_calculator("pyscf", **common)
    if key == "gpaw":
        periodic = False
        if structure is not None:
            try:
                periodic = bool(np.any(structure.get_pbc()))
            except AttributeError:
                periodic = bool(np.any(getattr(structure, "pbc", False)))
        mode = config.mode.lower()
        if mode == "auto":
            mode = "pw" if periodic else "fd"
        common = {
            "mode": mode,
            "plane_wave_cutoff_eV": config.plane_wave_cutoff_eV,
            "xc": config.xc,
            "charge": config.charge,
            "maxiter": config.max_scf_cycles,
            "convergence": {"energy": config.scf_tolerance},
            "txt": config.txt,
        }
        if periodic:
            common["kpts"] = config.kpoints
        common.update(kwargs)
        return create_dft_calculator("gpaw", **common)
    if key == "external":
        return create_dft_calculator("external", **kwargs)
    return create_dft_calculator(key, **kwargs)
