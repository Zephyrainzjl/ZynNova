from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import MissingBackendError


def create_classical_calculator(name: str, /, **kwargs: Any):
    """Create a classical ASE calculator by a stable ZynNova name.

    Supported built-ins are ``emt``, ``lennard_jones``/``lj``, ``morse``,
    ``tip3p``, ``tip4p``, ``eam``, ``tersoff``, ``lammpslib``, and
    ``lammpsrun``.  An already-created calculator may be passed through with
    ``name='external', calculator=...``.
    """
    key = name.strip().lower().replace("-", "_")
    try:
        if key == "emt":
            from ase.calculators.emt import EMT

            return EMT(**kwargs)
        if key in {"lennard_jones", "lj"}:
            from ase.calculators.lj import LennardJones

            return LennardJones(**kwargs)
        if key == "morse":
            from ase.calculators.morse import MorsePotential

            return MorsePotential(**kwargs)
        if key == "tip3p":
            from ase.calculators.tip3p import TIP3P

            return TIP3P(**kwargs)
        if key == "tip4p":
            from ase.calculators.tip4p import TIP4P

            return TIP4P(**kwargs)
        if key == "eam":
            from ase.calculators.eam import EAM

            if "potential" in kwargs:
                kwargs["potential"] = str(Path(kwargs["potential"]))
            return EAM(**kwargs)
        if key == "tersoff":
            from ase.calculators.tersoff import Tersoff

            return Tersoff(**kwargs)
        if key == "lammpslib":
            from ase.calculators.lammpslib import LAMMPSlib

            return LAMMPSlib(**kwargs)
        if key in {"lammpsrun", "lammps"}:
            from ase.calculators.lammpsrun import LAMMPS

            return LAMMPS(**kwargs)
    except ImportError as exc:
        raise MissingBackendError(
            f"Calculator {key!r} requires ASE or its external engine dependencies"
        ) from exc
    if key == "external":
        calculator = kwargs.get("calculator")
        if calculator is None:
            raise ValueError("name='external' requires calculator=...")
        return calculator
    raise ValueError(f"Unknown classical calculator: {name!r}")


def calculator_capabilities(calculator: Any) -> frozenset[str]:
    return frozenset(getattr(calculator, "implemented_properties", ()))
