"""ASE calculator for the stable ZIVAR energy and electronic observables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ._deps import MissingZIVARDependency, require_torch
from .checkpoint import load_zivar
from .data import atoms_to_batch

torch = require_torch()

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
except ImportError:
    Calculator = object  # type: ignore[assignment,misc]
    PropertyNotImplementedError = RuntimeError  # type: ignore[assignment,misc]
    all_changes = ("positions", "numbers", "cell", "pbc")


class ZIVARCalculator(Calculator):  # type: ignore[misc]
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "charges",
        "dipole",
        "magmoms",
        "magmom_vectors",
        "dipoles",
        "quadrupoles",
        "oxidation_states",
        "effective_field_T",
        "magnetic_torque_eV",
    ]

    def __init__(
        self,
        model: Any,
        *,
        device: str | Any | None = None,
        dtype: str | None = None,
        analytic_stress: bool = True,
        require_electronic_validity: bool = True,
        infer_oxidation_states: bool = False,
        **kwargs: Any,
    ) -> None:
        if Calculator is object:
            raise MissingZIVARDependency("ZIVARCalculator requires ASE")
        super().__init__(**kwargs)
        resolved_dtype = getattr(torch, dtype) if dtype else None
        self.model = model.to(device=device, dtype=resolved_dtype).eval()
        self.analytic_stress = bool(analytic_stress)
        self.require_electronic_validity = bool(require_electronic_validity)
        self.infer_oxidation_states = bool(infer_oxidation_states)

    def calculate(
        self,
        atoms: Any | None = None,
        properties: Iterable[str] = ("energy", "forces"),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("ASE did not provide atoms")
        requested = set(properties)
        if "oxidation_states" in requested and not self.model.config.electronic.oxidation.enabled:
            raise PropertyNotImplementedError(
                "oxidation_states requires an explicitly configured and supervised "
                "formal-oxidation head"
            )
        batch, conditions = atoms_to_batch(self.atoms, self.model)
        conditions["infer_oxidation_states"] = self.infer_oxidation_states or (
            "oxidation_states" in properties
        )
        volume = abs(float(np.linalg.det(self.atoms.cell.array)))
        need_stress = self.analytic_stress and volume > 1.0e-12 and "stress" in properties
        with torch.enable_grad():
            output = self.model.energy_forces_stress(
                batch,
                conditions=conditions,
                create_graph=False,
                compute_stress=need_stress,
                compute_spin_fields=(self.model.config.spin.mode == "spin_lattice"),
            )
        if self.require_electronic_validity and not bool(output["electronic_converged"]):
            residual = float(output["electronic_residual"].max().detach())
            raise RuntimeError(
                f"{output['electronic_method']} electronic state is invalid: "
                f"residual={residual:.6g}"
            )

        def array(name: str) -> np.ndarray:
            return output[name].detach().cpu().numpy()

        energy = float(array("energy").reshape(-1)[0])
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": array("forces"),
            "charges": array("charges"),
            "total_charge": float(array("total_charge").reshape(-1)[0]),
            "dipole": array("total_dipole").reshape(3),
            "magmoms": array("magmoms"),
            "magmom_vectors": array("magmom_vectors"),
            "total_magnetic_moment": array("total_magnetic_moment").reshape(3),
            "dipoles": array("dipoles"),
            "quadrupoles": array("quadrupoles"),
            "electronic_residual": array("electronic_residual"),
            "spin_constraint_residual": array("spin_constraint_residual"),
            "electronic_converged": bool(output["electronic_converged"]),
            "electronic_method": str(output["electronic_method"]),
            "atomic_energy": array("atomic_energy"),
            "coulomb_energy": array("coulomb_energy"),
            "electronegativity": array("electronegativity"),
            "hardness": array("hardness"),
            "magnetic_energy": float(array("magnetic_energy").reshape(-1)[0]),
            "exchange_energy": float(array("exchange_energy").reshape(-1)[0]),
            "dmi_energy": float(array("dmi_energy").reshape(-1)[0]),
            "magnetic_anisotropy_energy": float(
                array("magnetic_anisotropy_energy").reshape(-1)[0]
            ),
        }
        if "effective_field_T" in output:
            self.results["effective_field_T"] = array("effective_field_T")
            self.results["magnetic_torque_eV"] = array("magnetic_torque_eV")
        if "oxidation_states" in output:
            self.results["oxidation_states"] = array("oxidation_states")
            self.results["oxidation_expectation"] = array("oxidation_expectation")
            self.results["oxidation_confidence"] = array("oxidation_confidence")
            self.results["oxidation_entropy"] = array("oxidation_entropy")
        if need_stress:
            self.results["stress"] = array("stress").reshape(6)


def zivar_calculator(
    potential: Any | str | Path,
    *,
    device: str | Any = "cpu",
    dtype: str | None = None,
    analytic_stress: bool = True,
    require_electronic_validity: bool = True,
    infer_oxidation_states: bool = False,
) -> ZIVARCalculator:
    model = (
        load_zivar(potential, device=device, dtype=dtype)
        if isinstance(potential, str | Path)
        else potential
    )
    return ZIVARCalculator(
        model,
        device=device,
        dtype=dtype,
        analytic_stress=analytic_stress,
        require_electronic_validity=require_electronic_validity,
        infer_oxidation_states=infer_oxidation_states,
    )


def predict_structure(
    structures: Any | Iterable[Any],
    potential: Any | str | Path,
    *,
    device: str | Any = "cpu",
    dtype: str | None = None,
    infer_oxidation_states: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    single = hasattr(structures, "get_atomic_numbers")
    items = [structures] if single else list(structures)
    calculator = zivar_calculator(
        potential,
        device=device,
        dtype=dtype,
        infer_oxidation_states=infer_oxidation_states,
    )
    results: list[dict[str, Any]] = []
    for structure in items:
        atoms = structure.copy()
        atoms.calc = calculator
        requested = ["energy", "forces", "charges", "magmoms"]
        if infer_oxidation_states:
            requested.append("oxidation_states")
        if abs(float(np.linalg.det(atoms.cell.array))) > 1.0e-12:
            requested.append("stress")
        calculator.calculate(atoms, properties=requested)
        results.append(dict(calculator.results))
    return results[0] if single else results


__all__ = ["ZIVARCalculator", "predict_structure", "zivar_calculator"]
