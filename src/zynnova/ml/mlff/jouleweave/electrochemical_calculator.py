"""ASE calculator for constant-potential JouleWeave molecular dynamics."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ....dynamics import TorchPotentialCalculator
from ...common import load_checkpoint, resolve_device
from .calculator import jouleweave_input_adapter
from .grand_canonical import ConstantPotentialJouleWeave


def constant_potential_input_adapter(
    atoms: Any,
    *,
    positions: Any,
    cell: Any,
    device: Any,
    dtype: Any,
    default_electrode_potential_V: float | None = None,
) -> dict[str, Any]:
    """Add grand-canonical state variables to the normal JouleWeave input."""

    import torch

    inputs = jouleweave_input_adapter(
        atoms,
        positions=positions,
        cell=cell,
        device=device,
        dtype=dtype,
    )
    potential = atoms.info.get(
        "electrode_potential_V", default_electrode_potential_V
    )
    if potential is None:
        raise ValueError(
            "constant-potential MD requires atoms.info['electrode_potential_V'] "
            "or default_electrode_potential_V"
        )
    inputs["electrode_potential_V"] = torch.as_tensor(
        [float(potential)], device=device, dtype=dtype
    )
    if "electron_count" in atoms.info:
        inputs["electron_count"] = torch.as_tensor(
            [float(atoms.info["electron_count"])], device=device, dtype=dtype
        )
    if "reference_electron_count" in atoms.info:
        inputs["reference_electron_count"] = torch.as_tensor(
            [float(atoms.info["reference_electron_count"])],
            device=device,
            dtype=dtype,
        )
    return inputs


def _constant_potential_inference(
    model: ConstantPotentialJouleWeave,
    *,
    solve_electron_count: bool,
    electron_tolerance_eV: float,
    electron_maximum_iterations: int,
):
    import torch

    class _ConstantPotentialInference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
            resolved_inputs = inputs
            electron_solver_converged = True
            if solve_electron_count and "electron_count" not in inputs:
                electrons, electronic = self.model.solve_electron_count(
                    inputs,
                    tolerance_eV=electron_tolerance_eV,
                    maximum_iterations=electron_maximum_iterations,
                )
                resolved_inputs = dict(inputs)
                resolved_inputs["electron_count"] = electrons
                electron_solver_converged = bool(
                    electronic["electron_solver_converged"]
                )
            output = self.model.grand_potential_and_forces(
                resolved_inputs, create_graph=False
            )
            return {
                **output,
                "energy": output["grand_potential"],
                "electron_solver_converged": output["grand_potential"].new_tensor(
                    float(electron_solver_converged)
                ),
            }

    return _ConstantPotentialInference()


def constant_potential_jouleweave_calculator(
    model: ConstantPotentialJouleWeave,
    *,
    electrode_potential_V: float | None = None,
    device: str = "auto",
    dtype: str = "float32",
    solve_electron_count: bool = True,
    electron_tolerance_eV: float = 1.0e-4,
    electron_maximum_iterations: int = 24,
    finite_difference_stress: bool = False,
    compile_model: bool = False,
) -> TorchPotentialCalculator:
    """Create an ASE calculator whose conserved scalar is the grand potential.

    The calculator is appropriate for NVE/NVT/NPT workflows only after the
    wrapped potential has been trained and validated for the target chemistry,
    potential window, and reactive events.  NPT uses finite-difference stress
    because the current grand-canonical wrapper does not yet expose an analytic
    strain derivative.
    """

    if electron_tolerance_eV <= 0.0 or electron_maximum_iterations < 1:
        raise ValueError("electron self-consistency controls are invalid")
    resolved_device = resolve_device(device)
    import torch

    try:
        resolved_dtype = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {dtype!r}") from exc
    if not resolved_dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    model.to(device=resolved_device, dtype=resolved_dtype)
    wrapped = _constant_potential_inference(
        model,
        solve_electron_count=solve_electron_count,
        electron_tolerance_eV=electron_tolerance_eV,
        electron_maximum_iterations=electron_maximum_iterations,
    )
    adapter = partial(
        constant_potential_input_adapter,
        default_electrode_potential_V=electrode_potential_V,
    )
    extra_properties = {
        "canonical_energy": "canonical_energy",
        "electron_count": "electron_count",
        "fermi_level": "fermi_level_eV",
        "differential_capacitance": "differential_capacitance_e_per_V",
        "energy_uncertainty": "energy_standard_uncertainty_eV",
        "reaction_propensity": "reaction_propensity",
        "electron_solver_converged": "electron_solver_converged",
    }
    if getattr(model.backbone.config, "use_charge_head", False) or getattr(
        model.backbone.config, "use_qeq", False
    ):
        extra_properties["charges"] = "charges"
    return TorchPotentialCalculator(
        wrapped,
        device=resolved_device,
        dtype=dtype,
        input_adapter=adapter,
        energy_key="energy",
        forces_key="forces",
        extra_properties=extra_properties,
        compute_forces=True,
        stress_mode="finite_difference" if finite_difference_stress else "none",
        compile_model=compile_model,
    )


def load_constant_potential_checkpoint(
    backbone: Any,
    checkpoint: str | Path,
    *,
    config: Any | None = None,
    device: str = "cpu",
    dtype: str | None = None,
) -> ConstantPotentialJouleWeave:
    """Restore a constant-potential wrapper around a compatible backbone."""

    payload = load_checkpoint(checkpoint, map_location=device)
    model = ConstantPotentialJouleWeave(backbone, config=config)
    state = payload.get("model", payload.get("model_state", payload))
    model.load_state_dict(state)
    resolved = resolve_device(device)
    if dtype is None:
        model.to(resolved)
    else:
        import torch

        try:
            resolved_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype {dtype!r}") from exc
        model.to(device=resolved, dtype=resolved_dtype)
    model.eval()
    return model


__all__ = [
    "constant_potential_input_adapter",
    "constant_potential_jouleweave_calculator",
    "load_constant_potential_checkpoint",
]
