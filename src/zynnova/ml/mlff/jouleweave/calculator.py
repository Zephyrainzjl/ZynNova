from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....dynamics import TorchPotentialCalculator
from ...common import load_checkpoint, resolve_device
from .config import jouleweave_model_config_from_dict
from .model import JouleWeave


def jouleweave_input_adapter(
    atoms: Any,
    *,
    positions: Any,
    cell: Any,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    import torch

    atomic_numbers = torch.as_tensor(
        atoms.get_atomic_numbers(),
        device=device,
        dtype=torch.long,
    )
    initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    total_charge = float(atoms.info.get("total_charge", initial_charges.sum()))
    inputs = {
        "z": atomic_numbers,
        "atomic_numbers": atomic_numbers,
        "pos": positions,
        "positions": positions,
        "cell": cell,
        "pbc": torch.as_tensor(
            np.asarray(atoms.pbc),
            device=device,
            dtype=torch.bool,
        ),
        "batch": torch.zeros(
            len(atomic_numbers),
            device=device,
            dtype=torch.long,
        ),
        "total_charge": torch.as_tensor(
            [total_charge],
            device=device,
            dtype=dtype,
        ),
    }
    if "spin" in atoms.info:
        inputs["spin"] = torch.as_tensor(
            [atoms.info["spin"]],
            device=device,
            dtype=dtype,
        )
    if "fidelity" in atoms.info:
        inputs["fidelity"] = torch.as_tensor(
            [atoms.info["fidelity"]],
            device=device,
            dtype=torch.long,
        )
    return inputs


def _inference_wrapper(model: JouleWeave, *, analytic_stress: bool):
    import torch

    class JouleWeaveInference(torch.nn.Module):
        def __init__(self, wrapped: JouleWeave) -> None:
            super().__init__()
            self.model = wrapped

        def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
            output = self.model.energy_forces_stress(
                inputs,
                create_graph=False,
                compute_stress=analytic_stress,
            )
            if "stress" in output and output["stress"].shape == (1, 6):
                output["stress"] = output["stress"][0]
            return output

    return JouleWeaveInference(model)


def jouleweave_calculator(
    model: JouleWeave,
    *,
    device: str = "auto",
    dtype: str = "float32",
    analytic_stress: bool = False,
    compile_model: bool = False,
) -> TorchPotentialCalculator:
    """Create an ASE/ZynNova calculator for energy-conserving MD.

    Set ``analytic_stress=True`` for NPT/cell relaxation. NVE/NVT simulations can
    leave it disabled to avoid differentiating with respect to a strain tensor.
    """

    resolved = resolve_device(device)
    import torch

    try:
        resolved_dtype = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {dtype}") from exc
    if not resolved_dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch dtype")
    model.to(device=resolved, dtype=resolved_dtype)
    wrapped = _inference_wrapper(
        model,
        analytic_stress=analytic_stress,
    )

    extra_properties: dict[str, str] = {}
    if model.config.use_magmoms:
        # 左边是 ASE 属性名称，右边是模型输出字典的键。
        extra_properties["magmoms"] = "magmoms"

    if model.config.use_charge_head or model.config.use_qeq:
        extra_properties.update(
            {
                "charges": "charges",
                "dipole": "dipole",
            }
        )
    if model.config.use_qeq:
        extra_properties["qeq_charges"] = "qeq_charges"
    if model.config.use_oxidation_states:
        extra_properties.update(
            {
                "oxidation_states": "oxidation_states",
                "oxidation_state_logits": "oxidation_state_logits",
                "oxidation_state_probabilities": "oxidation_state_probabilities",
            }
        )
    return TorchPotentialCalculator(
        wrapped,
        device=resolved,
        dtype=dtype,
        input_adapter=jouleweave_input_adapter,
        stress_mode="model" if analytic_stress else "none",
        extra_properties=extra_properties,
        compute_forces=True,
        compile_model=compile_model,
    )


def load_jouleweave(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    use_ema: bool = True,
) -> JouleWeave:
    payload = load_checkpoint(checkpoint, map_location=device)
    if payload.get("model_name") != "jouleweave":
        raise ValueError("checkpoint is not a JouleWeave model")
    config = jouleweave_model_config_from_dict(payload["model_config"])
    model = JouleWeave(config)
    state_key = "ema_model_state" if use_ema and "ema_model_state" in payload else "model_state"
    model.load_state_dict(payload[state_key])
    resolved = resolve_device(device)
    if dtype is None:
        model.to(resolved)
    else:
        import torch

        try:
            resolved_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype: {dtype}") from exc
        model.to(device=resolved, dtype=resolved_dtype)
    model.eval()
    return model


def load_jouleweave_calculator(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
    analytic_stress: bool = False,
    use_ema: bool = True,
) -> TorchPotentialCalculator:
    resolved = resolve_device(device)
    model = load_jouleweave(
        checkpoint,
        device=str(resolved),
        dtype=dtype,
        use_ema=use_ema,
    )
    return jouleweave_calculator(
        model,
        device=str(resolved),
        dtype=dtype,
        analytic_stress=analytic_stress,
    )


__all__ = [
    "jouleweave_calculator",
    "jouleweave_input_adapter",
    "load_jouleweave",
    "load_jouleweave_calculator",
]
