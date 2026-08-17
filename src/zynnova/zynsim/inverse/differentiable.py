"""End-to-end differentiable reduced electrochemical multiphysics twin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "differentiable inversion requires PyTorch; install zynnova[zynsim-inverse]"
        ) from exc
    return torch


@dataclass(slots=True)
class DifferentiableSolverConfig:
    initial_soc: float = 1.0
    initial_temperature_K: float = 298.15
    ambient_temperature_K: float = 298.15
    initial_soh: float = 1.0
    substeps: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_soc <= 1.0 or self.initial_temperature_K <= 0.0:
            raise ValueError("initial differentiable state is invalid")
        if not 0.0 < self.initial_soh <= 1.0 or self.substeps < 1:
            raise ValueError("initial SOH/substeps are invalid")


class DifferentiableBatteryParameters:
    """Factory for a torch module containing constrained physical parameters."""

    def __new__(cls, initial: Mapping[str, float] | None = None):
        torch = _torch()
        nn = torch.nn
        defaults = {
            "capacity_Ah": 3.0,
            "ohmic_resistance_ohm": 0.025,
            "charge_transfer_resistance_ohm": 0.015,
            "double_layer_capacitance_F": 1500.0,
            "warburg_ohm_sqrt_s": 0.01,
            "thermal_capacity_J_K": 900.0,
            "cooling_W_K": 1.5,
            "entropic_V_K": 0.0,
            "expansion_per_soc": 0.06,
            "sei_rate_sqrt_s_inv": 2.0e-5,
            "damage_rate_s_inv": 1.0e-7,
            "plating_rate_s_inv": 1.0e-8,
            "ocv_0_V": 3.0,
            "ocv_1_V": 1.2,
            "ocv_2_V": -0.1,
        }
        defaults.update(dict(initial or {}))
        positive = {
            "capacity_Ah",
            "ohmic_resistance_ohm",
            "charge_transfer_resistance_ohm",
            "double_layer_capacitance_F",
            "warburg_ohm_sqrt_s",
            "thermal_capacity_J_K",
            "cooling_W_K",
            "expansion_per_soc",
            "sei_rate_sqrt_s_inv",
            "damage_rate_s_inv",
            "plating_rate_s_inv",
        }

        class _Parameters(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.positive_names = positive
                self.raw = nn.ParameterDict()
                for name, value in defaults.items():
                    tensor = torch.tensor(float(value), dtype=torch.get_default_dtype())
                    if name in positive:
                        value_tensor = tensor.clamp_min(1.0e-12)
                        tensor = torch.where(
                            value_tensor > 20.0,
                            value_tensor,
                            torch.log(torch.expm1(value_tensor)),
                        )
                    self.raw[name] = nn.Parameter(tensor)

            def physical(self) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for name, value in self.raw.items():
                    result[name] = torch.nn.functional.softplus(value) if name in self.positive_names else value
                return result

        return _Parameters()


class DifferentiableBatterySolver:
    """Factory returning a differentiable voltage/EIS/thermal/swelling/aging model."""

    def __new__(
        cls,
        parameters: Any | None = None,
        config: DifferentiableSolverConfig | None = None,
    ):
        torch = _torch()
        nn = torch.nn
        resolved = config or DifferentiableSolverConfig()
        parameter_module = parameters or DifferentiableBatteryParameters()

        class _Solver(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameters_model = parameter_module
                self.config = resolved

            def forward(self, time_s: Any, current_A: Any) -> dict[str, Any]:
                time = torch.as_tensor(time_s)
                current = torch.as_tensor(current_A, device=time.device, dtype=time.dtype)
                if time.ndim != 1 or current.shape != time.shape or len(time) < 2:
                    raise ValueError("time and current must be aligned one-dimensional tensors")
                p = self.parameters_model.physical()
                soc = time.new_tensor(self.config.initial_soc)
                temperature = time.new_tensor(self.config.initial_temperature_K)
                polarization = time.new_zeros(())
                sei = time.new_zeros(())
                damage = time.new_zeros(())
                plated = time.new_zeros(())
                soh = time.new_tensor(self.config.initial_soh)
                outputs = {
                    "soc": [soc],
                    "temperature_K": [temperature],
                    "polarization_V": [polarization],
                    "sei_state": [sei],
                    "damage": [damage],
                    "plated_lithium": [plated],
                    "soh": [soh],
                    "voltage_V": [self._ocv(soc, p)],
                    "expansion": [p["expansion_per_soc"] * (1.0 - soc)],
                }
                for index in range(len(time) - 1):
                    interval = time[index + 1] - time[index]
                    dt = interval / self.config.substeps
                    applied = current[index]
                    for _ in range(self.config.substeps):
                        effective_capacity = p["capacity_Ah"] * soh.clamp_min(1.0e-4)
                        soc = torch.clamp(
                            soc - applied * dt / (3600.0 * effective_capacity),
                            0.0,
                            1.0,
                        )
                        tau = p["charge_transfer_resistance_ohm"] * p["double_layer_capacitance_F"]
                        decay = torch.exp(-dt / tau.clamp_min(1.0e-8))
                        polarization = decay * polarization + p["charge_transfer_resistance_ohm"] * applied * (1.0 - decay)
                        ocv = self._ocv(soc, p)
                        voltage = ocv - applied * p["ohmic_resistance_ohm"] - polarization
                        heat = applied.square() * p["ohmic_resistance_ohm"] + polarization.square() / p["charge_transfer_resistance_ohm"].clamp_min(1.0e-8)
                        heat = heat + applied * temperature * p["entropic_V_K"]
                        temperature = temperature + dt * (
                            heat + p["cooling_W_K"] * (self.config.ambient_temperature_K - temperature)
                        ) / p["thermal_capacity_J_K"]
                        sei_increment = p["sei_rate_sqrt_s_inv"] * (
                            torch.sqrt((time[index] + dt).clamp_min(0.0) + 1.0)
                            - torch.sqrt(time[index].clamp_min(0.0) + 1.0)
                        ) * torch.exp(0.03 * (temperature - 298.15))
                        plating_drive = torch.relu(-voltage)
                        plated = plated + dt * p["plating_rate_s_inv"] * plating_drive
                        damage = torch.clamp(
                            damage + dt * p["damage_rate_s_inv"] * torch.relu(torch.abs(applied) - effective_capacity),
                            0.0,
                            1.0,
                        )
                        sei = sei + sei_increment
                        soh = torch.clamp(1.0 - sei - plated - 0.5 * damage, 0.0, 1.0)
                    outputs["soc"].append(soc)
                    outputs["temperature_K"].append(temperature)
                    outputs["polarization_V"].append(polarization)
                    outputs["sei_state"].append(sei)
                    outputs["damage"].append(damage)
                    outputs["plated_lithium"].append(plated)
                    outputs["soh"].append(soh)
                    outputs["voltage_V"].append(voltage)
                    outputs["expansion"].append(
                        p["expansion_per_soc"] * (1.0 - soc) * (1.0 + damage)
                    )
                result = {name: torch.stack(values) for name, values in outputs.items()}
                result["image_features"] = torch.stack(
                    (result["sei_state"], result["damage"], result["expansion"]),
                    dim=-1,
                )
                return result

            def impedance(self, frequency_Hz: Any, *, state: Mapping[str, Any] | None = None) -> Any:
                frequency = torch.as_tensor(frequency_Hz)
                p = self.parameters_model.physical()
                omega = 2.0 * torch.pi * frequency
                imaginary = torch.complex(torch.zeros_like(omega), omega)
                charge_transfer = p["charge_transfer_resistance_ohm"] / (
                    1.0 + imaginary * p["charge_transfer_resistance_ohm"] * p["double_layer_capacitance_F"]
                )
                regularized_frequency = imaginary + torch.complex(
                    torch.full_like(omega, 1.0e-20),
                    torch.zeros_like(omega),
                )
                warburg = p["warburg_ohm_sqrt_s"] / torch.sqrt(
                    regularized_frequency
                )
                degradation = 1.0
                if state is not None and "soh" in state:
                    degradation = 1.0 / torch.as_tensor(state["soh"]).reshape(-1)[-1].clamp_min(1.0e-3)
                return degradation * (
                    p["ohmic_resistance_ohm"] + charge_transfer + warburg
                )

            @staticmethod
            def _ocv(soc: Any, p: Mapping[str, Any]) -> Any:
                return p["ocv_0_V"] + p["ocv_1_V"] * soc + p["ocv_2_V"] * soc.square()

        return _Solver()


__all__ = [
    "DifferentiableBatteryParameters",
    "DifferentiableBatterySolver",
    "DifferentiableSolverConfig",
]
