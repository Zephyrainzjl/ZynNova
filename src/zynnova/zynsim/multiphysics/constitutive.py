"""Constitutive kernels for heat, swelling stress, and electrochemical aging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import FARADAY, GAS_CONSTANT


@dataclass(slots=True)
class CoupledMaterialConfig:
    volumetric_heat_capacity_J_m3_K: float = 2.5e6
    thermal_conductivity_W_m_K: float = 1.0
    cooling_coefficient_W_m3_K: float = 2.0e4
    ambient_temperature_K: float = 298.15
    internal_resistance_ohm: float = 0.02
    entropic_coefficient_V_K: float = 0.0
    young_modulus_Pa: float = 5.0e9
    poisson_ratio: float = 0.30
    chemical_expansion_coefficient: float = 0.06
    reference_soc: float = 0.5
    sei_rate_m_sqrt_s: float = 2.0e-10
    cei_rate_m_sqrt_s: float = 5.0e-11
    plating_exchange_current_A_m2: float = 0.0
    plating_equilibrium_V: float = 0.0
    active_material_loss_rate_s: float = 1.0e-8
    lithium_inventory_loss_per_m_SEI: float = 1.0e5

    def __post_init__(self) -> None:
        positive = (
            self.volumetric_heat_capacity_J_m3_K,
            self.thermal_conductivity_W_m_K,
            self.cooling_coefficient_W_m3_K,
            self.young_modulus_Pa,
        )
        if min(positive) <= 0.0:
            raise ValueError("thermal/mechanical material values must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must lie in (-1,0.5)")
        if min(
            self.sei_rate_m_sqrt_s,
            self.cei_rate_m_sqrt_s,
            self.plating_exchange_current_A_m2,
            self.active_material_loss_rate_s,
            self.lithium_inventory_loss_per_m_SEI,
        ) < 0.0:
            raise ValueError("aging rates cannot be negative")


class CoupledConstitutiveModel:
    def __init__(self, config: CoupledMaterialConfig | None = None) -> None:
        self.config = config or CoupledMaterialConfig()

    def heat_source_W_m3(
        self,
        *,
        current_A: float,
        terminal_voltage_V: float,
        open_circuit_voltage_V: float | None,
        temperature_K: np.ndarray,
        active_volume_m3: float,
    ) -> np.ndarray:
        if active_volume_m3 <= 0.0:
            raise ValueError("active volume must be positive")
        temperature = np.asarray(temperature_K, dtype=float)
        irreversible = current_A**2 * self.config.internal_resistance_ohm
        reaction = 0.0
        if open_circuit_voltage_V is not None:
            reaction = abs(current_A * (open_circuit_voltage_V - terminal_voltage_V))
        entropic = current_A * np.mean(temperature) * self.config.entropic_coefficient_V_K
        total = (irreversible + reaction + entropic) / active_volume_m3
        return np.full_like(temperature, float(total))

    def thermal_step(
        self,
        temperature_K: np.ndarray,
        heat_source_W_m3: np.ndarray,
        dt_s: float,
        *,
        laplacian: np.ndarray | None = None,
    ) -> np.ndarray:
        temperature = np.asarray(temperature_K, dtype=float)
        source = np.broadcast_to(np.asarray(heat_source_W_m3, dtype=float), temperature.shape)
        conduction = 0.0
        if laplacian is not None:
            matrix = np.asarray(laplacian, dtype=float)
            flat = temperature.reshape(-1)
            if matrix.shape != (flat.size, flat.size):
                raise ValueError("thermal Laplacian shape is inconsistent")
            conduction = self.config.thermal_conductivity_W_m_K * (matrix @ flat).reshape(temperature.shape)
        cooling = self.config.cooling_coefficient_W_m3_K * (
            self.config.ambient_temperature_K - temperature
        )
        updated = temperature + dt_s * (source + conduction + cooling) / self.config.volumetric_heat_capacity_J_m3_K
        if np.any(updated <= 0.0) or not np.isfinite(updated).all():
            raise ValueError("thermal update produced invalid temperature")
        return updated

    def mechanical_response(
        self,
        soc: float,
        damage: np.ndarray,
        *,
        constraint_factor: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        damage_values = np.asarray(damage, dtype=float)
        if np.any((damage_values < 0.0) | (damage_values > 1.0)):
            raise ValueError("damage must lie in [0,1]")
        strain = self.config.chemical_expansion_coefficient * (
            float(soc) - self.config.reference_soc
        )
        effective_modulus = self.config.young_modulus_Pa * (1.0 - damage_values) ** 2
        stress_scalar = (
            constraint_factor
            * effective_modulus
            * strain
            / max(1.0 - self.config.poisson_ratio, 1.0e-6)
        )
        stress = np.zeros(damage_values.shape + (6,), dtype=float)
        stress[..., :3] = stress_scalar[..., None]
        displacement = np.zeros(damage_values.shape + (3,), dtype=float)
        displacement[..., 0] = strain * (1.0 - damage_values)
        return stress, displacement

    def aging_step(
        self,
        *,
        sei_thickness_m: np.ndarray,
        cei_thickness_m: np.ndarray,
        plated_lithium_mol: np.ndarray,
        active_material_fraction: np.ndarray,
        lithium_inventory_fraction: float,
        time_s: float,
        dt_s: float,
        temperature_K: np.ndarray,
        anode_overpotential_V: float,
        damage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        temperature = np.asarray(temperature_K, dtype=float)
        arrhenius = np.exp(
            -2.0e4 / GAS_CONSTANT * (1.0 / np.mean(temperature) - 1.0 / 298.15)
        )
        sqrt_increment = np.sqrt(max(time_s + dt_s, 0.0)) - np.sqrt(max(time_s, 0.0))
        sei_increment = self.config.sei_rate_m_sqrt_s * arrhenius * sqrt_increment
        cei_increment = self.config.cei_rate_m_sqrt_s * arrhenius * sqrt_increment
        sei_new = np.asarray(sei_thickness_m, dtype=float) + sei_increment
        cei_new = np.asarray(cei_thickness_m, dtype=float) + cei_increment
        plating_current = self.config.plating_exchange_current_A_m2 * np.sinh(
            FARADAY * (self.config.plating_equilibrium_V - anode_overpotential_V)
            / (2.0 * GAS_CONSTANT * np.mean(temperature))
        )
        plated_new = np.asarray(plated_lithium_mol, dtype=float) + max(plating_current, 0.0) * dt_s / FARADAY
        active_new = np.asarray(active_material_fraction, dtype=float) * np.exp(
            -self.config.active_material_loss_rate_s
            * (1.0 + np.asarray(damage, dtype=float))
            * dt_s
        )
        inventory_loss = self.config.lithium_inventory_loss_per_m_SEI * float(sei_increment)
        inventory_new = float(np.clip(lithium_inventory_fraction - inventory_loss, 0.0, 1.0))
        return sei_new, cei_new, plated_new, active_new, inventory_new


__all__ = ["CoupledConstitutiveModel", "CoupledMaterialConfig"]
