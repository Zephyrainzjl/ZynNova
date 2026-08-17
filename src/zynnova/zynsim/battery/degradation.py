"""Mechanistic SEI-growth and lithium-plating side-reaction states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import FARADAY, GAS_CONSTANT


@dataclass(frozen=True, slots=True)
class SEIParameters:
    reference_exchange_current_A_m2: float = 1.0e-6
    equilibrium_potential_V: float = 0.4
    charge_transfer_coefficient: float = 0.5
    activation_energy_J_mol: float = 38000.0
    reference_temperature_K: float = 298.15
    molar_volume_m3_mol: float = 9.585e-5
    electrons_per_molecule: float = 2.0
    initial_thickness_m: float = 5.0e-9
    transport_length_m: float = 2.0e-8
    ionic_conductivity_S_m: float = 1.0e-6

    def __post_init__(self) -> None:
        values = (
            self.reference_exchange_current_A_m2,
            self.molar_volume_m3_mol,
            self.electrons_per_molecule,
            self.initial_thickness_m,
            self.transport_length_m,
            self.ionic_conductivity_S_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("SEI kinetic/transport values must be positive")
        if not 0.0 < self.charge_transfer_coefficient < 1.0:
            raise ValueError("SEI transfer coefficient must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class PlatingParameters:
    exchange_current_A_m2: float = 1.0e-3
    charge_transfer_coefficient: float = 0.5
    equilibrium_potential_V: float = 0.0
    stripping_rate_s: float = 1.0e-4
    coulombic_efficiency: float = 0.98

    def __post_init__(self) -> None:
        if self.exchange_current_A_m2 <= 0.0 or self.stripping_rate_s < 0.0:
            raise ValueError("plating kinetics are invalid")
        if not 0.0 < self.charge_transfer_coefficient < 1.0:
            raise ValueError("plating transfer coefficient must lie in (0, 1)")
        if not 0.0 <= self.coulombic_efficiency <= 1.0:
            raise ValueError("plating coulombic efficiency must lie in [0, 1]")


@dataclass(slots=True)
class DegradationState:
    sei_thickness_m: np.ndarray
    plated_lithium_mol_m2: np.ndarray
    lost_lithium_C_m2: np.ndarray
    sei_current_A_m2: np.ndarray
    plating_current_A_m2: np.ndarray

    @classmethod
    def initialize(
        cls, count: int, sei_parameters: SEIParameters | None = None
    ) -> DegradationState:
        if count < 1:
            raise ValueError("degradation state count must be positive")
        parameters = sei_parameters or SEIParameters()
        zeros = np.zeros(count, dtype=np.float64)
        return cls(
            sei_thickness_m=np.full(count, parameters.initial_thickness_m),
            plated_lithium_mol_m2=zeros.copy(),
            lost_lithium_C_m2=zeros.copy(),
            sei_current_A_m2=zeros.copy(),
            plating_current_A_m2=zeros.copy(),
        )

    def film_resistance_ohm_m2(self, parameters: SEIParameters) -> np.ndarray:
        return self.sei_thickness_m / parameters.ionic_conductivity_S_m


class DegradationModel:
    """Advance SEI and reversible/irreversible plating with local overpotential."""

    def __init__(
        self,
        sei: SEIParameters | None = None,
        plating: PlatingParameters | None = None,
    ) -> None:
        self.sei = sei or SEIParameters()
        self.plating = plating or PlatingParameters()

    def step(
        self,
        state: DegradationState,
        negative_electrode_potential_V: np.ndarray,
        electrolyte_potential_V: np.ndarray,
        temperature_K: float | np.ndarray,
        dt_s: float,
    ) -> DegradationState:
        if dt_s <= 0.0:
            raise ValueError("degradation time step must be positive")
        phi_s = np.asarray(negative_electrode_potential_V, dtype=np.float64)
        phi_e = np.asarray(electrolyte_potential_V, dtype=np.float64)
        temperature = np.broadcast_to(np.asarray(temperature_K, dtype=np.float64), phi_s.shape)
        if (
            phi_s.shape != state.sei_thickness_m.shape
            or phi_e.shape != phi_s.shape
            or np.any(temperature <= 0.0)
        ):
            raise ValueError("degradation fields must align and temperature must be positive")

        sei_exchange = self.sei.reference_exchange_current_A_m2 * np.exp(
            -self.sei.activation_energy_J_mol
            / GAS_CONSTANT
            * (1.0 / temperature - 1.0 / self.sei.reference_temperature_K)
        )
        sei_overpotential = phi_s - phi_e - self.sei.equilibrium_potential_V
        kinetic_sei = sei_exchange * np.exp(
            np.clip(
                -self.sei.charge_transfer_coefficient
                * FARADAY
                * sei_overpotential
                / (GAS_CONSTANT * temperature),
                -80.0,
                80.0,
            )
        )
        sei_current = kinetic_sei / (
            1.0 + state.sei_thickness_m / self.sei.transport_length_m
        )
        thickness = state.sei_thickness_m + dt_s * (
            self.sei.molar_volume_m3_mol
            * sei_current
            / (self.sei.electrons_per_molecule * FARADAY)
        )

        plating_overpotential = phi_s - phi_e - self.plating.equilibrium_potential_V
        plating_drive = np.maximum(
            np.exp(
                np.clip(
                    -self.plating.charge_transfer_coefficient
                    * FARADAY
                    * plating_overpotential
                    / (GAS_CONSTANT * temperature),
                    -80.0,
                    80.0,
                )
            )
            - 1.0,
            0.0,
        )
        plating_current = self.plating.exchange_current_A_m2 * plating_drive
        stripping_mol_m2_s = np.minimum(
            self.plating.stripping_rate_s * state.plated_lithium_mol_m2,
            np.maximum(plating_overpotential, 0.0)
            * self.plating.exchange_current_A_m2
            / FARADAY,
        )
        plated_increment = dt_s * (
            plating_current / FARADAY - stripping_mol_m2_s
        )
        plated = np.maximum(state.plated_lithium_mol_m2 + plated_increment, 0.0)
        irreversible_plating = (
            1.0 - self.plating.coulombic_efficiency
        ) * plating_current
        lost = state.lost_lithium_C_m2 + dt_s * (
            sei_current + irreversible_plating
        )
        return DegradationState(
            sei_thickness_m=thickness,
            plated_lithium_mol_m2=plated,
            lost_lithium_C_m2=lost,
            sei_current_A_m2=sei_current,
            plating_current_A_m2=plating_current,
        )


__all__ = [
    "DegradationModel",
    "DegradationState",
    "PlatingParameters",
    "SEIParameters",
]
