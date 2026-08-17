"""P2D state and per-step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .numerics import spherical_average
from .parameters import P2DParameters


@dataclass(slots=True)
class P2DStepDiagnostics:
    converged: bool
    potential_evaluations: int
    potential_residual_norm: float
    coupling_iterations: int
    coupling_error: float
    electrolyte_inventory_error: float
    negative_inventory_error: float
    positive_inventory_error: float
    message: str = ""


@dataclass(slots=True)
class P2DState:
    time_s: float
    electrolyte_concentration_mol_m3: np.ndarray
    negative_particle_concentration_mol_m3: np.ndarray
    positive_particle_concentration_mol_m3: np.ndarray
    electrolyte_potential_V: np.ndarray
    negative_solid_potential_V: np.ndarray
    positive_solid_potential_V: np.ndarray
    negative_interfacial_current_A_m2: np.ndarray
    positive_interfacial_current_A_m2: np.ndarray
    temperature_K: float
    terminal_voltage_V: float
    current_A: float = 0.0
    diagnostics: P2DStepDiagnostics | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def copy(self) -> P2DState:
        return P2DState(
            time_s=float(self.time_s),
            electrolyte_concentration_mol_m3=self.electrolyte_concentration_mol_m3.copy(),
            negative_particle_concentration_mol_m3=(
                self.negative_particle_concentration_mol_m3.copy()
            ),
            positive_particle_concentration_mol_m3=(
                self.positive_particle_concentration_mol_m3.copy()
            ),
            electrolyte_potential_V=self.electrolyte_potential_V.copy(),
            negative_solid_potential_V=self.negative_solid_potential_V.copy(),
            positive_solid_potential_V=self.positive_solid_potential_V.copy(),
            negative_interfacial_current_A_m2=(
                self.negative_interfacial_current_A_m2.copy()
            ),
            positive_interfacial_current_A_m2=(
                self.positive_interfacial_current_A_m2.copy()
            ),
            temperature_K=float(self.temperature_K),
            terminal_voltage_V=float(self.terminal_voltage_V),
            current_A=float(self.current_A),
            diagnostics=self.diagnostics,
            metadata=self.metadata.copy(),
        )

    def validate(self, parameters: P2DParameters) -> None:
        grid = parameters.discretization
        total = grid.negative_cells + grid.separator_cells + grid.positive_cells
        expected = {
            "electrolyte_concentration_mol_m3": (total,),
            "negative_particle_concentration_mol_m3": (
                grid.negative_cells,
                grid.negative_particle_cells,
            ),
            "positive_particle_concentration_mol_m3": (
                grid.positive_cells,
                grid.positive_particle_cells,
            ),
            "electrolyte_potential_V": (total,),
            "negative_solid_potential_V": (grid.negative_cells,),
            "positive_solid_potential_V": (grid.positive_cells,),
            "negative_interfacial_current_A_m2": (grid.negative_cells,),
            "positive_interfacial_current_A_m2": (grid.positive_cells,),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        if np.min(self.electrolyte_concentration_mol_m3) <= (
            parameters.minimum_concentration_mol_m3
        ):
            raise ValueError("electrolyte concentration must remain positive")
        if (
            np.min(self.negative_particle_concentration_mol_m3) <= 0.0
            or np.max(self.negative_particle_concentration_mol_m3)
            >= parameters.negative.maximum_concentration_mol_m3
        ):
            raise ValueError("negative particle concentration must remain inside (0, c_max)")
        if (
            np.min(self.positive_particle_concentration_mol_m3) <= 0.0
            or np.max(self.positive_particle_concentration_mol_m3)
            >= parameters.positive.maximum_concentration_mol_m3
        ):
            raise ValueError("positive particle concentration must remain inside (0, c_max)")
        if self.temperature_K <= 0.0 or not np.isfinite(self.terminal_voltage_V):
            raise ValueError("P2D temperature or voltage is invalid")

    def negative_stoichiometry(self, parameters: P2DParameters) -> float:
        return float(
            np.mean(spherical_average(self.negative_particle_concentration_mol_m3))
            / parameters.negative.maximum_concentration_mol_m3
        )

    def positive_stoichiometry(self, parameters: P2DParameters) -> float:
        return float(
            np.mean(spherical_average(self.positive_particle_concentration_mol_m3))
            / parameters.positive.maximum_concentration_mol_m3
        )

    def soc(self, parameters: P2DParameters) -> float:
        theta = self.negative_stoichiometry(parameters)
        lower = parameters.negative.stoichiometry_at_soc0
        upper = parameters.negative.stoichiometry_at_soc1
        return float(np.clip((theta - lower) / (upper - lower), 0.0, 1.0))


__all__ = ["P2DState", "P2DStepDiagnostics"]
