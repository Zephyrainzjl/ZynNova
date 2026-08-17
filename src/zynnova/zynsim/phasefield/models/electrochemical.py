"""Electrochemical intercalation, electrodeposition, and reactive phase-field models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..fields import FieldSpec
from .base import PhaseFieldModel, _clip


@dataclass(slots=True)
class IntercalationPhaseFieldModel(PhaseFieldModel):
    """Regular-solution intercalation model with reaction and stress coupling.

    The model can represent phase-separating electrode particles at arbitrary SOC.
    ``mechanical_chemical_potential`` may be supplied by ZynSim mechanics to inject
    hydrostatic-stress or coherency-strain contributions.
    """

    mobility: float = 1.0
    gradient_coefficient: float = 1.0
    regular_solution_parameter: float = 2.5
    thermal_energy: float = 1.0
    reaction_rate: float = 0.0
    applied_overpotential: float | Callable[[Any, float], Any] = 0.0
    charge_transfer_coefficient: float = 0.5
    mechanical_chemical_potential: Callable[[Any, float], Any] | None = None
    field_name: str = "lithium_fraction"
    name: str = "intercalation-phase-field"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (
            FieldSpec(
                self.field_name,
                conserved=True,
                lower_bound=1.0e-8,
                upper_bound=1.0 - 1.0e-8,
                description="site fraction of intercalated lithium",
            ),
        )

    def chemical_potential(self, concentration, operators, xp, time):
        c = _clip(xp, concentration, 1.0e-8, 1.0 - 1.0e-8)
        mu = (
            self.thermal_energy * (xp.log(c) - xp.log(1.0 - c))
            + self.regular_solution_parameter * (1.0 - 2.0 * c)
            - self.gradient_coefficient * operators.laplacian(c)
        )
        if self.mechanical_chemical_potential is not None:
            mu = mu + self.mechanical_chemical_potential(c, time)
        return mu

    def _overpotential(self, concentration, time):
        if callable(self.applied_overpotential):
            return self.applied_overpotential(concentration, time)
        return self.applied_overpotential

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid
        c = fields[self.field_name]
        mu = self.chemical_potential(c, operators, xp, time)
        transport = self.mobility * operators.laplacian(mu)
        if self.reaction_rate == 0.0:
            reaction = 0.0
        else:
            eta = self._overpotential(c, time) - mu
            alpha = self.charge_transfer_coefficient
            reaction = self.reaction_rate * (
                xp.exp(alpha * eta) - xp.exp(-(1.0 - alpha) * eta)
            )
        return {self.field_name: transport + reaction}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        return -self.mobility * self.gradient_coefficient * k2**2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        c = _clip(xp, fields[self.field_name], 1.0e-8, 1.0 - 1.0e-8)
        return (
            self.thermal_energy * (c * xp.log(c) + (1.0 - c) * xp.log(1.0 - c))
            + self.regular_solution_parameter * c * (1.0 - c)
            + 0.5 * self.gradient_coefficient * operators.grad_squared(c)
        )


@dataclass(slots=True)
class ElectrodepositionModel(PhaseFieldModel):
    """Coupled metal phase, ionic concentration, and electrostatic potential model."""

    phase_mobility: float = 1.0
    ion_mobility: float = 1.0
    electrical_relaxation: float = 5.0
    interface_gradient: float = 1.0
    barrier: float = 1.0
    reaction_rate: float = 1.0
    charge_transfer_coefficient: float = 0.5
    ion_charge: float = 1.0
    permittivity: float = 1.0
    applied_potential: float = 0.0
    phase_name: str = "metal"
    concentration_name: str = "ion_concentration"
    potential_name: str = "electric_potential"
    name: str = "electrodeposition"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (
            FieldSpec(self.phase_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
            FieldSpec(self.concentration_name, conserved=True, lower_bound=1.0e-10),
            FieldSpec(self.potential_name, conserved=False),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        eta = fields[self.phase_name]
        c = _clip(xp, fields[self.concentration_name], 1.0e-10, 1.0e30)
        potential = fields[self.potential_name]
        interpolation = eta**3 * (10.0 - 15.0 * eta + 6.0 * eta**2)
        derivative = 30.0 * eta**2 * (1.0 - eta) ** 2
        overpotential = self.applied_potential - potential - xp.log(c)
        alpha = self.charge_transfer_coefficient
        faradaic = self.reaction_rate * c * (
            xp.exp(alpha * overpotential) - xp.exp(-(1.0 - alpha) * overpotential)
        )
        phase_drive = (
            2.0 * self.barrier * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
            - self.interface_gradient * operators.laplacian(eta)
            - derivative * faradaic
        )
        phase_rate = -self.phase_mobility * phase_drive
        electrochemical_potential = xp.log(c) + self.ion_charge * potential
        ion_rate = self.ion_mobility * operators.laplacian(electrochemical_potential) - phase_rate
        charge_density = self.ion_charge * c * (1.0 - interpolation)
        potential_rate = self.electrical_relaxation * (
            self.permittivity * operators.laplacian(potential) + charge_density
        )
        return {
            self.phase_name: phase_rate,
            self.concentration_name: ion_rate,
            self.potential_name: potential_rate,
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.phase_name:
            return -self.phase_mobility * self.interface_gradient * k2
        if field_name == self.concentration_name:
            return -self.ion_mobility * k2
        if field_name == self.potential_name:
            return -self.electrical_relaxation * self.permittivity * k2
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        eta = fields[self.phase_name]
        c = _clip(xp, fields[self.concentration_name], 1.0e-10, 1.0e30)
        potential = fields[self.potential_name]
        return (
            self.barrier * eta**2 * (1.0 - eta) ** 2
            + 0.5 * self.interface_gradient * operators.grad_squared(eta)
            + c * (xp.log(c) - 1.0)
            + self.ion_charge * c * potential
            + 0.5 * self.permittivity * operators.grad_squared(potential)
        )


@dataclass(slots=True)
class ElectrochemicalReactionPhaseFieldModel(PhaseFieldModel):
    """Generic conserved species with nonlinear electrochemical reaction kinetics."""

    diffusivity: float = 1.0
    gradient_coefficient: float = 0.1
    reaction_rate: float = 1.0
    equilibrium_potential: float = 0.0
    applied_potential: float = 0.0
    symmetry_factor: float = 0.5
    species_name: str = "c"
    name: str = "electrochemical-reaction"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (FieldSpec(self.species_name, conserved=True, lower_bound=1.0e-8, upper_bound=1.0 - 1.0e-8),)

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        c = _clip(xp, fields[self.species_name], 1.0e-8, 1.0 - 1.0e-8)
        mu = xp.log(c) - xp.log(1.0 - c) - self.gradient_coefficient * operators.laplacian(c)
        eta = self.applied_potential - self.equilibrium_potential - mu
        alpha = self.symmetry_factor
        reaction = self.reaction_rate * (
            (1.0 - c) * xp.exp(alpha * eta) - c * xp.exp(-(1.0 - alpha) * eta)
        )
        return {self.species_name: self.diffusivity * operators.laplacian(mu) + reaction}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        return -self.diffusivity * self.gradient_coefficient * k2**2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        c = _clip(xp, fields[self.species_name], 1.0e-8, 1.0 - 1.0e-8)
        return c * xp.log(c) + (1.0 - c) * xp.log(1.0 - c) + 0.5 * self.gradient_coefficient * operators.grad_squared(c)


__all__ = [
    "ElectrochemicalReactionPhaseFieldModel",
    "ElectrodepositionModel",
    "IntercalationPhaseFieldModel",
]
