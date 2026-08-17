"""Pure-material, alloy, dendritic, and anisotropic solidification phase fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import FieldSpec
from .base import PhaseFieldModel, _clip


@dataclass(slots=True)
class DendriticSolidificationModel(PhaseFieldModel):
    """Thermal dendritic solidification with orientation-dependent interface width."""

    phase_mobility: float = 1.0
    thermal_diffusivity: float = 1.0
    interface_width: float = 1.0
    anisotropy_strength: float = 0.04
    symmetry_order: int = 4
    orientation_offset: float = 0.0
    latent_heat: float = 1.0
    thermal_coupling: float = 2.0
    phase_name: str = "phi"
    temperature_name: str = "temperature"
    name: str = "dendritic-solidification"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (
            FieldSpec(self.phase_name, conserved=False, lower_bound=-1.0, upper_bound=1.0),
            FieldSpec(self.temperature_name, conserved=True),
        )

    def _anisotropy(self, phi, operators, xp):
        gradient = operators.gradient(phi)
        if len(gradient) == 1:
            return 1.0
        if getattr(xp, "__name__", "").startswith("torch"):
            angle = xp.atan2(gradient[1], gradient[0] + 1.0e-14)
            cosine = xp.cos
        else:
            angle = xp.arctan2(gradient[1], gradient[0] + 1.0e-14)
            cosine = xp.cos
        return self.interface_width * (
            1.0
            + self.anisotropy_strength
            * cosine(self.symmetry_order * (angle - self.orientation_offset))
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        phi = fields[self.phase_name]
        temperature = fields[self.temperature_name]
        epsilon = self._anisotropy(phi, operators, xp)
        driving = phi - phi**3 - self.thermal_coupling * temperature * (1.0 - phi**2) ** 2
        phase_rate = self.phase_mobility * (
            epsilon**2 * operators.laplacian(phi) + driving
        )
        temperature_rate = (
            self.thermal_diffusivity * operators.laplacian(temperature)
            + 0.5 * self.latent_heat * phase_rate
        )
        return {self.phase_name: phase_rate, self.temperature_name: temperature_rate}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.phase_name:
            return self.phase_mobility * (1.0 - self.interface_width**2 * k2)
        if field_name == self.temperature_name:
            return -self.thermal_diffusivity * k2
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        phi = fields[self.phase_name]
        temperature = fields[self.temperature_name]
        epsilon = self._anisotropy(phi, operators, xp)
        return (
            0.25 * (1.0 - phi**2) ** 2
            + 0.5 * epsilon**2 * operators.grad_squared(phi)
            + self.thermal_coupling * temperature * (phi - phi**3 / 3.0)
            + 0.5 * temperature**2
        )


@dataclass(slots=True)
class BinaryAlloySolidificationModel(PhaseFieldModel):
    """Diffuse-interface binary alloy solidification with solute partitioning."""

    phase_mobility: float = 1.0
    solute_mobility: float = 1.0
    thermal_diffusivity: float = 1.0
    phase_gradient: float = 1.0
    solute_gradient: float = 0.05
    barrier: float = 1.0
    partition_coefficient: float = 0.2
    liquidus_slope: float = -1.0
    latent_heat: float = 1.0
    phase_name: str = "phi"
    composition_name: str = "c"
    temperature_name: str = "temperature"
    name: str = "binary-alloy-solidification"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (
            FieldSpec(self.phase_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
            FieldSpec(self.composition_name, conserved=True, lower_bound=1.0e-8, upper_bound=1.0 - 1.0e-8),
            FieldSpec(self.temperature_name, conserved=True),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        phi = fields[self.phase_name]
        c = _clip(xp, fields[self.composition_name], 1.0e-8, 1.0 - 1.0e-8)
        temperature = fields[self.temperature_name]
        h = phi**3 * (10.0 - 15.0 * phi + 6.0 * phi**2)
        dh = 30.0 * phi**2 * (1.0 - phi) ** 2
        equilibrium_shift = self.liquidus_slope * c + temperature
        phase_derivative = (
            2.0 * self.barrier * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
            + dh * equilibrium_shift
            - self.phase_gradient * operators.laplacian(phi)
        )
        phase_rate = -self.phase_mobility * phase_derivative
        chemical_potential = (
            xp.log(c) - xp.log(1.0 - c)
            + h * xp.log(max(self.partition_coefficient, 1.0e-12))
            - self.solute_gradient * operators.laplacian(c)
        )
        composition_rate = self.solute_mobility * operators.laplacian(chemical_potential)
        temperature_rate = self.thermal_diffusivity * operators.laplacian(temperature) + self.latent_heat * phase_rate
        return {
            self.phase_name: phase_rate,
            self.composition_name: composition_rate,
            self.temperature_name: temperature_rate,
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.phase_name:
            return -self.phase_mobility * self.phase_gradient * k2
        if field_name == self.composition_name:
            return -self.solute_mobility * self.solute_gradient * k2**2
        if field_name == self.temperature_name:
            return -self.thermal_diffusivity * k2
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        phi = fields[self.phase_name]
        c = _clip(xp, fields[self.composition_name], 1.0e-8, 1.0 - 1.0e-8)
        temperature = fields[self.temperature_name]
        h = phi**3 * (10.0 - 15.0 * phi + 6.0 * phi**2)
        mixing = c * xp.log(c) + (1.0 - c) * xp.log(1.0 - c)
        return (
            self.barrier * phi**2 * (1.0 - phi) ** 2
            + mixing
            + h * (self.liquidus_slope * c + temperature)
            + 0.5 * self.phase_gradient * operators.grad_squared(phi)
            + 0.5 * self.solute_gradient * operators.grad_squared(c)
            + 0.5 * temperature**2
        )


__all__ = ["BinaryAlloySolidificationModel", "DendriticSolidificationModel"]
