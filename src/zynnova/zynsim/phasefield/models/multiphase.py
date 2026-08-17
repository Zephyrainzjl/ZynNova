"""Multiphase, KKS, grand-potential, grain-growth, and sintering models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fields import FieldSpec
from .base import PhaseFieldModel, _clip


def _smoothstep(eta):
    return eta**3 * (10.0 - 15.0 * eta + 6.0 * eta**2)


def _smoothstep_derivative(eta):
    return 30.0 * eta**2 * (1.0 - eta) ** 2


@dataclass(slots=True)
class GrainGrowthModel(PhaseFieldModel):
    """Multi-order-parameter polycrystalline grain-growth model."""

    grains: int = 4
    mobility: float = 1.0
    barrier: float = 1.0
    cross_coupling: float = 2.0
    gradient_coefficient: float = 1.0
    prefix: str = "eta"
    name: str = "grain-growth"

    def __post_init__(self) -> None:
        if self.grains < 2:
            raise ValueError("grain growth requires at least two order parameters")

    @property
    def field_specs(self):
        return tuple(
            FieldSpec(f"{self.prefix}{index}", conserved=False, lower_bound=0.0, upper_bound=1.0)
            for index in range(self.grains)
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        squared_sum = 0.0
        for spec in self.field_specs:
            squared_sum = squared_sum + fields[spec.name] ** 2
        result = {}
        for spec in self.field_specs:
            eta = fields[spec.name]
            derivative = (
                self.barrier * (2.0 * eta - 6.0 * eta**2 + 4.0 * eta**3)
                + 2.0 * self.cross_coupling * eta * (squared_sum - eta**2)
                - self.gradient_coefficient * operators.laplacian(eta)
            )
            result[spec.name] = -self.mobility * derivative
        return result

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        return -self.mobility * (
            2.0 * self.barrier + self.gradient_coefficient * k2
        )

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        local = 0.0
        squared_fields = []
        gradient = 0.0
        for spec in self.field_specs:
            eta = fields[spec.name]
            squared_fields.append(eta**2)
            local = local + self.barrier * eta**2 * (1.0 - eta) ** 2
            gradient = gradient + 0.5 * self.gradient_coefficient * operators.grad_squared(eta)
        cross = 0.0
        for i in range(len(squared_fields)):
            for j in range(i + 1, len(squared_fields)):
                cross = cross + self.cross_coupling * squared_fields[i] * squared_fields[j]
        return local + cross + gradient

    def project(self, fields, *, xp=np):
        clipped = {name: _clip(xp, values, 0.0, 1.0) for name, values in fields.items()}
        total = 0.0
        for values in clipped.values():
            total = total + values
        epsilon = 1.0e-12
        return {name: values / (total + epsilon) for name, values in clipped.items()}


@dataclass(slots=True)
class MultiphaseFieldModel(GrainGrowthModel):
    """Simplex-constrained multiphase-field model with arbitrary phase count."""

    name: str = "multiphase-field"


@dataclass(slots=True)
class KKSModel(PhaseFieldModel):
    """Two-phase Kim--Kim--Suzuki model with analytic parabolic phase partitioning."""

    composition_mobility: float = 1.0
    order_mobility: float = 1.0
    phase_a_curvature: float = 4.0
    phase_b_curvature: float = 4.0
    phase_a_equilibrium: float = 0.2
    phase_b_equilibrium: float = 0.8
    barrier: float = 1.0
    gradient_coefficient: float = 1.0
    composition_gradient: float = 0.0
    composition_name: str = "c"
    order_name: str = "eta"
    name: str = "kks"

    @property
    def field_specs(self):
        return (
            FieldSpec(self.composition_name, conserved=True, lower_bound=0.0, upper_bound=1.0),
            FieldSpec(self.order_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
        )

    def _partition(self, composition, eta):
        h = _smoothstep(eta)
        aa = self.phase_a_curvature
        ab = self.phase_b_curvature
        denominator = (1.0 - h) / aa + h / ab
        chemical_potential = (
            composition
            - (1.0 - h) * self.phase_a_equilibrium
            - h * self.phase_b_equilibrium
        ) / (denominator + 1.0e-14)
        c_a = self.phase_a_equilibrium + chemical_potential / aa
        c_b = self.phase_b_equilibrium + chemical_potential / ab
        return c_a, c_b, chemical_potential, h

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        c = fields[self.composition_name]
        eta = fields[self.order_name]
        c_a, c_b, mu, h = self._partition(c, eta)
        f_a = 0.5 * self.phase_a_curvature * (c_a - self.phase_a_equilibrium) ** 2
        f_b = 0.5 * self.phase_b_curvature * (c_b - self.phase_b_equilibrium) ** 2
        d_eta = (
            _smoothstep_derivative(eta) * (f_b - f_a - mu * (c_b - c_a))
            + 2.0 * self.barrier * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
            - self.gradient_coefficient * operators.laplacian(eta)
        )
        if self.composition_gradient:
            mu = mu - self.composition_gradient * operators.laplacian(c)
        return {
            self.composition_name: self.composition_mobility * operators.laplacian(mu),
            self.order_name: -self.order_mobility * d_eta,
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.composition_name:
            return -self.composition_mobility * self.composition_gradient * k2**2
        if field_name == self.order_name:
            return -self.order_mobility * self.gradient_coefficient * k2
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        c = fields[self.composition_name]
        eta = fields[self.order_name]
        c_a, c_b, _, h = self._partition(c, eta)
        f_a = 0.5 * self.phase_a_curvature * (c_a - self.phase_a_equilibrium) ** 2
        f_b = 0.5 * self.phase_b_curvature * (c_b - self.phase_b_equilibrium) ** 2
        return (
            (1.0 - h) * f_a
            + h * f_b
            + self.barrier * eta**2 * (1.0 - eta) ** 2
            + 0.5 * self.gradient_coefficient * operators.grad_squared(eta)
            + 0.5 * self.composition_gradient * operators.grad_squared(c)
        )


@dataclass(slots=True)
class GrandPotentialModel(PhaseFieldModel):
    """Two-phase grand-potential formulation using chemical potential as a variable."""

    susceptibility_a: float = 1.0
    susceptibility_b: float = 1.0
    reference_mu_a: float = -0.5
    reference_mu_b: float = 0.5
    diffusivity: float = 1.0
    order_mobility: float = 1.0
    barrier: float = 1.0
    gradient_coefficient: float = 1.0
    order_name: str = "eta"
    chemical_potential_name: str = "mu"
    name: str = "grand-potential"

    @property
    def field_specs(self):
        return (
            FieldSpec(self.order_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
            FieldSpec(self.chemical_potential_name, conserved=True),
        )

    def phase_concentrations(self, mu):
        c_a = self.susceptibility_a * (mu - self.reference_mu_a)
        c_b = self.susceptibility_b * (mu - self.reference_mu_b)
        return c_a, c_b

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        eta = fields[self.order_name]
        mu = fields[self.chemical_potential_name]
        h = _smoothstep(eta)
        dh = _smoothstep_derivative(eta)
        c_a, c_b = self.phase_concentrations(mu)
        omega_a = -0.5 * self.susceptibility_a * (mu - self.reference_mu_a) ** 2
        omega_b = -0.5 * self.susceptibility_b * (mu - self.reference_mu_b) ** 2
        d_eta = (
            dh * (omega_b - omega_a)
            + 2.0 * self.barrier * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
            - self.gradient_coefficient * operators.laplacian(eta)
        )
        eta_rate = -self.order_mobility * d_eta
        susceptibility = (1.0 - h) * self.susceptibility_a + h * self.susceptibility_b
        source = -dh * (c_b - c_a) * eta_rate
        mu_rate = (
            self.diffusivity * operators.laplacian(mu) + source
        ) / (susceptibility + 1.0e-12)
        return {self.order_name: eta_rate, self.chemical_potential_name: mu_rate}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.order_name:
            return -self.order_mobility * self.gradient_coefficient * k2
        if field_name == self.chemical_potential_name:
            average_susceptibility = 0.5 * (
                self.susceptibility_a + self.susceptibility_b
            )
            return -self.diffusivity * k2 / average_susceptibility
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        eta = fields[self.order_name]
        mu = fields[self.chemical_potential_name]
        h = _smoothstep(eta)
        omega_a = -0.5 * self.susceptibility_a * (mu - self.reference_mu_a) ** 2
        omega_b = -0.5 * self.susceptibility_b * (mu - self.reference_mu_b) ** 2
        return (
            (1.0 - h) * omega_a
            + h * omega_b
            + self.barrier * eta**2 * (1.0 - eta) ** 2
            + 0.5 * self.gradient_coefficient * operators.grad_squared(eta)
        )


@dataclass(slots=True)
class SinteringModel(PhaseFieldModel):
    """Conserved density plus non-conserved grain order parameters for sintering."""

    grains: int = 3
    density_mobility: float = 1.0
    grain_mobility: float = 1.0
    density_gradient: float = 1.0
    grain_gradient: float = 1.0
    density_barrier: float = 1.0
    grain_barrier: float = 1.0
    coupling: float = 2.0
    density_name: str = "rho"
    prefix: str = "eta"
    name: str = "sintering"

    @property
    def field_specs(self):
        return (
            FieldSpec(self.density_name, conserved=True, lower_bound=0.0, upper_bound=1.0),
            *tuple(
                FieldSpec(f"{self.prefix}{index}", conserved=False, lower_bound=0.0, upper_bound=1.0)
                for index in range(self.grains)
            ),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        rho = fields[self.density_name]
        eta_sum = 0.0
        for index in range(self.grains):
            eta_sum = eta_sum + fields[f"{self.prefix}{index}"] ** 2
        mu_rho = (
            2.0 * self.density_barrier * rho * (1.0 - rho) * (1.0 - 2.0 * rho)
            + self.coupling * (rho - eta_sum)
            - self.density_gradient * operators.laplacian(rho)
        )
        result = {
            self.density_name: self.density_mobility * operators.laplacian(mu_rho)
        }
        for index in range(self.grains):
            name = f"{self.prefix}{index}"
            eta = fields[name]
            derivative = (
                2.0 * self.grain_barrier * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
                - 2.0 * self.coupling * eta * (rho - eta_sum)
                - self.grain_gradient * operators.laplacian(eta)
            )
            result[name] = -self.grain_mobility * derivative
        return result

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.density_name:
            return -self.density_mobility * self.density_gradient * k2**2
        return -self.grain_mobility * self.grain_gradient * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        rho = fields[self.density_name]
        eta_sum = 0.0
        grain_energy = 0.0
        for index in range(self.grains):
            eta = fields[f"{self.prefix}{index}"]
            eta_sum = eta_sum + eta**2
            grain_energy = grain_energy + (
                self.grain_barrier * eta**2 * (1.0 - eta) ** 2
                + 0.5 * self.grain_gradient * operators.grad_squared(eta)
            )
        return (
            self.density_barrier * rho**2 * (1.0 - rho) ** 2
            + 0.5 * self.coupling * (rho - eta_sum) ** 2
            + 0.5 * self.density_gradient * operators.grad_squared(rho)
            + grain_energy
        )


@dataclass(slots=True)
class OrientationFieldModel(PhaseFieldModel):
    """Kobayashi--Warren--Carter-style phase and orientation model."""

    phase_mobility: float = 1.0
    orientation_mobility: float = 1.0
    barrier: float = 1.0
    phase_gradient: float = 1.0
    orientation_gradient: float = 0.2
    orientation_regularization: float = 1.0e-6
    phase_name: str = "phi"
    orientation_name: str = "theta"
    name: str = "orientation-field"

    @property
    def field_specs(self):
        return (
            FieldSpec(self.phase_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
            FieldSpec(self.orientation_name, conserved=False),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        phi = fields[self.phase_name]
        theta = fields[self.orientation_name]
        theta_grad = operators.gradient(theta)
        magnitude = self.orientation_regularization
        for component in theta_grad:
            magnitude = magnitude + component**2
        magnitude = magnitude**0.5
        phase_derivative = (
            2.0 * self.barrier * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
            + 2.0 * self.orientation_gradient * phi * magnitude
            - self.phase_gradient * operators.laplacian(phi)
        )
        flux = tuple(
            self.orientation_gradient * phi**2 * component / magnitude
            for component in theta_grad
        )
        orientation_rate = self.orientation_mobility * operators.divergence(flux)
        return {
            self.phase_name: -self.phase_mobility * phase_derivative,
            self.orientation_name: orientation_rate,
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name == self.phase_name:
            return -self.phase_mobility * self.phase_gradient * k2
        return -self.orientation_mobility * self.orientation_gradient * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        phi = fields[self.phase_name]
        theta = fields[self.orientation_name]
        theta_grad = operators.grad_squared(theta)
        return (
            self.barrier * phi**2 * (1.0 - phi) ** 2
            + 0.5 * self.phase_gradient * operators.grad_squared(phi)
            + self.orientation_gradient * phi**2 * (theta_grad + self.orientation_regularization) ** 0.5
        )


__all__ = [
    "GrandPotentialModel",
    "GrainGrowthModel",
    "KKSModel",
    "MultiphaseFieldModel",
    "OrientationFieldModel",
    "SinteringModel",
]
