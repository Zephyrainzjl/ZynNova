"""Canonical Model-A, Model-B, Model-C, advective, and reaction phase fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..config import GridSpec
from ..fields import FieldSpec
from .base import PhaseFieldModel


@dataclass(slots=True)
class AllenCahnModel(PhaseFieldModel):
    """Non-conserved Model-A dynamics with a quartic double-well free energy."""

    mobility: float = 1.0
    quadratic: float = -1.0
    quartic: float = 1.0
    gradient_coefficient: float = 1.0
    external_bias: float | Callable[[Any, float], Any] = 0.0
    field_name: str = "phi"
    name: str = "allen-cahn"

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=False, description="non-conserved order parameter"),)

    def _bias(self, phi, time):
        return self.external_bias(phi, time) if callable(self.external_bias) else self.external_bias

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        phi = fields[self.field_name]
        chemical_potential = (
            self.quadratic * phi
            + self.quartic * phi**3
            - self.gradient_coefficient * operators.laplacian(phi)
            - self._bias(phi, time)
        )
        return {self.field_name: -self.mobility * chemical_potential}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        if field_name != self.field_name:
            return 0.0 * k2
        return -self.mobility * (self.quadratic + self.gradient_coefficient * k2)

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        phi = fields[self.field_name]
        return (
            0.5 * self.quadratic * phi**2
            + 0.25 * self.quartic * phi**4
            + 0.5 * self.gradient_coefficient * operators.grad_squared(phi)
            - self._bias(phi, 0.0) * phi
        )


@dataclass(slots=True)
class CahnHilliardModel(PhaseFieldModel):
    """Conserved Model-B dynamics with a quartic bulk free energy."""

    mobility: float = 1.0
    quadratic: float = -1.0
    quartic: float = 1.0
    gradient_coefficient: float = 1.0
    source: float | Callable[[Any, float], Any] = 0.0
    field_name: str = "c"
    name: str = "cahn-hilliard"

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=True, description="conserved composition"),)

    def _source(self, composition, time):
        return self.source(composition, time) if callable(self.source) else self.source

    def chemical_potential(self, composition, operators):
        return (
            self.quadratic * composition
            + self.quartic * composition**3
            - self.gradient_coefficient * operators.laplacian(composition)
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        composition = fields[self.field_name]
        chemical_potential = self.chemical_potential(composition, operators)
        return {
            self.field_name: self.mobility * operators.laplacian(chemical_potential)
            + self._source(composition, time)
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        if field_name != self.field_name:
            return 0.0 * k2
        return -self.mobility * k2 * (
            self.quadratic + self.gradient_coefficient * k2
        )

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        composition = fields[self.field_name]
        return (
            0.5 * self.quadratic * composition**2
            + 0.25 * self.quartic * composition**4
            + 0.5 * self.gradient_coefficient * operators.grad_squared(composition)
        )


@dataclass(slots=True)
class CoupledAllenCahnCahnHilliardModel(PhaseFieldModel):
    """Model-C dynamics coupling one conserved and one non-conserved variable."""

    composition_mobility: float = 1.0
    order_mobility: float = 1.0
    composition_quadratic: float = -1.0
    composition_quartic: float = 1.0
    order_quadratic: float = -1.0
    order_quartic: float = 1.0
    composition_gradient: float = 1.0
    order_gradient: float = 1.0
    coupling: float = 1.0
    composition_name: str = "c"
    order_name: str = "eta"
    name: str = "model-c"

    @property
    def field_specs(self):
        return (
            FieldSpec(self.composition_name, conserved=True),
            FieldSpec(self.order_name, conserved=False),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        c = fields[self.composition_name]
        eta = fields[self.order_name]
        mu_c = (
            self.composition_quadratic * c
            + self.composition_quartic * c**3
            + self.coupling * eta**2 * c
            - self.composition_gradient * operators.laplacian(c)
        )
        mu_eta = (
            self.order_quadratic * eta
            + self.order_quartic * eta**3
            + self.coupling * c**2 * eta
            - self.order_gradient * operators.laplacian(eta)
        )
        return {
            self.composition_name: self.composition_mobility * operators.laplacian(mu_c),
            self.order_name: -self.order_mobility * mu_eta,
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        if field_name == self.composition_name:
            return -self.composition_mobility * k2 * (
                self.composition_quadratic + self.composition_gradient * k2
            )
        if field_name == self.order_name:
            return -self.order_mobility * (
                self.order_quadratic + self.order_gradient * k2
            )
        return 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        c = fields[self.composition_name]
        eta = fields[self.order_name]
        return (
            0.5 * self.composition_quadratic * c**2
            + 0.25 * self.composition_quartic * c**4
            + 0.5 * self.order_quadratic * eta**2
            + 0.25 * self.order_quartic * eta**4
            + 0.5 * self.coupling * c**2 * eta**2
            + 0.5 * self.composition_gradient * operators.grad_squared(c)
            + 0.5 * self.order_gradient * operators.grad_squared(eta)
        )


@dataclass(slots=True)
class AdvectiveCahnHilliardModel(CahnHilliardModel):
    """Model-H composition equation with an injected velocity field.

    The momentum equation can be solved by ZynSim's fluid or multiphysics module and
    supplied through ``velocity``. This keeps the phase-field solver conservative while
    allowing Navier--Stokes, Darcy, or measured velocity coupling.
    """

    velocity: tuple[Any, ...] | Callable[[dict[str, Any], GridSpec, float], tuple[Any, ...]] | None = None
    name: str = "advective-cahn-hilliard"
    gradient_flow: bool = False

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        result = CahnHilliardModel.rhs(self, fields, grid, operators, xp=xp, time=time)
        if self.velocity is None:
            return result
        velocity = self.velocity(fields, grid, time) if callable(self.velocity) else self.velocity
        gradient = operators.gradient(fields[self.field_name])
        advection = 0.0
        for component, derivative in zip(velocity, gradient, strict=True):
            advection = advection + component * derivative
        result[self.field_name] = result[self.field_name] - advection
        return result


@dataclass(slots=True)
class ReactionCahnHilliardModel(CahnHilliardModel):
    """Cahn--Hilliard reaction model for precipitation, lithiation, and conversion."""

    reaction_rate: float = 0.0
    equilibrium: float = 0.0
    reaction_function: Callable[[Any, Any, float], Any] | None = None
    name: str = "reaction-cahn-hilliard"
    gradient_flow: bool = False

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        composition = fields[self.field_name]
        chemical_potential = self.chemical_potential(composition, operators)
        if self.reaction_function is None:
            reaction = -self.reaction_rate * (chemical_potential - self.equilibrium)
        else:
            reaction = self.reaction_function(composition, chemical_potential, time)
        return {
            self.field_name: self.mobility * operators.laplacian(chemical_potential) + reaction
        }


__all__ = [
    "AdvectiveCahnHilliardModel",
    "AllenCahnModel",
    "CahnHilliardModel",
    "CoupledAllenCahnCahnHilliardModel",
    "ReactionCahnHilliardModel",
]
