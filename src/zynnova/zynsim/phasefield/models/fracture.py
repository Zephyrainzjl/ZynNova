"""Brittle, cohesive, fatigue, and chemo-mechanical phase-field fracture models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..fields import FieldSpec
from .base import PhaseFieldModel, _clip


class FractureRegularization(str, Enum):
    AT1 = "AT1"
    AT2 = "AT2"
    COHESIVE = "cohesive"


@dataclass(slots=True)
class PhaseFieldFractureModel(PhaseFieldModel):
    """Irreversible AT1, AT2, or cohesive phase-field fracture evolution.

    The mechanical tensile-energy history can be a scalar, array, or callback. The
    actual displacement solve can remain in ``zynsim.fem``/``zynsim.multiphysics``;
    this model advances the crack phase field with a viscous regularization.
    """

    fracture_toughness: float = 1.0
    length_scale: float = 1.0
    viscosity: float = 1.0
    regularization: FractureRegularization = FractureRegularization.AT2
    tensile_energy_history: float | Any | Callable[[dict[str, Any], float], Any] = 0.0
    residual_stiffness: float = 1.0e-8
    critical_energy: float = 0.0
    damage_name: str = "damage"
    history_name: str | None = None
    name: str = "phase-field-fracture"
    gradient_flow: bool = False

    def __post_init__(self) -> None:
        self.regularization = FractureRegularization(self.regularization)

    @property
    def field_specs(self):
        specs = [FieldSpec(self.damage_name, conserved=False, lower_bound=0.0, upper_bound=1.0)]
        if self.history_name:
            specs.append(FieldSpec(self.history_name, conserved=False, lower_bound=0.0))
        return tuple(specs)

    def _history(self, fields, time):
        if self.history_name:
            return fields[self.history_name]
        if callable(self.tensile_energy_history):
            return self.tensile_energy_history(dict(fields), time)
        return self.tensile_energy_history

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid
        damage = fields[self.damage_name]
        history = self._history(fields, time)
        gc = self.fracture_toughness
        ell = self.length_scale
        degradation_derivative = -2.0 * (1.0 - damage)
        if self.regularization == FractureRegularization.AT2:
            crack_derivative = gc * (damage / ell - ell * operators.laplacian(damage))
        elif self.regularization == FractureRegularization.AT1:
            crack_derivative = gc * (0.5 / ell - ell * operators.laplacian(damage))
        else:
            crack_derivative = gc * (
                damage / (ell * (1.0 + damage) ** 2)
                - ell * operators.laplacian(damage)
            )
        driving = degradation_derivative * (history - self.critical_energy) + crack_derivative
        damage_rate = -driving / max(self.viscosity, 1.0e-15)
        # Irreversibility: viscous damage is only allowed to increase.
        if getattr(xp, "__name__", "").startswith("torch"):
            damage_rate = xp.relu(damage_rate)
        else:
            damage_rate = xp.maximum(damage_rate, 0.0)
        result = {self.damage_name: damage_rate}
        if self.history_name:
            result[self.history_name] = xp.maximum(history - fields[self.history_name], 0.0)
        return result

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        if field_name != self.damage_name:
            return 0.0 * k2
        gc = self.fracture_toughness
        ell = self.length_scale
        if self.regularization == FractureRegularization.AT2:
            return -(gc / max(self.viscosity, 1.0e-15)) * (1.0 / ell + ell * k2)
        return -(gc * ell / max(self.viscosity, 1.0e-15)) * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid
        damage = fields[self.damage_name]
        history = self._history(fields, 0.0)
        degradation = (1.0 - damage) ** 2 + self.residual_stiffness
        if self.regularization == FractureRegularization.AT2:
            local = 0.5 * damage**2 / self.length_scale
        elif self.regularization == FractureRegularization.AT1:
            local = damage / self.length_scale
        else:
            local = damage**2 / (self.length_scale * (1.0 + damage))
        return (
            degradation * history
            + self.fracture_toughness * (
                local + 0.5 * self.length_scale * operators.grad_squared(damage)
            )
        )

    def project(self, fields, *, xp=np):
        projected = super().project(fields, xp=xp)
        previous = fields.get("_previous_damage")
        if previous is not None:
            projected[self.damage_name] = xp.maximum(projected[self.damage_name], previous)
        return projected


@dataclass(slots=True)
class FatiguePhaseFieldFractureModel(PhaseFieldFractureModel):
    """Fatigue-degraded fracture toughness with an accumulated history variable."""

    fatigue_rate: float = 1.0e-3
    fatigue_threshold: float = 0.0
    fatigue_name: str = "fatigue"
    name: str = "fatigue-phase-field-fracture"

    @property
    def field_specs(self):
        base = list(PhaseFieldFractureModel.field_specs.fget(self))
        base.append(FieldSpec(self.fatigue_name, conserved=False, lower_bound=0.0))
        return tuple(base)

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        fatigue = fields[self.fatigue_name]
        history = self._history(fields, time)
        effective_gc = self.fracture_toughness / (1.0 + fatigue)
        original = self.fracture_toughness
        self.fracture_toughness = effective_gc
        try:
            result = PhaseFieldFractureModel.rhs(self, fields, grid, operators, xp=xp, time=time)
        finally:
            self.fracture_toughness = original
        result[self.fatigue_name] = self.fatigue_rate * xp.maximum(
            history - self.fatigue_threshold,
            0.0,
        )
        return result


@dataclass(slots=True)
class ChemoMechanicalFractureModel(PhaseFieldModel):
    """Coupled Cahn--Hilliard transport and phase-field fracture."""

    diffusivity: float = 1.0
    composition_gradient: float = 0.1
    regular_solution: float = 2.5
    fracture: PhaseFieldFractureModel = None  # type: ignore[assignment]
    chemical_expansion_coupling: float = 1.0
    composition_name: str = "c"
    damage_name: str = "damage"
    name: str = "chemo-mechanical-fracture"
    gradient_flow: bool = False

    def __post_init__(self) -> None:
        if self.fracture is None:
            self.fracture = PhaseFieldFractureModel(damage_name=self.damage_name)

    @property
    def field_specs(self):
        return (
            FieldSpec(self.composition_name, conserved=True, lower_bound=1.0e-8, upper_bound=1.0 - 1.0e-8),
            FieldSpec(self.damage_name, conserved=False, lower_bound=0.0, upper_bound=1.0),
        )

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        c = _clip(xp, fields[self.composition_name], 1.0e-8, 1.0 - 1.0e-8)
        damage = fields[self.damage_name]
        mu = (
            xp.log(c) - xp.log(1.0 - c)
            + self.regular_solution * (1.0 - 2.0 * c)
            - self.composition_gradient * operators.laplacian(c)
            + self.chemical_expansion_coupling * damage
        )
        concentration_rate = self.diffusivity * operators.laplacian(mu)
        damage_result = self.fracture.rhs(
            {self.damage_name: damage},
            grid,
            operators,
            xp=xp,
            time=time,
        )[self.damage_name]
        damage_result = damage_result + self.chemical_expansion_coupling * xp.maximum(
            c - c.mean(), 0.0
        )
        return {self.composition_name: concentration_rate, self.damage_name: damage_result}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        if field_name == self.composition_name:
            return -self.diffusivity * self.composition_gradient * k2**2
        return self.fracture.linear_symbol(self.damage_name, grid, k2, xp=xp)

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        c = _clip(xp, fields[self.composition_name], 1.0e-8, 1.0 - 1.0e-8)
        damage = fields[self.damage_name]
        chemical = (
            c * xp.log(c)
            + (1.0 - c) * xp.log(1.0 - c)
            + self.regular_solution * c * (1.0 - c)
            + 0.5 * self.composition_gradient * operators.grad_squared(c)
        )
        fracture = self.fracture.free_energy_density(
            {self.damage_name: damage}, grid, operators, xp=xp
        )
        return chemical + fracture + self.chemical_expansion_coupling * c * damage


__all__ = [
    "ChemoMechanicalFractureModel",
    "FatiguePhaseFieldFractureModel",
    "FractureRegularization",
    "PhaseFieldFractureModel",
]
