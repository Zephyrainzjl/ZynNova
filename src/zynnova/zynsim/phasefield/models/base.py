"""Extensible free-energy and kinetic interfaces for phase-field models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..config import GridSpec
from ..fields import FieldSpec, PhaseFieldState
from ..operators import NumpyDifferentialOperators

Array = Any
Fields = Mapping[str, Array]


def _mean(value: Any) -> Any:
    if hasattr(value, "mean"):
        return value.mean()
    return np.mean(value)


def _clip(xp: Any, values: Any, lower: float, upper: float) -> Any:
    if getattr(xp, "__name__", "").startswith("torch"):
        return xp.clamp(values, lower, upper)
    return xp.clip(values, lower, upper)


class PhaseFieldModel(ABC):
    """Base class for all phase-field models.

    Models expose a total right-hand side and, when possible, a constant-coefficient
    spectral linear operator. The latter enables ETDRK4 and IMEX-BDF2 while the same
    model remains usable with finite-difference, Torch, and JAX solvers.
    """

    name: str = "phase-field-model"
    gradient_flow: bool = True

    @property
    @abstractmethod
    def field_specs(self) -> tuple[FieldSpec, ...]:
        """Return all state variables and their conservation laws."""

    @abstractmethod
    def rhs(
        self,
        fields: Fields,
        grid: GridSpec,
        operators: Any,
        *,
        xp: Any = np,
        time: float = 0.0,
    ) -> dict[str, Array]:
        """Evaluate the coupled PDE right-hand side in physical space."""

    def linear_symbol(self, field_name: str, grid: GridSpec, k2: Array, *, xp: Any = np) -> Array:
        """Return a constant-coefficient Fourier symbol for implicit integration."""

        del field_name, grid, xp
        return 0.0 * k2

    def free_energy_density(
        self,
        fields: Fields,
        grid: GridSpec,
        operators: Any,
        *,
        xp: Any = np,
    ) -> Array:
        del fields, grid, operators, xp
        raise NotImplementedError(f"{type(self).__name__} does not define a free energy")

    def free_energy(self, state: PhaseFieldState) -> float:
        operators = NumpyDifferentialOperators(
            state.grid,
            method="spectral" if state.grid.periodic else "finite-difference",
        )
        density = self.free_energy_density(
            {name: np.asarray(values) for name, values in state.fields.items()},
            state.grid,
            operators,
            xp=np,
        )
        return operators.integrate(np.asarray(density, dtype=float))

    def project(self, fields: dict[str, Array], *, xp: Any = np) -> dict[str, Array]:
        """Apply bounds or algebraic constraints after a time step."""

        specs = {spec.name: spec for spec in self.field_specs}
        projected: dict[str, Array] = {}
        for name, values in fields.items():
            spec = specs.get(name)
            if spec is None or (spec.lower_bound is None and spec.upper_bound is None):
                projected[name] = values
                continue
            lower = spec.lower_bound if spec.lower_bound is not None else -float("inf")
            upper = spec.upper_bound if spec.upper_bound is not None else float("inf")
            projected[name] = _clip(xp, values, lower, upper)
        return projected

    def validate_state(self, state: PhaseFieldState) -> None:
        required = {spec.name for spec in self.field_specs}
        missing = required.difference(state.fields)
        if missing:
            raise ValueError(f"state is missing fields: {sorted(missing)}")
        unexpected = set(state.fields).difference(required)
        if unexpected:
            raise ValueError(f"state contains unexpected fields: {sorted(unexpected)}")

    def initial_state(
        self,
        grid: GridSpec,
        fields: Mapping[str, Array],
        *,
        time: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> PhaseFieldState:
        state = PhaseFieldState(
            grid,
            dict(fields),
            time=float(time),
            metadata=dict(metadata or {}),
        )
        self.validate_state(state)
        return state


@dataclass(slots=True)
class CustomPhaseFieldModel(PhaseFieldModel):
    """User-defined model for equations not yet present in the built-in catalog."""

    specs: tuple[FieldSpec, ...]
    rhs_function: Callable[[Fields, GridSpec, Any, Any, float], dict[str, Array]]
    energy_function: Callable[[Fields, GridSpec, Any, Any], Array] | None = None
    linear_symbols: Mapping[str, Callable[[GridSpec, Array, Any], Array] | float] = field(
        default_factory=dict
    )
    projection_function: Callable[[dict[str, Array], Any], dict[str, Array]] | None = None
    model_name: str = "custom-variational-model"
    is_gradient_flow: bool = True

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def gradient_flow(self) -> bool:
        return self.is_gradient_flow

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        return self.specs

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        return self.rhs_function(fields, grid, operators, xp, time)

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        value = self.linear_symbols.get(field_name, 0.0)
        if callable(value):
            return value(grid, k2, xp)
        return float(value) + 0.0 * k2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        if self.energy_function is None:
            return super().free_energy_density(fields, grid, operators, xp=xp)
        return self.energy_function(fields, grid, operators, xp)

    def project(self, fields, *, xp=np):
        if self.projection_function is not None:
            return self.projection_function(fields, xp)
        return super().project(fields, xp=xp)


__all__ = ["CustomPhaseFieldModel", "Fields", "PhaseFieldModel", "_clip", "_mean"]
