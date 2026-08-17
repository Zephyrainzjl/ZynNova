"""JAX GPU/TPU-capable differentiable Fourier phase-field solver."""

from __future__ import annotations

from collections.abc import Iterable
import time as walltime

import numpy as np

from .._backend import jax_available
from ..config import SolverConfig, TimeScheme
from ..fields import PhaseFieldResult, PhaseFieldState, PhaseFieldTrajectory
from ..operators import ArrayDifferentialOperators
from .base import PhaseFieldCallback, PhaseFieldSolver


class JAXSpectralPhaseFieldSolver(PhaseFieldSolver):
    """JAX array-programming solver for differentiable forward and inverse models."""

    def __init__(self, model, config: SolverConfig):
        if not jax_available():
            raise RuntimeError("JAX and jaxlib are required for JAXSpectralPhaseFieldSolver")
        if config.dtype == "float64":
            import jax

            jax.config.update("jax_enable_x64", True)
        if config.scheme not in {TimeScheme.ETDRK4, TimeScheme.SEMI_IMPLICIT_EULER}:
            raise ValueError("JAX spectral solver supports ETDRK4 and semi-implicit Euler")
        super().__init__(model, config)

    @staticmethod
    def _fft(values):
        import jax.numpy as jnp

        return jnp.fft.fftn(values)

    @staticmethod
    def _ifft(values):
        import jax.numpy as jnp

        return jnp.fft.ifftn(values).real

    def _apply(self, multiplier, values):
        return self._ifft(multiplier * self._fft(values))

    def _symbols(self, grid, operators):
        import jax.numpy as jnp

        return {
            spec.name: jnp.asarray(
                self.model.linear_symbol(spec.name, grid, operators.k2, xp=jnp)
            )
            for spec in self.model.field_specs
        }

    def _nonlinear(self, fields, grid, operators, symbols, time):
        import jax.numpy as jnp

        total = self.model.rhs(fields, grid, operators, xp=jnp, time=time)
        return {
            name: total[name] - self._apply(symbols[name], fields[name])
            for name in fields
        }

    @staticmethod
    def _phi_functions(z):
        import jax.numpy as jnp

        small = jnp.abs(z) < 1.0e-6
        safe_z = jnp.where(small, 1.0, z)
        phi1_regular = jnp.expm1(z) / safe_z
        phi1 = jnp.where(small, 1.0 + z / 2.0 + z**2 / 6.0, phi1_regular)
        phi2 = jnp.where(small, 0.5 + z / 6.0 + z**2 / 24.0, (phi1 - 1.0) / safe_z)
        phi3 = jnp.where(small, 1.0 / 6.0 + z / 24.0 + z**2 / 120.0, (phi2 - 0.5) / safe_z)
        return phi1, phi2, phi3

    def _etdrk4_step(self, fields, grid, operators, symbols, time, dt):
        import jax.numpy as jnp

        coefficients = {}
        for name, symbol in symbols.items():
            z = dt * symbol
            E = jnp.exp(z)
            E2 = jnp.exp(z / 2.0)
            half_phi1, _, _ = self._phi_functions(z / 2.0)
            phi1, phi2, phi3 = self._phi_functions(z)
            coefficients[name] = (
                E,
                E2,
                0.5 * dt * half_phi1,
                dt * (phi1 - 3.0 * phi2 + 4.0 * phi3),
                dt * (phi2 - 2.0 * phi3),
                dt * (-phi2 + 4.0 * phi3),
            )
        n0 = self._nonlinear(fields, grid, operators, symbols, time)
        a = {
            name: self._apply(coefficients[name][1], fields[name])
            + self._apply(coefficients[name][2], n0[name])
            for name in fields
        }
        na = self._nonlinear(a, grid, operators, symbols, time + 0.5 * dt)
        b = {
            name: self._apply(coefficients[name][1], fields[name])
            + self._apply(coefficients[name][2], na[name])
            for name in fields
        }
        nb = self._nonlinear(b, grid, operators, symbols, time + 0.5 * dt)
        c = {
            name: self._apply(coefficients[name][1], a[name])
            + self._apply(coefficients[name][2], 2.0 * nb[name] - n0[name])
            for name in fields
        }
        nc = self._nonlinear(c, grid, operators, symbols, time + dt)
        return self.model.project(
            {
                name: self._apply(coefficients[name][0], fields[name])
                + self._apply(coefficients[name][3], n0[name])
                + 2.0 * self._apply(coefficients[name][4], na[name] + nb[name])
                + self._apply(coefficients[name][5], nc[name])
                for name in fields
            },
            xp=jnp,
        )

    def _semi_implicit_step(self, fields, grid, operators, symbols, time, dt):
        import jax.numpy as jnp

        total = self.model.rhs(fields, grid, operators, xp=jnp, time=time)
        result = {}
        for name in fields:
            linear = self._apply(symbols[name], fields[name])
            nonlinear = total[name] - linear
            result[name] = self._ifft(
                self._fft(fields[name] + dt * nonlinear) / (1.0 - dt * symbols[name])
            )
        return self.model.project(result, xp=jnp)

    def solve_array(self, initial_state: PhaseFieldState):
        import jax.numpy as jnp

        if not initial_state.grid.periodic:
            raise ValueError("JAX spectral integration requires periodic boundaries")
        self.model.validate_state(initial_state)
        dtype = getattr(jnp, self.config.dtype)
        fields = {name: jnp.asarray(values, dtype=dtype) for name, values in initial_state.fields.items()}
        operators = ArrayDifferentialOperators(initial_state.grid, jnp)
        symbols = self._symbols(initial_state.grid, operators)
        target_time = self._target_time(initial_state)
        time = float(initial_state.time)
        steps = 0
        while time < target_time - 1.0e-15 and steps < self._maximum_steps():
            dt = min(self.config.dt, target_time - time)
            if self.config.scheme == TimeScheme.ETDRK4:
                fields = self._etdrk4_step(fields, initial_state.grid, operators, symbols, time, dt)
            else:
                fields = self._semi_implicit_step(fields, initial_state.grid, operators, symbols, time, dt)
            time += dt
            steps += 1
        return fields, time, steps

    def run(self, initial_state, *, callbacks: Iterable[PhaseFieldCallback] = ()):
        start = walltime.perf_counter()
        fields, final_time, steps = self.solve_array(initial_state)
        final_state = PhaseFieldState(
            initial_state.grid,
            fields,
            time=final_time,
            step=initial_state.step + steps,
            metadata=dict(initial_state.metadata),
        )
        trajectory = PhaseFieldTrajectory()
        trajectory.append(initial_state)
        trajectory.append(final_state)
        for callback in callbacks:
            callback(final_state, None)
        return PhaseFieldResult(
            trajectory,
            "jax-spectral",
            self.config.scheme.value,
            walltime.perf_counter() - start,
            steps,
            0,
            True,
            "completed",
        )


__all__ = ["JAXSpectralPhaseFieldSolver"]
