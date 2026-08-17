"""High-accuracy Fourier pseudo-spectral ETDRK4 and IMEX-BDF2 solvers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import time as walltime

import numpy as np

from ..config import SolverConfig, TimeScheme
from ..fields import (
    PhaseFieldDiagnostics,
    PhaseFieldResult,
    PhaseFieldState,
    PhaseFieldTrajectory,
)
from ..models import PhaseFieldModel
from ..operators import NumpyDifferentialOperators, numpy_wave_numbers
from .base import PhaseFieldCallback, PhaseFieldSolver, normalized_error


@dataclass(frozen=True, slots=True)
class _ETDCoefficients:
    E: np.ndarray
    E2: np.ndarray
    Q: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


class SpectralPhaseFieldSolver(PhaseFieldSolver):
    """Periodic Fourier solver with fourth-order exponential time integration.

    ETDRK4 uses a contour-integral evaluation of the phi functions. When adaptive
    stepping is enabled, a full step is compared with two half steps; gradient-flow
    models additionally reject steps that violate the configured free-energy decay.
    """

    def __init__(self, model: PhaseFieldModel, config: SolverConfig):
        super().__init__(model, config)
        if config.scheme not in {
            TimeScheme.ETDRK4,
            TimeScheme.IMEX_BDF2,
            TimeScheme.SEMI_IMPLICIT_EULER,
        }:
            raise ValueError("spectral solver supports ETDRK4, IMEX-BDF2, and semi-implicit Euler")
        self._coefficient_cache: dict[tuple[str, float], _ETDCoefficients] = {}

    @staticmethod
    def _axes(grid) -> tuple[int, ...]:
        return tuple(range(-grid.dimensions, 0))

    def _fft(self, values, grid):
        return np.fft.fftn(values, axes=self._axes(grid))

    def _ifft(self, values, grid):
        return np.fft.ifftn(values, axes=self._axes(grid)).real

    def _linear_symbols(self, grid) -> dict[str, np.ndarray]:
        k2, _ = numpy_wave_numbers(grid)
        return {
            spec.name: np.asarray(
                self.model.linear_symbol(spec.name, grid, k2, xp=np),
                dtype=float,
            )
            for spec in self.model.field_specs
        }

    def _apply_symbol(self, values, symbol, grid):
        return self._ifft(symbol * self._fft(values, grid), grid)

    def _nonlinear_rhs(self, fields, grid, operators, symbols, time):
        total = self.model.rhs(fields, grid, operators, xp=np, time=time)
        return {
            name: np.asarray(total[name]) - self._apply_symbol(fields[name], symbols[name], grid)
            for name in fields
        }

    def _etd_coefficients(self, name, symbol, dt):
        key = (name, float(dt))
        cached = self._coefficient_cache.get(key)
        if cached is not None:
            return cached
        z = dt * symbol
        E = np.exp(z)
        E2 = np.exp(z / 2.0)
        roots = np.exp(1j * np.pi * (np.arange(32) + 0.5) / 32.0)
        lr = z[..., None] + roots
        Q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=-1))
        f1 = dt * np.real(
            np.mean(
                (-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3,
                axis=-1,
            )
        )
        f2 = dt * np.real(
            np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=-1)
        )
        f3 = dt * np.real(
            np.mean(
                (-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3,
                axis=-1,
            )
        )
        coefficients = _ETDCoefficients(E, E2, Q, f1, f2, f3)
        self._coefficient_cache[key] = coefficients
        return coefficients

    def _multiply(self, multiplier, values, grid):
        return self._ifft(multiplier * self._fft(values, grid), grid)

    def _etdrk4_step(self, fields, grid, operators, symbols, time, dt):
        coefficients = {
            name: self._etd_coefficients(name, symbols[name], dt) for name in fields
        }
        n0 = self._nonlinear_rhs(fields, grid, operators, symbols, time)
        a = {
            name: self._multiply(coefficients[name].E2, fields[name], grid)
            + self._multiply(coefficients[name].Q, n0[name], grid)
            for name in fields
        }
        na = self._nonlinear_rhs(a, grid, operators, symbols, time + 0.5 * dt)
        b = {
            name: self._multiply(coefficients[name].E2, fields[name], grid)
            + self._multiply(coefficients[name].Q, na[name], grid)
            for name in fields
        }
        nb = self._nonlinear_rhs(b, grid, operators, symbols, time + 0.5 * dt)
        c = {
            name: self._multiply(coefficients[name].E2, a[name], grid)
            + self._multiply(coefficients[name].Q, 2.0 * nb[name] - n0[name], grid)
            for name in fields
        }
        nc = self._nonlinear_rhs(c, grid, operators, symbols, time + dt)
        updated = {
            name: self._multiply(coefficients[name].E, fields[name], grid)
            + self._multiply(coefficients[name].f1, n0[name], grid)
            + 2.0
            * self._multiply(coefficients[name].f2, na[name] + nb[name], grid)
            + self._multiply(coefficients[name].f3, nc[name], grid)
            for name in fields
        }
        return self.model.project(updated, xp=np)

    def _semi_implicit_step(self, fields, grid, operators, symbols, time, dt):
        nonlinear = self._nonlinear_rhs(fields, grid, operators, symbols, time)
        updated = {}
        for name, values in fields.items():
            numerator = self._fft(values + dt * nonlinear[name], grid)
            updated[name] = self._ifft(numerator / (1.0 - dt * symbols[name]), grid)
        return self.model.project(updated, xp=np), nonlinear

    def _bdf2_step(
        self,
        fields,
        previous_fields,
        previous_nonlinear,
        grid,
        operators,
        symbols,
        time,
        dt,
    ):
        nonlinear = self._nonlinear_rhs(fields, grid, operators, symbols, time)
        updated = {}
        for name, values in fields.items():
            rhs = (
                2.0 * values / dt
                - 0.5 * previous_fields[name] / dt
                + 2.0 * nonlinear[name]
                - previous_nonlinear[name]
            )
            denominator = 1.5 / dt - symbols[name]
            updated[name] = self._ifft(self._fft(rhs, grid) / denominator, grid)
        return self.model.project(updated, xp=np), nonlinear

    def _energy(self, state):
        try:
            value = float(self.model.free_energy(state))
            return value if np.isfinite(value) else np.nan
        except (NotImplementedError, FloatingPointError, ValueError):
            return np.nan

    @staticmethod
    def _mass_and_extrema(state, specs):
        cell_volume = float(np.prod(state.grid.spacing))
        mass = {}
        extrema = {}
        for spec in specs:
            values = np.asarray(state.fields[spec.name])
            if spec.conserved:
                mass[spec.name] = float(np.sum(values) * cell_volume)
            extrema[spec.name] = (float(np.min(values)), float(np.max(values)))
        return mass, extrema

    def run(self, initial_state, *, callbacks: Iterable[PhaseFieldCallback] = ()):
        if not initial_state.grid.periodic:
            raise ValueError("Fourier pseudo-spectral integration requires periodic boundaries")
        self.model.validate_state(initial_state)
        state = initial_state.numpy().copy()
        operators = NumpyDifferentialOperators(state.grid, method="spectral")
        symbols = self._linear_symbols(state.grid)
        trajectory = PhaseFieldTrajectory()
        trajectory.append(state)
        target_time = self._target_time(state)
        maximum_steps = self._maximum_steps()
        dt = min(self.config.dt, target_time - state.time)
        if self.config.adaptive.maximum_dt is not None:
            dt = min(dt, self.config.adaptive.maximum_dt)
        previous_energy = self._energy(state)
        previous_fields = None
        previous_nonlinear = None
        accepted_steps = 0
        rejected_steps = 0
        start = walltime.perf_counter()

        while state.time < target_time - 1.0e-15 and accepted_steps < maximum_steps:
            dt = min(dt, target_time - state.time)
            old_fields = {name: np.asarray(values).copy() for name, values in state.fields.items()}
            candidate_fields = None
            error = 0.0

            if self.config.scheme == TimeScheme.ETDRK4:
                if self.config.adaptive.enabled:
                    coarse = self._etdrk4_step(
                        old_fields, state.grid, operators, symbols, state.time, dt
                    )
                    half = self._etdrk4_step(
                        old_fields, state.grid, operators, symbols, state.time, 0.5 * dt
                    )
                    fine = self._etdrk4_step(
                        half, state.grid, operators, symbols, state.time + 0.5 * dt, 0.5 * dt
                    )
                    error = normalized_error(
                        coarse,
                        fine,
                        relative_tolerance=self.config.adaptive.relative_tolerance,
                        absolute_tolerance=self.config.adaptive.absolute_tolerance,
                    )
                    candidate_fields = fine
                else:
                    candidate_fields = self._etdrk4_step(
                        old_fields, state.grid, operators, symbols, state.time, dt
                    )
            elif self.config.scheme == TimeScheme.SEMI_IMPLICIT_EULER:
                candidate_fields, current_nonlinear = self._semi_implicit_step(
                    old_fields, state.grid, operators, symbols, state.time, dt
                )
            else:
                if previous_fields is None or previous_nonlinear is None:
                    candidate_fields, current_nonlinear = self._semi_implicit_step(
                        old_fields, state.grid, operators, symbols, state.time, dt
                    )
                else:
                    candidate_fields, current_nonlinear = self._bdf2_step(
                        old_fields,
                        previous_fields,
                        previous_nonlinear,
                        state.grid,
                        operators,
                        symbols,
                        state.time,
                        dt,
                    )

            candidate = PhaseFieldState(
                state.grid,
                candidate_fields,
                time=state.time + dt,
                step=state.step + 1,
                metadata=dict(state.metadata),
            )
            candidate_energy = self._energy(candidate)
            energy_increase = (
                self.model.gradient_flow
                and self.config.adaptive.enforce_energy_decay
                and np.isfinite(previous_energy)
                and np.isfinite(candidate_energy)
                and candidate_energy
                > previous_energy
                + self.config.adaptive.energy_tolerance * max(1.0, abs(previous_energy))
            )
            error_reject = self.config.adaptive.enabled and error > 1.0

            if error_reject or energy_increase:
                rejected_steps += 1
                if rejected_steps > self.config.adaptive.maximum_rejections * max(1, accepted_steps + 1):
                    message = "adaptive integrator exceeded the rejection budget"
                    return PhaseFieldResult(
                        trajectory,
                        "numpy-spectral",
                        self.config.scheme.value,
                        walltime.perf_counter() - start,
                        accepted_steps,
                        rejected_steps,
                        False,
                        message,
                    )
                if dt <= self.config.adaptive.minimum_dt * (1.0 + 1.0e-12):
                    message = "minimum dt reached before a stable step could be accepted"
                    return PhaseFieldResult(
                        trajectory,
                        "numpy-spectral",
                        self.config.scheme.value,
                        walltime.perf_counter() - start,
                        accepted_steps,
                        rejected_steps,
                        False,
                        message,
                    )
                factor = self.config.adaptive.shrink_limit
                if error > 0.0 and np.isfinite(error):
                    factor = max(
                        self.config.adaptive.shrink_limit,
                        self.config.adaptive.safety * error ** (-0.2),
                    )
                dt = max(self.config.adaptive.minimum_dt, dt * min(0.8, factor))
                self._coefficient_cache.clear()
                continue

            if self.config.scheme == TimeScheme.IMEX_BDF2:
                previous_fields = old_fields
                previous_nonlinear = current_nonlinear
            state = candidate
            accepted_steps += 1
            energy_change = candidate_energy - previous_energy if (
                np.isfinite(candidate_energy) and np.isfinite(previous_energy)
            ) else np.nan
            previous_energy = candidate_energy
            mass, extrema = self._mass_and_extrema(state, self.model.field_specs)
            diagnostics = PhaseFieldDiagnostics(
                state.step,
                state.time,
                dt,
                candidate_energy,
                energy_change,
                True,
                mass,
                extrema,
                residual_norm=float(error),
                rejected_steps=rejected_steps,
            )
            if state.step % self.config.diagnostics_every == 0:
                trajectory.diagnostics.append(diagnostics)
            if state.step % self.config.save_every == 0 or state.time >= target_time - 1.0e-15:
                trajectory.append(state)
            for callback in callbacks:
                callback(state, diagnostics)

            if self.config.adaptive.enabled and self.config.scheme == TimeScheme.ETDRK4:
                if error <= 1.0e-16:
                    factor = self.config.adaptive.growth_limit
                else:
                    factor = self.config.adaptive.safety * error ** (-0.2)
                    factor = min(self.config.adaptive.growth_limit, max(1.0, factor))
                dt *= factor
                if self.config.adaptive.maximum_dt is not None:
                    dt = min(dt, self.config.adaptive.maximum_dt)
                self._coefficient_cache.clear()

        converged = state.time >= target_time - 1.0e-12
        message = "completed" if converged else "maximum step count reached"
        return PhaseFieldResult(
            trajectory,
            "numpy-spectral",
            self.config.scheme.value,
            walltime.perf_counter() - start,
            accepted_steps,
            rejected_steps,
            converged,
            message,
        )


__all__ = ["SpectralPhaseFieldSolver"]
