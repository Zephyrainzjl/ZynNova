"""High-order finite-difference solver for periodic, Neumann, and Dirichlet grids."""

from __future__ import annotations

from collections.abc import Iterable
import time as walltime

import numpy as np

from ..config import SolverConfig, TimeScheme
from ..fields import (
    PhaseFieldDiagnostics,
    PhaseFieldResult,
    PhaseFieldState,
    PhaseFieldTrajectory,
)
from ..operators import NumpyDifferentialOperators
from .base import PhaseFieldCallback, PhaseFieldSolver, normalized_error


class FiniteDifferencePhaseFieldSolver(PhaseFieldSolver):
    """Second/fourth-order method-of-lines solver with SSPRK3 or RK4.

    The spatial kernels automatically use the pybind11/OpenMP backend when available.
    This path supports non-periodic boundaries and arbitrary registered models. For
    stiff fourth-order equations, prefer the spectral solver whenever periodic
    boundaries are physically appropriate.
    """

    def __init__(self, model, config: SolverConfig, *, backend: str = "auto"):
        super().__init__(model, config)
        if config.scheme not in {TimeScheme.SSPRK3, TimeScheme.RK4}:
            raise ValueError("finite-difference solver supports SSPRK3 and RK4")
        self.backend = backend

    def _rhs(self, fields, state, operators, time):
        return self.model.rhs(fields, state.grid, operators, xp=np, time=time)

    @staticmethod
    def _combine(base, increments):
        return {
            name: np.asarray(base[name]) + increments[name]
            for name in base
        }

    def _ssprk3(self, fields, state, operators, time, dt):
        k1 = self._rhs(fields, state, operators, time)
        u1 = self.model.project(
            {name: fields[name] + dt * k1[name] for name in fields}, xp=np
        )
        k2 = self._rhs(u1, state, operators, time + dt)
        u2 = self.model.project(
            {
                name: 0.75 * fields[name] + 0.25 * (u1[name] + dt * k2[name])
                for name in fields
            },
            xp=np,
        )
        k3 = self._rhs(u2, state, operators, time + 0.5 * dt)
        return self.model.project(
            {
                name: (1.0 / 3.0) * fields[name]
                + (2.0 / 3.0) * (u2[name] + dt * k3[name])
                for name in fields
            },
            xp=np,
        )

    def _rk4(self, fields, state, operators, time, dt):
        k1 = self._rhs(fields, state, operators, time)
        y2 = {name: fields[name] + 0.5 * dt * k1[name] for name in fields}
        k2 = self._rhs(y2, state, operators, time + 0.5 * dt)
        y3 = {name: fields[name] + 0.5 * dt * k2[name] for name in fields}
        k3 = self._rhs(y3, state, operators, time + 0.5 * dt)
        y4 = {name: fields[name] + dt * k3[name] for name in fields}
        k4 = self._rhs(y4, state, operators, time + dt)
        return self.model.project(
            {
                name: fields[name]
                + dt
                * (
                    k1[name]
                    + 2.0 * k2[name]
                    + 2.0 * k3[name]
                    + k4[name]
                )
                / 6.0
                for name in fields
            },
            xp=np,
        )

    def _step(self, fields, state, operators, time, dt):
        if self.config.scheme == TimeScheme.SSPRK3:
            return self._ssprk3(fields, state, operators, time, dt)
        return self._rk4(fields, state, operators, time, dt)

    def _energy(self, state):
        try:
            return float(self.model.free_energy(state))
        except (NotImplementedError, ValueError, FloatingPointError):
            return np.nan

    def run(self, initial_state, *, callbacks: Iterable[PhaseFieldCallback] = ()):
        self.model.validate_state(initial_state)
        state = initial_state.numpy().copy()
        operators = NumpyDifferentialOperators(
            state.grid,
            method="finite-difference",
            order=self.config.derivative_order,
            finite_difference_backend=self.backend,
        )
        trajectory = PhaseFieldTrajectory()
        trajectory.append(state)
        target_time = self._target_time(state)
        maximum_steps = self._maximum_steps()
        dt = min(self.config.dt, target_time - state.time)
        if self.config.adaptive.maximum_dt is not None:
            dt = min(dt, self.config.adaptive.maximum_dt)
        previous_energy = self._energy(state)
        accepted_steps = 0
        rejected_steps = 0
        start = walltime.perf_counter()

        while state.time < target_time - 1.0e-15 and accepted_steps < maximum_steps:
            dt = min(dt, target_time - state.time)
            old = {name: np.asarray(values).copy() for name, values in state.fields.items()}
            if self.config.adaptive.enabled:
                coarse = self._step(old, state, operators, state.time, dt)
                half = self._step(old, state, operators, state.time, 0.5 * dt)
                fine = self._step(half, state, operators, state.time + 0.5 * dt, 0.5 * dt)
                error = normalized_error(
                    coarse,
                    fine,
                    relative_tolerance=self.config.adaptive.relative_tolerance,
                    absolute_tolerance=self.config.adaptive.absolute_tolerance,
                )
                candidate_fields = fine
            else:
                candidate_fields = self._step(old, state, operators, state.time, dt)
                error = 0.0

            finite = all(np.isfinite(np.asarray(value)).all() for value in candidate_fields.values())
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
            if not finite or error > 1.0 or energy_increase:
                rejected_steps += 1
                dt = max(
                    self.config.adaptive.minimum_dt,
                    dt
                    * max(
                        self.config.adaptive.shrink_limit,
                        self.config.adaptive.safety * max(error, 1.0) ** (-0.25),
                    ),
                )
                if dt <= self.config.adaptive.minimum_dt * (1.0 + 1.0e-12):
                    return PhaseFieldResult(
                        trajectory,
                        "cpp-fd" if self.backend == "cpp" else "numpy-fd",
                        self.config.scheme.value,
                        walltime.perf_counter() - start,
                        accepted_steps,
                        rejected_steps,
                        False,
                        "minimum dt reached before a stable finite-difference step",
                    )
                continue

            state = candidate
            accepted_steps += 1
            energy_change = candidate_energy - previous_energy if (
                np.isfinite(candidate_energy) and np.isfinite(previous_energy)
            ) else np.nan
            previous_energy = candidate_energy
            cell_volume = float(np.prod(state.grid.spacing))
            mass = {
                spec.name: float(np.sum(state.fields[spec.name]) * cell_volume)
                for spec in self.model.field_specs
                if spec.conserved
            }
            extrema = {
                spec.name: (
                    float(np.min(state.fields[spec.name])),
                    float(np.max(state.fields[spec.name])),
                )
                for spec in self.model.field_specs
            }
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

            if self.config.adaptive.enabled:
                factor = self.config.adaptive.growth_limit if error <= 1.0e-16 else (
                    self.config.adaptive.safety * error ** (-0.25)
                )
                dt *= min(self.config.adaptive.growth_limit, max(1.0, factor))
                if self.config.adaptive.maximum_dt is not None:
                    dt = min(dt, self.config.adaptive.maximum_dt)

        converged = state.time >= target_time - 1.0e-12
        return PhaseFieldResult(
            trajectory,
            "cpp-fd" if self.backend == "cpp" else "numpy-fd",
            self.config.scheme.value,
            walltime.perf_counter() - start,
            accepted_steps,
            rejected_steps,
            converged,
            "completed" if converged else "maximum step count reached",
        )


__all__ = ["FiniteDifferencePhaseFieldSolver"]
