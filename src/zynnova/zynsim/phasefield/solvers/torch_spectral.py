"""GPU-accelerated and differentiable Torch Fourier phase-field solver."""

from __future__ import annotations

from collections.abc import Iterable
import time as walltime

import numpy as np

from .._backend import torch_available
from ..config import SolverConfig, TimeScheme
from ..fields import PhaseFieldResult, PhaseFieldState, PhaseFieldTrajectory
from ..operators import ArrayDifferentialOperators
from .base import PhaseFieldCallback, PhaseFieldSolver


class TorchSpectralPhaseFieldSolver(PhaseFieldSolver):
    """Differentiable periodic solver using Torch FFT on CPU, CUDA, or MPS."""

    def __init__(self, model, config: SolverConfig):
        if not torch_available():
            raise RuntimeError("Torch is required for TorchSpectralPhaseFieldSolver")
        if config.scheme not in {TimeScheme.ETDRK4, TimeScheme.SEMI_IMPLICIT_EULER}:
            raise ValueError("Torch spectral solver supports ETDRK4 and semi-implicit Euler")
        super().__init__(model, config)
        import torch

        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

    def _fft(self, values, dimensions):
        import torch

        return torch.fft.fftn(values, dim=tuple(range(-dimensions, 0)))

    def _ifft(self, values, dimensions):
        import torch

        return torch.fft.ifftn(values, dim=tuple(range(-dimensions, 0))).real

    def _symbols(self, grid, operators):
        import torch

        k2 = operators.k2.to(self.device, self.dtype)
        return {
            spec.name: torch.as_tensor(
                self.model.linear_symbol(spec.name, grid, k2, xp=torch),
                device=self.device,
                dtype=self.dtype,
            )
            for spec in self.model.field_specs
        }

    def _apply(self, multiplier, values, dimensions):
        return self._ifft(multiplier * self._fft(values, dimensions), dimensions)

    def _nonlinear(self, fields, grid, operators, symbols, time):
        import torch

        total = self.model.rhs(fields, grid, operators, xp=torch, time=time)
        return {
            name: total[name] - self._apply(symbols[name], fields[name], grid.dimensions)
            for name in fields
        }

    def _phi1(self, z):
        import torch

        small = torch.abs(z) < 1.0e-7
        regular = torch.expm1(z) / torch.where(small, torch.ones_like(z), z)
        series = 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0
        return torch.where(small, series, regular)

    def _etdrk4_step(self, fields, grid, operators, symbols, time, dt):
        import torch

        coeff = {}
        for name, symbol in symbols.items():
            z = dt * symbol
            E = torch.exp(z)
            E2 = torch.exp(z / 2.0)
            Q = 0.5 * dt * self._phi1(z / 2.0)
            # Krogstad-style fourth-order exponential Runge--Kutta coefficients.
            phi1 = self._phi1(z)
            small = torch.abs(z) < 1.0e-6
            phi2 = torch.where(
                small,
                0.5 + z / 6.0 + z**2 / 24.0,
                (phi1 - 1.0) / z,
            )
            phi3 = torch.where(
                small,
                1.0 / 6.0 + z / 24.0 + z**2 / 120.0,
                (phi2 - 0.5) / z,
            )
            f1 = dt * (phi1 - 3.0 * phi2 + 4.0 * phi3)
            f2 = dt * (phi2 - 2.0 * phi3)
            f3 = dt * (-phi2 + 4.0 * phi3)
            coeff[name] = (E, E2, Q, f1, f2, f3)

        n0 = self._nonlinear(fields, grid, operators, symbols, time)
        a = {
            name: self._apply(coeff[name][1], fields[name], grid.dimensions)
            + self._apply(coeff[name][2], n0[name], grid.dimensions)
            for name in fields
        }
        na = self._nonlinear(a, grid, operators, symbols, time + 0.5 * dt)
        b = {
            name: self._apply(coeff[name][1], fields[name], grid.dimensions)
            + self._apply(coeff[name][2], na[name], grid.dimensions)
            for name in fields
        }
        nb = self._nonlinear(b, grid, operators, symbols, time + 0.5 * dt)
        c = {
            name: self._apply(coeff[name][1], a[name], grid.dimensions)
            + self._apply(coeff[name][2], 2.0 * nb[name] - n0[name], grid.dimensions)
            for name in fields
        }
        nc = self._nonlinear(c, grid, operators, symbols, time + dt)
        return self.model.project(
            {
                name: self._apply(coeff[name][0], fields[name], grid.dimensions)
                + self._apply(coeff[name][3], n0[name], grid.dimensions)
                + 2.0 * self._apply(coeff[name][4], na[name] + nb[name], grid.dimensions)
                + self._apply(coeff[name][5], nc[name], grid.dimensions)
                for name in fields
            },
            xp=torch,
        )

    def _semi_implicit_step(self, fields, grid, operators, symbols, time, dt):
        total = self.model.rhs(fields, grid, operators, xp=__import__("torch"), time=time)
        updated = {}
        for name in fields:
            linear = self._apply(symbols[name], fields[name], grid.dimensions)
            nonlinear = total[name] - linear
            numerator = self._fft(fields[name] + dt * nonlinear, grid.dimensions)
            updated[name] = self._ifft(
                numerator / (1.0 - dt * symbols[name]), grid.dimensions
            )
        import torch

        return self.model.project(updated, xp=torch)

    def solve_tensor(self, initial_state: PhaseFieldState):
        """Return final Torch tensors while preserving the complete autograd graph."""

        import torch

        if not initial_state.grid.periodic:
            raise ValueError("Torch spectral integration requires periodic boundaries")
        self.model.validate_state(initial_state)
        fields = {
            name: (
                values.to(device=self.device, dtype=self.dtype)
                if isinstance(values, torch.Tensor)
                else torch.as_tensor(values, device=self.device, dtype=self.dtype)
            )
            for name, values in initial_state.fields.items()
        }
        operators = ArrayDifferentialOperators(initial_state.grid, torch)
        operators.k = tuple(item.to(self.device, self.dtype) for item in operators.k)
        operators.k2 = operators.k2.to(self.device, self.dtype)
        symbols = self._symbols(initial_state.grid, operators)
        target_time = self._target_time(initial_state)
        time = float(initial_state.time)
        step = 0
        maximum_steps = self._maximum_steps()
        while time < target_time - 1.0e-15 and step < maximum_steps:
            dt = min(self.config.dt, target_time - time)
            if self.config.scheme == TimeScheme.ETDRK4:
                fields = self._etdrk4_step(
                    fields, initial_state.grid, operators, symbols, time, dt
                )
            else:
                fields = self._semi_implicit_step(
                    fields, initial_state.grid, operators, symbols, time, dt
                )
            time += dt
            step += 1
        return fields, time, step

    def run(self, initial_state, *, callbacks: Iterable[PhaseFieldCallback] = ()):
        start = walltime.perf_counter()
        fields, final_time, steps = self.solve_tensor(initial_state)
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
            f"torch-spectral:{self.device}",
            self.config.scheme.value,
            walltime.perf_counter() - start,
            steps,
            0,
            True,
            "completed",
        )


__all__ = ["TorchSpectralPhaseFieldSolver"]
