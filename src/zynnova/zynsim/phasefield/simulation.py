"""High-level phase-field simulation facade and backend selection."""

from __future__ import annotations

from collections.abc import Iterable

from ._backend import jax_available, native_available, torch_available
from .config import SolverBackend, SolverConfig, TimeScheme
from .fields import PhaseFieldResult, PhaseFieldState
from .models import PhaseFieldModel
from .solvers import (
    FiniteDifferencePhaseFieldSolver,
    JAXSpectralPhaseFieldSolver,
    PhaseFieldCallback,
    SpectralPhaseFieldSolver,
    TorchSpectralPhaseFieldSolver,
)


class PhaseFieldSimulation:
    """Select and execute the appropriate 1D/2D/3D phase-field backend."""

    def __init__(self, model: PhaseFieldModel, config: SolverConfig):
        self.model = model
        self.config = config

    def resolve_backend(self, state: PhaseFieldState) -> SolverBackend:
        backend = self.config.backend
        if backend != SolverBackend.AUTO:
            return backend
        if state.grid.periodic:
            return SolverBackend.NUMPY_SPECTRAL
        return SolverBackend.CPP_FD if native_available() else SolverBackend.NUMPY_FD

    def build_solver(self, state: PhaseFieldState):
        backend = self.resolve_backend(state)
        if backend == SolverBackend.NUMPY_SPECTRAL:
            return SpectralPhaseFieldSolver(self.model, self.config)
        if backend in {SolverBackend.NUMPY_FD, SolverBackend.CPP_FD}:
            if self.config.scheme not in {TimeScheme.SSPRK3, TimeScheme.RK4}:
                self.config.scheme = TimeScheme.SSPRK3
            return FiniteDifferencePhaseFieldSolver(
                self.model,
                self.config,
                backend="cpp" if backend == SolverBackend.CPP_FD else "python",
            )
        if backend == SolverBackend.TORCH_SPECTRAL:
            if not torch_available():
                raise RuntimeError("Torch backend requested but torch is not installed")
            return TorchSpectralPhaseFieldSolver(self.model, self.config)
        if backend == SolverBackend.JAX_SPECTRAL:
            if not jax_available():
                raise RuntimeError("JAX backend requested but jax/jaxlib are not installed")
            return JAXSpectralPhaseFieldSolver(self.model, self.config)
        raise ValueError(f"unsupported phase-field backend: {backend.value}")

    def run(
        self,
        initial_state: PhaseFieldState,
        *,
        callbacks: Iterable[PhaseFieldCallback] = (),
    ) -> PhaseFieldResult:
        return self.build_solver(initial_state).run(initial_state, callbacks=callbacks)


def simulate(
    model: PhaseFieldModel,
    initial_state: PhaseFieldState,
    config: SolverConfig,
    *,
    callbacks: Iterable[PhaseFieldCallback] = (),
) -> PhaseFieldResult:
    return PhaseFieldSimulation(model, config).run(initial_state, callbacks=callbacks)


__all__ = ["PhaseFieldSimulation", "simulate"]
