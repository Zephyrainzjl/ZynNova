"""Matrix-free monolithic Newton--Krylov infrastructure.

The original multiphysics solver is partitioned.  This module provides the
missing monolithic nonlinear infrastructure while keeping physics-specific
residuals injectable.  It is suitable for electrochemical--thermal--mechanical
systems whose state is packed into one vector and whose Jacobian is too large
to assemble explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


ResidualFunction = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]
JacobianVectorFunction = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], np.ndarray]
PreconditionerFunction = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True, slots=True)
class FieldSlice:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


class FieldLayout:
    """Deterministically pack and unpack named multiphysics fields."""

    def __init__(self, fields: Mapping[str, np.ndarray | tuple[int, ...]]) -> None:
        slices: list[FieldSlice] = []
        cursor = 0
        for name, value in fields.items():
            shape = tuple(value) if isinstance(value, tuple) else tuple(np.asarray(value).shape)
            size = int(np.prod(shape, dtype=np.int64))
            slices.append(FieldSlice(str(name), shape, cursor, cursor + size))
            cursor += size
        if cursor == 0:
            raise ValueError("field layout cannot be empty")
        self.fields = tuple(slices)
        self.size = cursor
        self._by_name = {field.name: field for field in self.fields}

    def pack(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        vector = np.empty(self.size, dtype=float)
        for field in self.fields:
            if field.name not in values:
                raise KeyError(f"missing field {field.name!r}")
            array = np.asarray(values[field.name], dtype=float)
            if array.shape != field.shape:
                raise ValueError(f"field {field.name!r} has shape {array.shape}, expected {field.shape}")
            vector[field.start : field.stop] = array.reshape(-1)
        return vector

    def unpack(self, vector: np.ndarray, *, copy: bool = False) -> dict[str, np.ndarray]:
        flat = np.asarray(vector, dtype=float).reshape(-1)
        if flat.shape != (self.size,):
            raise ValueError("packed vector has wrong size")
        result = {}
        for field in self.fields:
            view = flat[field.start : field.stop].reshape(field.shape)
            result[field.name] = view.copy() if copy else view
        return result

    def slice(self, name: str) -> slice:
        field = self._by_name[name]
        return slice(field.start, field.stop)


@dataclass(frozen=True, slots=True)
class MonolithicNewtonKrylovConfig:
    nonlinear_relative_tolerance: float = 1.0e-8
    nonlinear_absolute_tolerance: float = 1.0e-10
    maximum_newton_iterations: int = 30
    linear_relative_tolerance: float = 1.0e-3
    maximum_linear_iterations: int = 500
    finite_difference_relative_step: float = 1.0e-7
    line_search_minimum: float = 1.0e-6
    line_search_reduction: float = 0.5
    armijo_coefficient: float = 1.0e-4
    gmres_restart: int = 80

    def __post_init__(self) -> None:
        positive = (
            self.nonlinear_relative_tolerance,
            self.nonlinear_absolute_tolerance,
            self.linear_relative_tolerance,
            self.finite_difference_relative_step,
            self.line_search_minimum,
            self.line_search_reduction,
            self.armijo_coefficient,
        )
        if min(positive) <= 0.0:
            raise ValueError("Newton--Krylov tolerances must be positive")
        if self.maximum_newton_iterations < 1 or self.maximum_linear_iterations < 1 or self.gmres_restart < 2:
            raise ValueError("iteration limits are invalid")
        if self.line_search_reduction >= 1.0 or self.armijo_coefficient >= 1.0:
            raise ValueError("line-search factors must be below one")


@dataclass(frozen=True, slots=True)
class MonolithicIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    linear_iterations: int
    step_length: float


@dataclass(frozen=True, slots=True)
class MonolithicSolution:
    vector: np.ndarray
    converged: bool
    residual_norm: float
    iterations: tuple[MonolithicIteration, ...]


class MonolithicNewtonKrylovSolver:
    """Inexact Newton solve with matrix-free GMRES and Armijo globalization."""

    def __init__(
        self,
        residual: ResidualFunction,
        *,
        jacobian_vector: JacobianVectorFunction | None = None,
        preconditioner: PreconditionerFunction | None = None,
        config: MonolithicNewtonKrylovConfig | None = None,
    ) -> None:
        self.residual = residual
        self.jacobian_vector = jacobian_vector
        self.preconditioner = preconditioner
        self.config = config or MonolithicNewtonKrylovConfig()

    def solve(
        self,
        initial: np.ndarray,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> MonolithicSolution:
        try:
            from scipy.sparse.linalg import LinearOperator, gmres
        except ImportError as exc:  # pragma: no cover
            raise ImportError("monolithic Newton--Krylov solve requires scipy") from exc
        ctx = dict(context or {})
        x = np.asarray(initial, dtype=float).reshape(-1).copy()
        residual = np.asarray(self.residual(x, ctx), dtype=float).reshape(-1)
        if residual.shape != x.shape or not np.isfinite(residual).all():
            raise ValueError("initial residual is invalid")
        initial_norm = max(float(np.linalg.norm(residual)), self.config.nonlinear_absolute_tolerance)
        history: list[MonolithicIteration] = []
        for iteration in range(1, self.config.maximum_newton_iterations + 1):
            norm = float(np.linalg.norm(residual))
            relative = norm / initial_norm
            if norm <= self.config.nonlinear_absolute_tolerance or relative <= self.config.nonlinear_relative_tolerance:
                return MonolithicSolution(x, True, norm, tuple(history))

            def matvec(direction: np.ndarray) -> np.ndarray:
                vector = np.asarray(direction, dtype=float)
                if self.jacobian_vector is not None:
                    return np.asarray(self.jacobian_vector(x, vector, ctx), dtype=float)
                scale = self.config.finite_difference_relative_step * (1.0 + np.linalg.norm(x)) / max(np.linalg.norm(vector), 1.0e-30)
                return (np.asarray(self.residual(x + scale * vector, ctx), dtype=float) - residual) / scale

            operator = LinearOperator((x.size, x.size), matvec=matvec, dtype=float)
            preconditioner = None
            if self.preconditioner is not None:
                preconditioner = LinearOperator(
                    (x.size, x.size),
                    matvec=lambda vector: np.asarray(self.preconditioner(x, vector, ctx), dtype=float),
                    dtype=float,
                )
            linear_count = 0

            def callback(_: Any) -> None:
                nonlocal linear_count
                linear_count += 1

            step, info = gmres(
                operator,
                -residual,
                M=preconditioner,
                rtol=min(self.config.linear_relative_tolerance, max(0.1 * relative, 1.0e-8)),
                atol=0.0,
                restart=self.config.gmres_restart,
                maxiter=self.config.maximum_linear_iterations,
                callback=callback,
                callback_type="pr_norm",
            )
            if info < 0 or not np.isfinite(step).all():
                raise RuntimeError(f"GMRES failed in Newton iteration {iteration}: info={info}")
            alpha = 1.0
            merit = 0.5 * norm * norm
            accepted = False
            while alpha >= self.config.line_search_minimum:
                candidate = x + alpha * step
                candidate_residual = np.asarray(self.residual(candidate, ctx), dtype=float).reshape(-1)
                if np.isfinite(candidate_residual).all():
                    candidate_merit = 0.5 * float(np.dot(candidate_residual, candidate_residual))
                    if candidate_merit <= (1.0 - self.config.armijo_coefficient * alpha) * merit:
                        x = candidate
                        residual = candidate_residual
                        accepted = True
                        break
                alpha *= self.config.line_search_reduction
            if not accepted:
                raise RuntimeError(f"Newton line search failed at iteration {iteration}")
            history.append(MonolithicIteration(iteration, norm, relative, linear_count, alpha))
        final = float(np.linalg.norm(residual))
        return MonolithicSolution(x, False, final, tuple(history))


__all__ = [
    "FieldLayout",
    "FieldSlice",
    "MonolithicIteration",
    "MonolithicNewtonKrylovConfig",
    "MonolithicNewtonKrylovSolver",
    "MonolithicSolution",
]
