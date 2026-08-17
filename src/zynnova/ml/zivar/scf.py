"""Fail-closed matrix-free solver for constrained quadratic SCF functionals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .errors import SCFConvergenceError
from .operators import (
    ConstraintProjector,
    IdentityPreconditioner,
    MatrixFreeOperator,
    Preconditioner,
)

torch = require_torch()


@dataclass(frozen=True, slots=True)
class SCFSolverConfig:
    """Numerical controls for the projected preconditioned-CG solve."""

    atol: float = 1.0e-10
    rtol: float = 1.0e-8
    energy_atol: float = 1.0e-12
    max_iter: int = 256
    constraint_atol: float = 1.0e-10
    recompute_interval: int = 16
    curvature_tolerance: float = 0.0

    def __post_init__(self) -> None:
        finite_nonnegative = (
            self.atol,
            self.rtol,
            self.energy_atol,
            self.constraint_atol,
            self.curvature_tolerance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite_nonnegative):
            raise ValueError("SCF tolerances must be finite and non-negative")
        if self.atol == 0.0 and self.rtol == 0.0:
            raise ValueError("at least one SCF residual tolerance must be positive")
        if self.energy_atol <= 0.0:
            raise ValueError("SCF energy tolerance must be positive")
        if self.max_iter < 1:
            raise ValueError("SCF max_iter must be positive")
        if self.recompute_interval < 1:
            raise ValueError("SCF recompute_interval must be positive")


@dataclass(frozen=True, slots=True)
class SCFReport:
    """Detached convergence evidence safe to log or serialise."""

    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    constraint_residual: float
    energy_change: float
    energy_error: float
    termination: str
    residual_history: tuple[float, ...]
    energy_history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SCFResult:
    """Stationary state and energy returned only after all gates pass."""

    solution: Any
    energy: Any
    lagrange_multipliers: Any
    report: SCFReport


def _detached_float(value: Any) -> float:
    return float(value.detach().item())


def _rms(value: Any) -> Any:
    return torch.linalg.vector_norm(value) / math.sqrt(max(1, int(value.numel())))


def _max_abs(value: Any) -> Any:
    if value.numel() == 0:
        return value.new_zeros(())
    return value.abs().max()


def _validate_finite(value: Any, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().detach()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _make_report(
    *,
    converged: bool,
    iterations: int,
    initial_residual: float,
    final_residual: float,
    constraint_residual: float,
    energy_change: float,
    termination: str,
    residual_history: list[float],
    energy_history: list[float],
    energy_error: float = math.inf,
) -> SCFReport:
    tiny = torch.finfo(torch.float64).tiny
    relative = final_residual / max(initial_residual, tiny)
    if initial_residual == 0.0:
        relative = 0.0 if final_residual == 0.0 else math.inf
    return SCFReport(
        converged=converged,
        iterations=iterations,
        initial_residual=initial_residual,
        final_residual=final_residual,
        relative_residual=relative,
        constraint_residual=constraint_residual,
        energy_change=energy_change,
        energy_error=energy_error,
        termination=termination,
        residual_history=tuple(residual_history),
        energy_history=tuple(energy_history),
    )


def _raise_failure(
    *,
    termination: str,
    iterations: int,
    initial_residual: float,
    final_residual: float,
    constraint_residual: float,
    energy_change: float,
    residual_history: list[float],
    energy_history: list[float],
    energy_error: float = math.inf,
) -> None:
    report = _make_report(
        converged=False,
        iterations=iterations,
        initial_residual=initial_residual,
        final_residual=final_residual,
        constraint_residual=constraint_residual,
        energy_change=energy_change,
        termination=termination,
        residual_history=residual_history,
        energy_history=energy_history,
        energy_error=energy_error,
    )
    raise SCFConvergenceError(
        "SCF did not converge: "
        f"{report.termination}; iterations={report.iterations}, "
        f"residual={report.final_residual:.6e}, "
        f"constraint_residual={report.constraint_residual:.6e}, "
        f"energy_error={report.energy_error:.6e}",
        report=report,
    )


def _validate_problem(
    operator: MatrixFreeOperator,
    linear: Any,
    preconditioner: Preconditioner,
) -> int:
    if linear.ndim != 1 or linear.numel() == 0:
        raise ValueError("SCF linear coefficient must have shape [N]")
    if not linear.is_floating_point():
        raise TypeError("SCF coefficients must use a floating dtype")
    size = int(linear.numel())
    if tuple(operator.shape) != (size, size):
        raise ValueError("SCF operator shape does not match the linear coefficient")
    if operator.dtype != linear.dtype or torch.device(operator.device) != linear.device:
        raise ValueError("SCF operator must match coefficient dtype and device")
    if tuple(preconditioner.shape) != (size, size):
        raise ValueError("SCF preconditioner shape does not match the operator")
    if (
        preconditioner.dtype != linear.dtype
        or torch.device(preconditioner.device) != linear.device
    ):
        raise ValueError("SCF preconditioner must match coefficient dtype and device")
    _validate_finite(linear, name="SCF linear coefficient")
    return size


def solve_quadratic_scf(
    operator: MatrixFreeOperator,
    linear: Any,
    *,
    constraint: Any | None = None,
    target: Any | None = None,
    preconditioner: Preconditioner | None = None,
    warm_start: Any | None = None,
    constant: Any | float | None = None,
    config: SCFSolverConfig | None = None,
) -> SCFResult:
    """Minimise ``0.5*x^T A*x + linear^T*x + constant`` subject to ``C*x=d``.

    ``A`` is accessed exclusively through ``operator.matvec``.  The returned
    state is accepted only when the projected-gradient, directional
    stationary-energy estimate, and constraint residuals satisfy their
    configured tolerances.  Failure never returns the last iterate as a
    result.
    """

    settings = config or SCFSolverConfig()
    if preconditioner is None:
        preconditioner = IdentityPreconditioner(
            int(linear.numel()), dtype=linear.dtype, device=linear.device
        )
    size = _validate_problem(operator, linear, preconditioner)

    if constraint is None:
        if target is not None:
            raise ValueError("constraint target requires a constraint matrix")
        constraint = linear.new_empty((0, size))
        target = linear.new_empty((0,))
    elif target is None:
        raise ValueError("constraint matrix requires a target")
    projector = ConstraintProjector(constraint, size)
    if target is None:  # Narrowing for static type checkers.
        raise AssertionError("unreachable missing target")
    projector._check_target(target)

    if warm_start is None:
        solution = linear.new_zeros(size)
    else:
        if warm_start.shape != (size,):
            raise ValueError(f"SCF warm_start must have shape [{size}]")
        if warm_start.dtype != linear.dtype or warm_start.device != linear.device:
            raise ValueError("SCF warm_start must match coefficient dtype and device")
        _validate_finite(warm_start, name="SCF warm_start")
        solution = warm_start
    solution = projector.make_feasible(solution, target)

    if constant is None:
        constant_tensor = linear.new_zeros(())
    else:
        constant_tensor = torch.as_tensor(constant, dtype=linear.dtype, device=linear.device)
        if constant_tensor.ndim != 0:
            raise ValueError("SCF constant energy must be scalar")
        _validate_finite(constant_tensor, name="SCF constant energy")

    def apply_operator(value: Any) -> Any:
        result = operator.matvec(value)
        if result.shape != value.shape:
            raise ValueError("SCF operator returned an incompatible shape")
        if result.dtype != value.dtype or result.device != value.device:
            raise ValueError("SCF operator changed dtype or device")
        _validate_finite(result, name="SCF operator result")
        return result

    def energy(value: Any, applied: Any) -> Any:
        return 0.5 * torch.dot(value, applied) + torch.dot(linear, value) + constant_tensor

    def stationarity_energy_error(projected_residual: Any) -> float:
        """Return the exact energy lowering along one preconditioned direction."""

        residual_max = _detached_float(_max_abs(projected_residual))
        numerical_floor = 64.0 * torch.finfo(projected_residual.dtype).eps * max(
            1.0, initial_residual
        )
        if residual_max <= numerical_floor:
            return 0.0
        direction = projector.project(preconditioner.apply(projected_residual))
        _validate_finite(direction, name="SCF energy-error direction")
        numerator = torch.dot(projected_residual, direction)
        numerator_value = _detached_float(numerator)
        if not math.isfinite(numerator_value):
            return math.inf
        if numerator_value <= 0.0:
            # Constraint projection can make an O(eps) preconditioned dot
            # product slightly negative at an already stationary solution.
            # The raw projected residual remains a valid descent direction
            # and avoids turning round-off into a false non-convergence.
            direction = projected_residual
            numerator = torch.dot(projected_residual, projected_residual)
            numerator_value = _detached_float(numerator)
            if numerator_value == 0.0:
                return 0.0
        applied = projector.project(apply_operator(direction))
        curvature = _detached_float(torch.dot(direction, applied))
        if curvature <= 0.0 or not math.isfinite(curvature):
            return math.inf
        return 0.5 * numerator_value * numerator_value / curvature

    applied_solution = apply_operator(solution)
    gradient = applied_solution + linear
    residual = -projector.project(gradient)
    residual_norm = _rms(residual)
    initial_residual = _detached_float(residual_norm)
    current_energy = energy(solution, applied_solution)
    _validate_finite(current_energy, name="SCF energy")
    residual_history = [initial_residual]
    energy_history = [_detached_float(current_energy)]
    constraint_norm = _detached_float(_max_abs(projector.residual(solution, target)))
    threshold = settings.atol + settings.rtol * initial_residual

    energy_error_value = (
        stationarity_energy_error(residual)
        if initial_residual <= threshold
        else math.inf
    )
    if initial_residual <= threshold and energy_error_value <= settings.energy_atol:
        if constraint_norm > settings.constraint_atol:
            _raise_failure(
                termination="constraint tolerance exceeded",
                iterations=0,
                initial_residual=initial_residual,
                final_residual=initial_residual,
                constraint_residual=constraint_norm,
                energy_change=0.0,
                residual_history=residual_history,
                energy_history=energy_history,
                energy_error=energy_error_value,
            )
        multipliers = projector.multipliers(gradient)
        report = _make_report(
            converged=True,
            iterations=0,
            initial_residual=initial_residual,
            final_residual=initial_residual,
            constraint_residual=constraint_norm,
            energy_change=0.0,
            termination="converged",
            residual_history=residual_history,
            energy_history=energy_history,
            energy_error=energy_error_value,
        )
        return SCFResult(solution, current_energy, multipliers, report)

    preconditioned = preconditioner.apply(residual)
    _validate_finite(preconditioned, name="SCF preconditioner result")
    preconditioned = projector.project(preconditioned)
    rho = torch.dot(residual, preconditioned)
    rho_value = _detached_float(rho)
    if rho_value <= 0.0 or not math.isfinite(rho_value):
        _raise_failure(
            termination="non-positive preconditioned residual",
            iterations=0,
            initial_residual=initial_residual,
            final_residual=initial_residual,
            constraint_residual=constraint_norm,
            energy_change=0.0,
            residual_history=residual_history,
            energy_history=energy_history,
        )
    direction = preconditioned
    energy_change = 0.0

    for iteration in range(1, settings.max_iter + 1):
        applied_direction = apply_operator(direction)
        projected_applied = projector.project(applied_direction)
        curvature = torch.dot(direction, projected_applied)
        curvature_value = _detached_float(curvature)
        scale = _detached_float(
            torch.linalg.vector_norm(direction)
            * torch.linalg.vector_norm(projected_applied)
        )
        negative_limit = -settings.curvature_tolerance * max(
            scale, torch.finfo(linear.dtype).tiny
        )
        if not math.isfinite(curvature_value) or curvature_value <= negative_limit:
            _raise_failure(
                termination="negative curvature",
                iterations=iteration - 1,
                initial_residual=initial_residual,
                final_residual=residual_history[-1],
                constraint_residual=constraint_norm,
                energy_change=energy_change,
                residual_history=residual_history,
                energy_history=energy_history,
            )
        breakdown_limit = 32.0 * torch.finfo(linear.dtype).eps * max(
            scale, torch.finfo(linear.dtype).tiny
        )
        if curvature_value <= breakdown_limit:
            _raise_failure(
                termination="Krylov curvature breakdown",
                iterations=iteration - 1,
                initial_residual=initial_residual,
                final_residual=residual_history[-1],
                constraint_residual=constraint_norm,
                energy_change=energy_change,
                residual_history=residual_history,
                energy_history=energy_history,
            )

        step = rho / curvature
        solution = solution + step * direction
        applied_solution = applied_solution + step * applied_direction
        if iteration % settings.recompute_interval == 0:
            solution = projector.make_feasible(solution, target)
            applied_solution = apply_operator(solution)
            residual = -projector.project(applied_solution + linear)
        else:
            residual = projector.project(residual - step * projected_applied)

        next_energy = energy(solution, applied_solution)
        _validate_finite(next_energy, name="SCF energy")
        energy_change = abs(_detached_float(next_energy - current_energy))
        current_energy = next_energy
        residual_norm = _rms(residual)
        residual_value = _detached_float(residual_norm)
        residual_history.append(residual_value)
        energy_history.append(_detached_float(current_energy))

        if residual_value <= threshold:
            solution = projector.make_feasible(solution, target)
            applied_solution = apply_operator(solution)
            gradient = applied_solution + linear
            residual = -projector.project(gradient)
            residual_value = _detached_float(_rms(residual))
            constraint_norm = _detached_float(
                _max_abs(projector.residual(solution, target))
            )
            residual_history[-1] = residual_value
            current_energy = energy(solution, applied_solution)
            energy_history[-1] = _detached_float(current_energy)
            energy_error_value = stationarity_energy_error(residual)
            if (
                residual_value > threshold
                or energy_error_value > settings.energy_atol
            ):
                # A recursively updated residual was over-optimistic.  Restart
                # PCG from the explicitly recomputed residual.
                preconditioned = projector.project(preconditioner.apply(residual))
                rho = torch.dot(residual, preconditioned)
                rho_value = _detached_float(rho)
                if rho_value <= 0.0 or not math.isfinite(rho_value):
                    _raise_failure(
                        termination="non-positive restarted residual",
                        iterations=iteration,
                        initial_residual=initial_residual,
                        final_residual=residual_value,
                        constraint_residual=constraint_norm,
                        energy_change=energy_change,
                        residual_history=residual_history,
                        energy_history=energy_history,
                        energy_error=energy_error_value,
                    )
                direction = preconditioned
                continue
            if constraint_norm > settings.constraint_atol:
                _raise_failure(
                    termination="constraint tolerance exceeded",
                    iterations=iteration,
                    initial_residual=initial_residual,
                    final_residual=residual_value,
                    constraint_residual=constraint_norm,
                    energy_change=energy_change,
                    residual_history=residual_history,
                    energy_history=energy_history,
                    energy_error=energy_error_value,
                )
            multipliers = projector.multipliers(gradient)
            report = _make_report(
                converged=True,
                iterations=iteration,
                initial_residual=initial_residual,
                final_residual=residual_value,
                constraint_residual=constraint_norm,
                energy_change=energy_change,
                termination="converged",
                residual_history=residual_history,
                energy_history=energy_history,
                energy_error=energy_error_value,
            )
            return SCFResult(solution, current_energy, multipliers, report)

        preconditioned = preconditioner.apply(residual)
        _validate_finite(preconditioned, name="SCF preconditioner result")
        preconditioned = projector.project(preconditioned)
        next_rho = torch.dot(residual, preconditioned)
        next_rho_value = _detached_float(next_rho)
        if next_rho_value <= 0.0 or not math.isfinite(next_rho_value):
            _raise_failure(
                termination="non-positive preconditioned residual",
                iterations=iteration,
                initial_residual=initial_residual,
                final_residual=residual_value,
                constraint_residual=constraint_norm,
                energy_change=energy_change,
                residual_history=residual_history,
                energy_history=energy_history,
            )
        direction = preconditioned + (next_rho / rho) * direction
        rho = next_rho

    solution = projector.make_feasible(solution, target)
    applied_solution = apply_operator(solution)
    final_residual = _detached_float(
        _rms(projector.project(applied_solution + linear))
    )
    constraint_norm = _detached_float(_max_abs(projector.residual(solution, target)))
    residual_history[-1] = final_residual
    _raise_failure(
        termination="maximum iterations exceeded",
        iterations=settings.max_iter,
        initial_residual=initial_residual,
        final_residual=final_residual,
        constraint_residual=constraint_norm,
        energy_change=energy_change,
        residual_history=residual_history,
        energy_history=energy_history,
        energy_error=energy_error_value,
    )
    raise AssertionError("unreachable SCF failure")


__all__ = [
    "SCFConvergenceError",
    "SCFReport",
    "SCFResult",
    "SCFSolverConfig",
    "solve_quadratic_scf",
]
