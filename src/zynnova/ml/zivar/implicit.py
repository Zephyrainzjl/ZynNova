"""Implicit KKT differentiation for the quadratic reference SCF problem."""

from __future__ import annotations

from typing import Any

from ._deps import require_torch
from .operators import DenseLinearOperator
from .scf import SCFSolverConfig, solve_quadratic_scf

torch = require_torch()


class _ImplicitQuadraticSolve(torch.autograd.Function):
    """Custom autograd rule that solves the transposed KKT system in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        hessian: Any,
        linear: Any,
        constraint: Any,
        target: Any,
        config: SCFSolverConfig,
    ) -> Any:
        result = solve_quadratic_scf(
            DenseLinearOperator(hessian),
            linear,
            constraint=constraint,
            target=target,
            config=config,
        )
        ctx.config = config
        ctx.save_for_backward(
            hessian,
            result.solution,
            constraint,
            result.lagrange_multipliers,
        )
        return result.solution

    @staticmethod
    def backward(ctx: Any, gradient_output: Any) -> tuple[Any, Any, Any, Any, None]:
        hessian, solution, constraint, multipliers = ctx.saved_tensors
        zero_target = constraint.new_zeros(constraint.shape[0])
        # min_u 0.5 u^T H u - g^T u, C u = 0 gives
        # H u + C^T v = g: exactly the transposed KKT adjoint system.
        adjoint = solve_quadratic_scf(
            DenseLinearOperator(hessian),
            -gradient_output,
            constraint=constraint,
            target=zero_target,
            config=ctx.config,
        )
        vector = adjoint.solution
        adjoint_multiplier = adjoint.lagrange_multipliers
        gradient_hessian = -torch.outer(vector, solution)
        gradient_linear = -vector
        gradient_constraint = -(
            torch.outer(multipliers, vector)
            + torch.outer(adjoint_multiplier, solution)
        )
        gradient_target = adjoint_multiplier
        return (
            gradient_hessian,
            gradient_linear,
            gradient_constraint,
            gradient_target,
            None,
        )


def implicit_quadratic_solve(
    hessian: Any,
    linear: Any,
    *,
    constraint: Any | None = None,
    target: Any | None = None,
    config: SCFSolverConfig | None = None,
) -> Any:
    """Return the constrained minimiser with an implicit KKT backward rule.

    This dense-Hessian entry point is the small-system differentiable oracle.
    Production matrix-free functionals use the same adjoint equation while
    exposing their parameters through specialised operator autograd kernels.
    The Hessian is symmetrised explicitly, so its gradient has the correct
    symmetric parameterisation.
    """

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("implicit SCF Hessian must be square")
    size = int(hessian.shape[0])
    if linear.shape != (size,):
        raise ValueError(f"implicit SCF linear coefficient must have shape [{size}]")
    if hessian.dtype != linear.dtype or hessian.device != linear.device:
        raise ValueError("implicit SCF Hessian and linear coefficient must match")
    if constraint is None:
        if target is not None:
            raise ValueError("constraint target requires a constraint matrix")
        constraint = linear.new_empty((0, size))
        target = linear.new_empty((0,))
    elif target is None:
        raise ValueError("constraint matrix requires a target")
    if constraint.shape[1:] != (size,):
        raise ValueError(f"implicit SCF constraint must have shape [K,{size}]")
    if target is None:  # Narrowing for static type checkers.
        raise AssertionError("unreachable missing target")
    if target.shape != (constraint.shape[0],):
        raise ValueError("implicit SCF target must have shape [K]")
    tensors = (constraint, target)
    if any(value.dtype != linear.dtype or value.device != linear.device for value in tensors):
        raise ValueError("all implicit SCF tensors must share dtype and device")
    if constraint.layout != torch.strided:
        raise ValueError("implicit dense oracle requires a strided constraint tensor")
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    return _ImplicitQuadraticSolve.apply(
        symmetric_hessian,
        linear,
        constraint,
        target,
        config or SCFSolverConfig(),
    )


__all__ = ["implicit_quadratic_solve"]
