"""Differentiable nonlinear residual solves for discretized battery equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ResidualFunction = Callable[..., Any]


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "differentiable implicit solves require PyTorch; "
            "install zynnova[zynsim-inverse]"
        ) from exc
    return torch, nn


@dataclass(slots=True)
class DifferentiableImplicitConfig:
    maximum_iterations: int = 12
    tolerance: float = 1.0e-8
    damping: float = 1.0
    diagonal_regularization: float = 1.0e-10
    fail_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1 or self.tolerance <= 0.0:
            raise ValueError("implicit-solver iteration controls are invalid")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("implicit-solver damping must lie in (0,1]")
        if self.diagonal_regularization < 0.0:
            raise ValueError("diagonal regularization cannot be negative")


@dataclass(frozen=True, slots=True)
class ImplicitSolveDiagnostics:
    converged: bool
    iterations: int
    residual_norm: float


class DifferentiableImplicitSolver:
    """Factory for an unrolled Newton layer with end-to-end gradients.

    ``residual(state, *parameters)`` must return a tensor with the same number
    of scalar entries as ``state``.  Every Newton iteration is represented by
    Torch operations, so gradients propagate through the complete nonlinear
    solve.  This layer can wrap a Torch discretization of P2D, finite-element,
    thermal, mechanics, phase-field, or aging residuals.
    """

    def __new__(
        cls,
        residual: ResidualFunction,
        config: DifferentiableImplicitConfig | None = None,
    ):
        torch, nn = _torch_modules()
        resolved = config or DifferentiableImplicitConfig()

        class _ImplicitSolver(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.residual_function = residual
                self.config = resolved
                self.last_diagnostics = ImplicitSolveDiagnostics(
                    False, 0, float("inf")
                )

            def forward(self, initial_state: Any, *parameters: Any) -> Any:
                state = torch.as_tensor(initial_state)
                if not state.is_floating_point():
                    state = state.to(torch.get_default_dtype())
                original_shape = state.shape
                state = state.reshape(-1)
                converged = False
                residual_norm = float("inf")
                iterations = 0
                for iterations in range(1, self.config.maximum_iterations + 1):
                    if not state.requires_grad:
                        state = state.requires_grad_(True)

                    def flattened_residual(candidate: Any) -> Any:
                        value = self.residual_function(
                            candidate.reshape(original_shape), *parameters
                        )
                        return torch.as_tensor(
                            value, device=candidate.device, dtype=candidate.dtype
                        ).reshape(-1)

                    value = flattened_residual(state)
                    if value.numel() != state.numel():
                        raise ValueError(
                            "implicit residual and state must contain the same "
                            "number of scalar entries"
                        )
                    residual_norm_tensor = torch.linalg.vector_norm(value)
                    residual_norm = float(residual_norm_tensor.detach().cpu().item())
                    if residual_norm <= self.config.tolerance:
                        converged = True
                        break
                    jacobian = torch.autograd.functional.jacobian(
                        flattened_residual,
                        state,
                        create_graph=True,
                        vectorize=True,
                    )
                    identity = torch.eye(
                        state.numel(), device=state.device, dtype=state.dtype
                    )
                    system = (
                        jacobian
                        + self.config.diagonal_regularization * identity
                    )
                    update = torch.linalg.solve(system, -value)
                    state = state + self.config.damping * update
                self.last_diagnostics = ImplicitSolveDiagnostics(
                    converged, iterations, residual_norm
                )
                if not converged and self.config.fail_on_nonconvergence:
                    raise RuntimeError(
                        "differentiable implicit solve did not converge; "
                        f"residual={residual_norm:.3e}"
                    )
                return state.reshape(original_shape)

        return _ImplicitSolver()


__all__ = [
    "DifferentiableImplicitConfig",
    "DifferentiableImplicitSolver",
    "ImplicitSolveDiagnostics",
    "ResidualFunction",
]
