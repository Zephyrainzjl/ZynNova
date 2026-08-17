"""Matrix-free linear algebra primitives for variational electronic solves.

The production SCF path must never require an ``N x N`` materialisation.  This
module therefore exposes the small protocol needed by the solver while still
providing dense implementations for tests and small-system reference work.
Constraint matrices are allowed to be sparse; only their small ``K x K`` Gram
matrix is factorised.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ._deps import require_torch

torch = require_torch()


def _check_vector_or_matrix(value: Any, size: int, *, name: str) -> None:
    if value.ndim not in {1, 2} or value.shape[0] != size:
        raise ValueError(f"{name} must have shape [{size}] or [{size}, R]")


def _sparse_mm(matrix: Any, value: Any) -> Any:
    """Multiply a two-dimensional dense or sparse tensor by one or more RHSs."""

    vector = value.ndim == 1
    right = value[:, None] if vector else value
    if matrix.layout == torch.strided:
        result = matrix @ right
    else:
        sparse = matrix if matrix.layout == torch.sparse_coo else matrix.to_sparse_coo()
        result = torch.sparse.mm(sparse, right)
    return result[:, 0] if vector else result


@runtime_checkable
class MatrixFreeOperator(Protocol):
    """Symmetric linear operator consumed by the constrained SCF solver."""

    shape: tuple[int, int]
    dtype: Any
    device: Any

    def matvec(self, value: Any) -> Any:
        """Apply the operator to ``[N]`` or ``[N, R]`` values."""


class DenseLinearOperator:
    """Small-system reference operator backed by a symmetric dense matrix."""

    def __init__(self, matrix: Any) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("operator matrix must be square")
        if not matrix.is_floating_point():
            raise TypeError("operator matrix must use a floating dtype")
        if not bool(torch.isfinite(matrix).all().detach()):
            raise ValueError("operator matrix contains non-finite values")
        tolerance = 128.0 * torch.finfo(matrix.dtype).eps
        scale = max(1.0, float(matrix.detach().abs().max().item()))
        asymmetry = float((matrix - matrix.T).detach().abs().max().item())
        if asymmetry > tolerance * scale:
            raise ValueError("operator matrix must be symmetric")
        self.matrix = matrix
        self.shape = (int(matrix.shape[0]), int(matrix.shape[1]))
        self.dtype = matrix.dtype
        self.device = matrix.device

    def matvec(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.shape[1], name="operator input")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError("operator input must match operator dtype and device")
        return self.matrix @ value


class CallableLinearOperator:
    """Matrix-free operator defined by a tensor-preserving callback."""

    def __init__(
        self,
        size: int,
        matvec_fn: Callable[[Any], Any],
        *,
        dtype: Any,
        device: Any,
    ) -> None:
        if size < 1:
            raise ValueError("operator size must be positive")
        self.shape = (int(size), int(size))
        self.dtype = dtype
        self.device = torch.device(device)
        self._matvec_fn = matvec_fn

    def matvec(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.shape[1], name="operator input")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError("operator input must match operator dtype and device")
        result = self._matvec_fn(value)
        if result.shape != value.shape:
            raise ValueError("operator callback returned an incompatible shape")
        if result.device != value.device or result.dtype != value.dtype:
            raise ValueError("operator callback changed dtype or device")
        return result


@runtime_checkable
class Preconditioner(Protocol):
    """Positive-definite approximate inverse used by projected PCG."""

    shape: tuple[int, int]
    dtype: Any
    device: Any

    def apply(self, value: Any) -> Any:
        """Apply the approximate inverse to ``[N]`` or ``[N, R]`` values."""


class IdentityPreconditioner:
    """Identity preconditioner, useful as a safe matrix-free default."""

    def __init__(self, size: int, *, dtype: Any, device: Any) -> None:
        if size < 1:
            raise ValueError("preconditioner size must be positive")
        self.shape = (int(size), int(size))
        self.dtype = dtype
        self.device = torch.device(device)

    def apply(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.shape[1], name="preconditioner input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("preconditioner input must match dtype and device")
        return value


class DiagonalPreconditioner:
    """Inverse of a strictly positive diagonal approximation."""

    def __init__(self, diagonal: Any) -> None:
        if diagonal.ndim != 1 or diagonal.numel() == 0:
            raise ValueError("preconditioner diagonal must have shape [N]")
        if not diagonal.is_floating_point():
            raise TypeError("preconditioner diagonal must use a floating dtype")
        if not bool(torch.isfinite(diagonal).all().detach()):
            raise ValueError("preconditioner diagonal contains non-finite values")
        if bool(torch.any(diagonal.detach() <= 0.0)):
            raise ValueError("preconditioner diagonal must be strictly positive")
        self.diagonal = diagonal
        size = int(diagonal.numel())
        self.shape = (size, size)
        self.dtype = diagonal.dtype
        self.device = diagonal.device

    def apply(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.shape[1], name="preconditioner input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("preconditioner input must match dtype and device")
        divisor = self.diagonal if value.ndim == 1 else self.diagonal[:, None]
        return value / divisor


class BlockDiagonalPreconditioner:
    """Exact inverse of contiguous small SPD blocks.

    The blocks normally correspond to the onsite ``q/p/Q/m`` response of one
    atom.  They are factorised independently and never assembled into a global
    dense matrix.
    """

    def __init__(self, blocks: Sequence[Any]) -> None:
        if not blocks:
            raise ValueError("at least one preconditioner block is required")
        factors: list[Any] = []
        sizes: list[int] = []
        reference = blocks[0]
        if not reference.is_floating_point():
            raise TypeError("preconditioner blocks must use a floating dtype")
        for block in blocks:
            if block.ndim != 2 or block.shape[0] != block.shape[1] or block.numel() == 0:
                raise ValueError("every preconditioner block must be non-empty and square")
            if block.dtype != reference.dtype or block.device != reference.device:
                raise ValueError("all preconditioner blocks must share dtype and device")
            tolerance = 128.0 * torch.finfo(block.dtype).eps
            scale = max(1.0, float(block.detach().abs().max().item()))
            if float((block - block.T).detach().abs().max().item()) > tolerance * scale:
                raise ValueError("preconditioner blocks must be symmetric")
            factor, info = torch.linalg.cholesky_ex(block)
            if int(info.detach().item()) != 0:
                raise ValueError("preconditioner blocks must be positive definite")
            factors.append(factor)
            sizes.append(int(block.shape[0]))
        self._factors = tuple(factors)
        self._sizes = tuple(sizes)
        size = sum(sizes)
        self.shape = (size, size)
        self.dtype = reference.dtype
        self.device = reference.device

    def apply(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.shape[1], name="preconditioner input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("preconditioner input must match dtype and device")
        vector = value.ndim == 1
        right = value[:, None] if vector else value
        solved: list[Any] = []
        start = 0
        for size, factor in zip(self._sizes, self._factors, strict=True):
            solved.append(torch.cholesky_solve(right[start : start + size], factor))
            start += size
        result = torch.cat(solved, dim=0)
        return result[:, 0] if vector else result


class ConstraintProjector:
    """Euclidean projector for a small full-row-rank constraint ``C x = d``.

    ``C`` may be strided, COO, CSR, CSC, BSR or BSC.  Only ``C C^T`` is dense,
    so total-charge and fragment constraints remain linear-memory operations.
    Singular or inconsistent specifications fail closed instead of silently
    using a pseudoinverse.
    """

    def __init__(self, constraint: Any, n_variables: int | None = None) -> None:
        if constraint.ndim != 2:
            raise ValueError("constraint must have shape [K,N]")
        if not constraint.is_floating_point():
            raise TypeError("constraint must use a floating dtype")
        if constraint.layout == torch.sparse_coo:
            constraint = constraint.coalesce()
        rows, columns = (int(constraint.shape[0]), int(constraint.shape[1]))
        if n_variables is not None and columns != n_variables:
            raise ValueError("constraint width does not match the variable count")
        if rows > columns:
            raise ValueError("constraint cannot have more rows than variables")
        values = constraint if constraint.layout == torch.strided else constraint.values()
        if not bool(torch.isfinite(values).all().detach()):
            raise ValueError("constraint contains non-finite values")
        self.constraint = constraint
        self.n_constraints = rows
        self.n_variables = columns
        self.dtype = constraint.dtype
        self.device = constraint.device
        if rows == 0:
            self._gram_factor = constraint.new_zeros((0, 0))
            return
        transpose_dense = constraint.transpose(0, 1).to_dense()
        gram = _sparse_mm(constraint, transpose_dense)
        gram = 0.5 * (gram + gram.T)
        factor, info = torch.linalg.cholesky_ex(gram)
        if int(info.detach().item()) != 0:
            raise ValueError("constraint rows must be linearly independent")
        self._gram_factor = factor

    def _check_target(self, target: Any) -> None:
        if target.shape != (self.n_constraints,):
            raise ValueError(f"constraint target must have shape [{self.n_constraints}]")
        if target.dtype != self.dtype or target.device != self.device:
            raise ValueError("constraint target must match constraint dtype and device")
        if not bool(torch.isfinite(target).all().detach()):
            raise ValueError("constraint target contains non-finite values")

    def apply_constraint(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.n_variables, name="constraint input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("constraint input must match dtype and device")
        return _sparse_mm(self.constraint, value)

    def apply_adjoint(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.n_constraints, name="constraint adjoint input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("constraint adjoint input must match dtype and device")
        return _sparse_mm(self.constraint.transpose(0, 1), value)

    def solve_gram(self, value: Any) -> Any:
        _check_vector_or_matrix(value, self.n_constraints, name="constraint Gram input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("constraint Gram input must match dtype and device")
        if self.n_constraints == 0:
            return value
        vector = value.ndim == 1
        right = value[:, None] if vector else value
        result = torch.cholesky_solve(right, self._gram_factor)
        return result[:, 0] if vector else result

    def project(self, value: Any) -> Any:
        """Project a vector onto ``null(C)``."""

        _check_vector_or_matrix(value, self.n_variables, name="projection input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("projection input must match dtype and device")
        if self.n_constraints == 0:
            return value
        return value - self.apply_adjoint(self.solve_gram(self.apply_constraint(value)))

    def make_feasible(self, value: Any, target: Any) -> Any:
        """Return the nearest value satisfying ``C x = target``."""

        _check_vector_or_matrix(value, self.n_variables, name="feasibility input")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError("feasibility input must match dtype and device")
        self._check_target(target)
        if self.n_constraints == 0:
            return value
        correction = target - self.apply_constraint(value)
        return value + self.apply_adjoint(self.solve_gram(correction))

    def particular(self, target: Any) -> Any:
        """Return the minimum-norm solution of ``C x = target``."""

        self._check_target(target)
        zero = target.new_zeros(self.n_variables)
        return self.make_feasible(zero, target)

    def multipliers(self, gradient: Any) -> Any:
        """Return ``lambda`` minimising ``|gradient + C^T lambda|``."""

        _check_vector_or_matrix(gradient, self.n_variables, name="gradient")
        return -self.solve_gram(self.apply_constraint(gradient))

    def residual(self, value: Any, target: Any) -> Any:
        self._check_target(target)
        return self.apply_constraint(value) - target


__all__ = [
    "BlockDiagonalPreconditioner",
    "CallableLinearOperator",
    "ConstraintProjector",
    "DenseLinearOperator",
    "DiagonalPreconditioner",
    "IdentityPreconditioner",
    "MatrixFreeOperator",
    "Preconditioner",
]
