"""Sparse linear solve utilities with symmetry-preserving constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import BackendUnavailableError, ConvergenceError
from .boundary import DirichletBC


@dataclass(slots=True)
class LinearSolveOptions:
    method: str = "direct"
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 0.0
    max_iterations: int | None = None
    preconditioner: str = "jacobi"
    check_finite: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"direct", "cg", "gmres"}:
            raise ValueError("method must be 'direct', 'cg', or 'gmres'")
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance < 0.0:
            raise ValueError("linear solver tolerances are invalid")
        if self.preconditioner not in {"none", "jacobi"}:
            raise ValueError("preconditioner must be 'none' or 'jacobi'")


def _scipy_sparse() -> tuple[Any, Any]:
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as exc:
        raise BackendUnavailableError(
            "sparse zynsim solves require SciPy; install zynnova[simulation]"
        ) from exc
    return sparse, sparse_linalg


def apply_dirichlet(
    matrix: Any,
    right_hand_side: np.ndarray,
    boundary: DirichletBC | list[DirichletBC] | tuple[DirichletBC, ...],
) -> tuple[Any, np.ndarray]:
    """Apply essential conditions while retaining a symmetric matrix.

    The constrained-column contribution is first moved to the right-hand side,
    after which constrained rows and columns are replaced by identity rows.
    """

    sparse, _ = _scipy_sparse()
    conditions = (boundary,) if isinstance(boundary, DirichletBC) else tuple(boundary)
    if not conditions:
        return matrix, np.asarray(right_hand_side, dtype=float).copy()
    dofs = np.concatenate([condition.dofs for condition in conditions])
    values = np.concatenate([condition.values for condition in conditions])
    if len(np.unique(dofs)) != len(dofs):
        raise ValueError("overlapping Dirichlet conditions must be merged explicitly")
    if np.max(dofs, initial=-1) >= matrix.shape[0]:
        raise ValueError("Dirichlet degree of freedom exceeds matrix dimensions")

    constrained = sparse.csr_matrix(matrix)
    rhs = np.asarray(right_hand_side, dtype=np.float64).reshape(-1).copy()
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("right-hand side size does not match matrix")
    rhs -= constrained[:, dofs] @ values
    editable = constrained.tolil(copy=True)
    editable[:, dofs] = 0.0
    editable[dofs, :] = 0.0
    editable[dofs, dofs] = 1.0
    rhs[dofs] = values
    return editable.tocsr(), rhs


def solve_linear(
    matrix: Any,
    right_hand_side: np.ndarray,
    options: LinearSolveOptions | None = None,
) -> np.ndarray:
    """Solve a real sparse linear system and verify the achieved residual."""

    sparse, sparse_linalg = _scipy_sparse()
    options = options or LinearSolveOptions()
    a = sparse.csr_matrix(matrix, dtype=np.float64)
    b = np.asarray(right_hand_side, dtype=np.float64).reshape(-1)
    if a.shape[0] != a.shape[1] or b.shape != (a.shape[0],):
        raise ValueError("linear system dimensions are inconsistent")
    if options.check_finite and (
        not np.all(np.isfinite(a.data)) or not np.all(np.isfinite(b))
    ):
        raise ValueError("linear system contains non-finite values")

    if options.method == "direct":
        solution = sparse_linalg.spsolve(a, b)
        info = 0
    else:
        preconditioner = None
        if options.preconditioner == "jacobi":
            diagonal = a.diagonal()
            safe = np.where(np.abs(diagonal) > 1.0e-30, 1.0 / diagonal, 1.0)
            preconditioner = sparse_linalg.LinearOperator(a.shape, matvec=lambda x: safe * x)
        common = {
            "rtol": options.relative_tolerance,
            "atol": options.absolute_tolerance,
            "maxiter": options.max_iterations,
            "M": preconditioner,
        }
        if options.method == "cg":
            solution, info = sparse_linalg.cg(a, b, **common)
        else:
            solution, info = sparse_linalg.gmres(a, b, **common)
    if info != 0:
        raise ConvergenceError(f"{options.method} linear solve failed with status {info}")
    solution = np.asarray(solution, dtype=np.float64)
    residual = np.linalg.norm(a @ solution - b)
    scale = max(np.linalg.norm(b), 1.0)
    threshold = max(
        20.0 * options.absolute_tolerance,
        20.0 * options.relative_tolerance * scale,
        1.0e-9 * scale if options.method == "direct" else 0.0,
    )
    if not np.all(np.isfinite(solution)) or residual > threshold:
        raise ConvergenceError(
            f"linear solution residual {residual:.3e} exceeds {threshold:.3e}"
        )
    return solution


__all__ = ["LinearSolveOptions", "apply_dirichlet", "solve_linear"]
