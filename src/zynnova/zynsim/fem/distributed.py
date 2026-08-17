"""Optional PETSc/MPI linear solves and deterministic mesh partitioning.

The serial SciPy backend remains the zero-dependency path.  This module adds a
strictly optional distributed path suitable for matrices assembled by ZynSim or
external finite-element packages.  It does not silently fall back when a user
explicitly requests PETSc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class DistributedLinearSolveOptions:
    method: Literal["cg", "gmres", "bcgs", "minres"] = "cg"
    preconditioner: Literal["gamg", "hypre", "jacobi", "ilu", "none"] = "gamg"
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-12
    divergence_tolerance: float = 1.0e8
    maximum_iterations: int = 20_000
    monitor: bool = False
    options_prefix: str = "zynsim_"
    return_global_solution: bool = True

    def __post_init__(self) -> None:
        if min(self.relative_tolerance, self.absolute_tolerance, self.divergence_tolerance) <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")


@dataclass(frozen=True, slots=True)
class DistributedSolveDiagnostics:
    converged: bool
    reason: int
    iterations: int
    residual_norm: float
    method: str
    preconditioner: str
    communicator_size: int


def solve_petsc(
    matrix: Any,
    rhs: np.ndarray,
    *,
    initial_guess: np.ndarray | None = None,
    options: DistributedLinearSolveOptions | None = None,
) -> tuple[np.ndarray, DistributedSolveDiagnostics]:
    """Solve a sparse linear system with PETSc KSP.

    A SciPy CSR matrix is accepted for a serial communicator.  In an MPI run,
    callers must pass an already-distributed ``PETSc.Mat`` so ownership and
    off-process entries remain explicit; silently replicating a global CSR on
    every rank is intentionally rejected.  ``rhs`` and ``initial_guess`` may
    contain either the global vector or the locally owned row segment.
    """

    try:
        from petsc4py import PETSc
    except ImportError as exc:  # pragma: no cover - optional HPC dependency
        raise ImportError(
            "PETSc solve requested but petsc4py is unavailable; install zynnova[zynsim-hpc]"
        ) from exc
    cfg = options or DistributedLinearSolveOptions()
    if isinstance(matrix, PETSc.Mat):
        operator = matrix
        communicator = operator.getComm()
    else:
        communicator = PETSc.COMM_WORLD
        if communicator.getSize() != 1:
            raise ValueError(
                "MPI PETSc solves require an explicitly distributed PETSc.Mat; "
                "a replicated SciPy CSR is accepted only in serial"
            )
        try:
            from scipy.sparse import csr_matrix
        except ImportError as exc:  # pragma: no cover
            raise ImportError("CSR conversion requires scipy") from exc
        csr = csr_matrix(matrix, dtype=float)
        operator = PETSc.Mat().createAIJ(
            size=csr.shape,
            csr=(
                csr.indptr.astype(PETSc.IntType),
                csr.indices.astype(PETSc.IntType),
                csr.data,
            ),
            comm=communicator,
        )
        operator.assemble()
    n_rows, n_cols = operator.getSize()
    if n_rows != n_cols:
        raise ValueError("matrix must be square")
    ownership_start, ownership_stop = operator.getOwnershipRange()
    local_size = int(ownership_stop - ownership_start)

    def populate(vector, values: np.ndarray, name: str) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape == (n_rows,):
            local = array[ownership_start:ownership_stop]
        elif array.shape == (local_size,):
            local = array
        else:
            raise ValueError(
                f"{name} must contain either {n_rows} global or {local_size} local values"
            )
        indices = np.arange(ownership_start, ownership_stop, dtype=PETSc.IntType)
        vector.setValues(indices, local)
        vector.assemblyBegin()
        vector.assemblyEnd()

    b = operator.createVecLeft()
    populate(b, rhs, "rhs")
    x = operator.createVecRight()
    if initial_guess is not None:
        populate(x, initial_guess, "initial_guess")
    else:
        x.set(0.0)
    ksp = PETSc.KSP().create(communicator)
    ksp.setOptionsPrefix(cfg.options_prefix)
    ksp.setOperators(operator)
    ksp.setType(cfg.method)
    pc = ksp.getPC()
    if cfg.preconditioner == "none":
        pc.setType(PETSc.PC.Type.NONE)
    elif cfg.preconditioner == "hypre":
        pc.setType(PETSc.PC.Type.HYPRE)
    else:
        pc.setType(cfg.preconditioner)
    ksp.setTolerances(
        rtol=cfg.relative_tolerance,
        atol=cfg.absolute_tolerance,
        divtol=cfg.divergence_tolerance,
        max_it=cfg.maximum_iterations,
    )
    ksp.setInitialGuessNonzero(initial_guess is not None)
    if cfg.monitor:
        ksp.setMonitor(
            lambda _, iteration, norm: print(
                f"[PETSc rank={communicator.getRank()}] it={iteration} residual={norm:.6e}"
            )
        )
    ksp.setFromOptions()
    ksp.solve(b, x)
    reason = int(ksp.getConvergedReason())
    diagnostics = DistributedSolveDiagnostics(
        converged=reason > 0,
        reason=reason,
        iterations=int(ksp.getIterationNumber()),
        residual_norm=float(ksp.getResidualNorm()),
        method=str(ksp.getType()),
        preconditioner=str(pc.getType()),
        communicator_size=int(communicator.getSize()),
    )
    if not diagnostics.converged:
        raise RuntimeError(
            f"PETSc KSP failed: reason={diagnostics.reason}, iterations={diagnostics.iterations}, "
            f"residual={diagnostics.residual_norm:.6e}"
        )
    if cfg.return_global_solution and communicator.getSize() > 1:
        scatter, gathered = PETSc.Scatter.toAll(x)
        scatter.scatter(x, gathered, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        result = gathered.getArray(readonly=True).copy()
        scatter.destroy()
        gathered.destroy()
    else:
        result = x.getArray(readonly=True).copy()
    return result, diagnostics


def morton_partition(
    centroids: np.ndarray,
    partition_count: int,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Partition cells by a locality-preserving 3-D Morton ordering.

    This inexpensive partitioner is deterministic and has no graph-library
    dependency.  It is useful for out-of-core blocks and as a preprocessing
    fallback before ParMETIS/PT-Scotch is available.
    """

    points = np.asarray(centroids, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("centroids must have shape (n,3)")
    count = int(partition_count)
    if count < 1:
        raise ValueError("partition_count must be positive")
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, np.finfo(float).eps)
    integer = np.clip(np.rint((points - lo) / span * ((1 << 21) - 1)), 0, (1 << 21) - 1).astype(np.uint64)
    codes = _morton3(integer[:, 0], integer[:, 1], integer[:, 2])
    order = np.argsort(codes, kind="stable")
    cell_weights = np.ones(len(points), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if cell_weights.shape != (len(points),) or np.any(cell_weights < 0.0):
        raise ValueError("weights must be non-negative with one value per cell")
    cumulative = np.cumsum(cell_weights[order])
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, count + 1)[1:-1]
    cut_positions = np.searchsorted(cumulative, targets, side="right")
    partition_sorted = np.empty(len(points), dtype=np.int32)
    start = 0
    for part, stop in enumerate((*cut_positions, len(points))):
        partition_sorted[start:stop] = part
        start = int(stop)
    result = np.empty_like(partition_sorted)
    result[order] = partition_sorted
    return result


def _morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    def spread(values: np.ndarray) -> np.ndarray:
        values = values & np.uint64(0x1FFFFF)
        values = (values | (values << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        values = (values | (values << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        values = (values | (values << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        values = (values | (values << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        values = (values | (values << np.uint64(2))) & np.uint64(0x1249249249249249)
        return values
    return spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))


__all__ = [
    "DistributedLinearSolveOptions",
    "DistributedSolveDiagnostics",
    "morton_partition",
    "solve_petsc",
]
