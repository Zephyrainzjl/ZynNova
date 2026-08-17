"""Finite-strain compressible Neo-Hookean Newton solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.boundary import DirichletBC, SurfaceLoad
from ..core.linalg import LinearSolveOptions, solve_linear
from ..core.mesh import Mesh
from ..core.results import SolverDiagnostics
from ..exceptions import ConvergenceError
from ._backend import BackendName, native_module, resolve_backend
from .assembly import assemble_body_force, assemble_surface_load
from .elements import compressible_neo_hookean


def _sparse() -> Any:
    try:
        import scipy.sparse as sparse
    except ImportError as exc:
        raise ImportError("nonlinear FEM requires SciPy; install zynnova[simulation]") from exc
    return sparse


@dataclass(slots=True)
class NonlinearElasticitySolution:
    displacement: np.ndarray
    diagnostics: SolverDiagnostics
    strain_energy_J: float
    minimum_jacobian: float


@dataclass(slots=True)
class NeoHookeanProblem:
    mesh: Mesh
    shear_modulus: float | np.ndarray
    lame_lambda: float | np.ndarray
    dirichlet: tuple[DirichletBC, ...]
    body_force: float | np.ndarray = 0.0
    tractions: tuple[SurfaceLoad, ...] = ()
    load_steps: int = 4
    tolerance: float = 1.0e-8
    max_iterations: int = 30
    line_search_steps: int = 12
    backend: BackendName = "auto"
    linear_options: LinearSolveOptions = field(
        default_factory=lambda: LinearSolveOptions(method="direct")
    )

    def __post_init__(self) -> None:
        if self.load_steps < 1 or self.max_iterations < 1 or self.line_search_steps < 1:
            raise ValueError("nonlinear iteration counts must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("nonlinear tolerance must be positive")

    def solve(self, initial: np.ndarray | None = None) -> NonlinearElasticitySolution:
        size = 3 * self.mesh.n_nodes
        displacement = (
            np.zeros(size, dtype=np.float64)
            if initial is None
            else np.asarray(initial, dtype=np.float64).reshape(size).copy()
        )
        external = assemble_body_force(self.mesh, self.body_force)
        for traction in self.tractions:
            external += assemble_surface_load(self.mesh, traction, vector=True)
        constrained_dofs, final_values = _merge_constraints(self.dirichlet, size)
        free = np.setdiff1d(np.arange(size), constrained_dofs, assume_unique=True)
        history: list[float] = []
        total_iterations = 0
        minimum_jacobian = 1.0
        strain_energy = 0.0

        for load_step in range(1, self.load_steps + 1):
            load_factor = load_step / self.load_steps
            displacement[constrained_dofs] = load_factor * final_values
            target_external = load_factor * external
            for _ in range(self.max_iterations):
                total_iterations += 1
                internal, tangent, strain_energy, minimum_jacobian = self._assemble(
                    displacement
                )
                residual = internal - target_external
                norm = float(np.linalg.norm(residual[free]))
                scale = max(float(np.linalg.norm(target_external[free])), 1.0)
                relative = norm / scale
                history.append(relative)
                if relative <= self.tolerance:
                    break
                increment = solve_linear(
                    tangent[free][:, free],
                    -residual[free],
                    self.linear_options,
                )
                accepted = False
                baseline = norm
                alpha = 1.0
                for _line_search in range(self.line_search_steps):
                    trial = displacement.copy()
                    trial[free] += alpha * increment
                    try:
                        trial_internal, _, _, _ = self._assemble(trial)
                    except ValueError:
                        alpha *= 0.5
                        continue
                    trial_norm = float(
                        np.linalg.norm((trial_internal - target_external)[free])
                    )
                    if trial_norm < (1.0 - 1.0e-4 * alpha) * baseline:
                        displacement = trial
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    raise ConvergenceError(
                        f"Neo-Hookean line search failed at load step {load_step}"
                    )
            else:
                raise ConvergenceError(
                    f"Neo-Hookean Newton solve did not converge at load step {load_step}; "
                    f"last relative residual={history[-1]:.3e}"
                )

        return NonlinearElasticitySolution(
            displacement=displacement.reshape(self.mesh.n_nodes, 3),
            diagnostics=SolverDiagnostics(
                True,
                total_iterations,
                history[-1] if history else 0.0,
                "load-stepped Newton-Raphson with backtracking",
                tuple(history),
            ),
            strain_energy_J=float(strain_energy),
            minimum_jacobian=float(minimum_jacobian),
        )

    def _assemble(self, displacement: np.ndarray) -> tuple[np.ndarray, Any, float, float]:
        sparse = _sparse()
        mu = np.broadcast_to(
            np.asarray(self.shear_modulus, dtype=np.float64), (self.mesh.n_cells,)
        )
        lame = np.broadcast_to(
            np.asarray(self.lame_lambda, dtype=np.float64), (self.mesh.n_cells,)
        )
        selected = resolve_backend(self.backend)
        internal = np.zeros(3 * self.mesh.n_nodes, dtype=np.float64)
        rows = np.empty(self.mesh.n_cells * 144, dtype=np.int64)
        columns = np.empty_like(rows)
        values = np.empty(self.mesh.n_cells * 144, dtype=np.float64)
        total_energy = 0.0
        minimum_jacobian = float("inf")
        native = native_module() if selected == "cpp" else None
        cursor = 0
        for element, cell in enumerate(self.mesh.cells):
            dofs = (3 * cell[:, None] + np.arange(3)[None, :]).reshape(-1)
            local_u = displacement[dofs].reshape(4, 3)
            if native is None:
                local = compressible_neo_hookean(
                    self.mesh.nodes[cell], local_u, mu[element], lame[element]
                )
                residual = local.residual
                tangent = local.tangent
                energy = local.energy
                jacobian = local.jacobian
            else:
                payload = native.tet4_neo_hookean(
                    self.mesh.nodes[cell], local_u, mu[element], lame[element]
                )
                residual = np.asarray(payload["residual"])
                tangent = np.asarray(payload["tangent"])
                energy = float(payload["energy"])
                jacobian = float(payload["jacobian"])
            internal[dofs] += residual.reshape(-1)
            total_energy += energy
            minimum_jacobian = min(minimum_jacobian, jacobian)
            row_grid, column_grid = np.meshgrid(dofs, dofs, indexing="ij")
            block = slice(cursor, cursor + 144)
            rows[block] = row_grid.reshape(-1)
            columns[block] = column_grid.reshape(-1)
            values[block] = tangent.reshape(-1)
            cursor += 144
        size = 3 * self.mesh.n_nodes
        tangent_matrix = sparse.coo_matrix(
            (values, (rows, columns)), shape=(size, size)
        ).tocsr()
        return internal, tangent_matrix, total_energy, minimum_jacobian


def _merge_constraints(
    conditions: tuple[DirichletBC, ...], size: int
) -> tuple[np.ndarray, np.ndarray]:
    if not conditions:
        raise ValueError("nonlinear elasticity requires displacement constraints")
    mapping: dict[int, float] = {}
    for condition in conditions:
        for dof, value in zip(condition.dofs, condition.values, strict=True):
            dof_int = int(dof)
            if dof_int >= size:
                raise ValueError("Dirichlet dof is out of range")
            if dof_int in mapping and not np.isclose(mapping[dof_int], value):
                raise ValueError("conflicting Dirichlet values")
            mapping[dof_int] = float(value)
    dofs = np.asarray(sorted(mapping), dtype=np.int64)
    values = np.asarray([mapping[int(dof)] for dof in dofs], dtype=np.float64)
    return dofs, values


__all__ = ["NeoHookeanProblem", "NonlinearElasticitySolution"]
