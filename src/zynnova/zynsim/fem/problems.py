"""Linear steady, transient, thermal, and elastic finite-element problems."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.boundary import DirichletBC, SurfaceLoad
from ..core.linalg import LinearSolveOptions, apply_dirichlet, solve_linear
from ..core.mesh import Mesh
from ..core.results import SolverDiagnostics
from ._backend import BackendName
from .assembly import (
    ScalarOperators,
    assemble_body_force,
    assemble_eigenstrain_load,
    assemble_elasticity,
    assemble_scalar,
    assemble_scalar_source,
    assemble_surface_load,
    cell_gradients,
)
from .elements import isotropic_constitutive, strain_displacement, tet4_geometry


@dataclass(slots=True)
class ScalarSolution:
    values: np.ndarray
    gradient: np.ndarray
    flux: np.ndarray
    diagnostics: SolverDiagnostics
    backend: str


@dataclass(slots=True)
class ElasticitySolution:
    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    von_mises: np.ndarray
    reaction: np.ndarray
    diagnostics: SolverDiagnostics


@dataclass(slots=True)
class ScalarFEMProblem:
    mesh: Mesh
    conductivity: float | np.ndarray
    source: float | np.ndarray = 0.0
    dirichlet: tuple[DirichletBC, ...] = ()
    fluxes: tuple[SurfaceLoad, ...] = ()
    backend: BackendName = "auto"
    linear_options: LinearSolveOptions = field(default_factory=LinearSolveOptions)

    def operators(self) -> ScalarOperators:
        return assemble_scalar(
            self.mesh, self.conductivity, capacity=1.0, backend=self.backend
        )

    def right_hand_side(self) -> np.ndarray:
        result = assemble_scalar_source(self.mesh, self.source)
        for flux in self.fluxes:
            result += assemble_surface_load(self.mesh, flux, vector=False)
        return result

    def solve(self) -> ScalarSolution:
        operators = self.operators()
        matrix = operators.stiffness
        rhs = self.right_hand_side()
        if self.dirichlet:
            matrix, rhs = apply_dirichlet(matrix, rhs, self.dirichlet)
        values = solve_linear(matrix, rhs, self.linear_options)
        gradient = cell_gradients(self.mesh, values)
        tensors = _transport_tensors(self.conductivity, self.mesh.n_cells)
        flux = -np.einsum("eij,ej->ei", tensors, gradient)
        residual = float(np.linalg.norm(matrix @ values - rhs))
        return ScalarSolution(
            values=values,
            gradient=gradient,
            flux=flux,
            diagnostics=SolverDiagnostics(True, 1, residual, "linear solve"),
            backend=operators.backend,
        )


@dataclass(slots=True)
class TransientScalarFEM:
    mesh: Mesh
    conductivity: float | np.ndarray
    capacity: float | np.ndarray = 1.0
    dirichlet: tuple[DirichletBC, ...] = ()
    fluxes: tuple[SurfaceLoad, ...] = ()
    theta: float = 1.0
    lumped_mass: bool = False
    backend: BackendName = "auto"
    linear_options: LinearSolveOptions = field(default_factory=LinearSolveOptions)
    _operators: ScalarOperators | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0.5, 1.0]")

    @property
    def operators(self) -> ScalarOperators:
        if self._operators is None:
            self._operators = assemble_scalar(
                self.mesh,
                self.conductivity,
                self.capacity,
                lumped_mass=self.lumped_mass,
                backend=self.backend,
            )
        return self._operators

    def step(
        self,
        previous: np.ndarray,
        dt_s: float,
        *,
        source_previous: float | np.ndarray = 0.0,
        source_next: float | np.ndarray | None = None,
    ) -> np.ndarray:
        if dt_s <= 0.0:
            raise ValueError("time step must be positive")
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (self.mesh.n_nodes,):
            raise ValueError("previous state must have shape (n_nodes,)")
        if source_next is None:
            source_next = source_previous
        m_matrix = self.operators.mass
        k_matrix = self.operators.stiffness
        load_previous = assemble_scalar_source(self.mesh, source_previous)
        load_next = assemble_scalar_source(self.mesh, source_next)
        for flux in self.fluxes:
            surface = assemble_surface_load(self.mesh, flux, vector=False)
            load_previous += surface
            load_next += surface
        matrix = m_matrix + self.theta * dt_s * k_matrix
        rhs = (
            (m_matrix - (1.0 - self.theta) * dt_s * k_matrix) @ previous
            + dt_s
            * ((1.0 - self.theta) * load_previous + self.theta * load_next)
        )
        if self.dirichlet:
            matrix, rhs = apply_dirichlet(matrix, rhs, self.dirichlet)
        return solve_linear(matrix, rhs, self.linear_options)


@dataclass(slots=True)
class LinearElasticityProblem:
    mesh: Mesh
    young_modulus: float | np.ndarray
    poisson_ratio: float | np.ndarray
    dirichlet: tuple[DirichletBC, ...]
    body_force: float | np.ndarray = 0.0
    tractions: tuple[SurfaceLoad, ...] = ()
    eigenstrain: np.ndarray | None = None
    backend: BackendName = "auto"
    linear_options: LinearSolveOptions = field(default_factory=LinearSolveOptions)

    def solve(self) -> ElasticitySolution:
        stiffness = assemble_elasticity(
            self.mesh,
            self.young_modulus,
            self.poisson_ratio,
            backend=self.backend,
        )
        external = assemble_body_force(self.mesh, self.body_force)
        for traction in self.tractions:
            external += assemble_surface_load(self.mesh, traction, vector=True)
        if self.eigenstrain is not None:
            external += assemble_eigenstrain_load(
                self.mesh,
                self.young_modulus,
                self.poisson_ratio,
                self.eigenstrain,
            )
        constrained_matrix, constrained_rhs = apply_dirichlet(
            stiffness, external, self.dirichlet
        )
        flat_displacement = solve_linear(
            constrained_matrix, constrained_rhs, self.linear_options
        )
        strain, stress = _strain_stress(
            self.mesh,
            flat_displacement,
            self.young_modulus,
            self.poisson_ratio,
            self.eigenstrain,
        )
        reaction = np.asarray(stiffness @ flat_displacement - external).reshape(
            self.mesh.n_nodes, 3
        )
        residual = float(
            np.linalg.norm(constrained_matrix @ flat_displacement - constrained_rhs)
        )
        return ElasticitySolution(
            displacement=flat_displacement.reshape(self.mesh.n_nodes, 3),
            strain=strain,
            stress=stress,
            von_mises=_von_mises(stress),
            reaction=reaction,
            diagnostics=SolverDiagnostics(True, 1, residual, "linear solve"),
        )


ThermalFEM = TransientScalarFEM


def _transport_tensors(value: float | np.ndarray, count: int) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        return np.repeat((np.eye(3) * float(raw))[None, :, :], count, axis=0)
    if raw.shape == (3, 3):
        return np.repeat(raw[None, :, :], count, axis=0)
    if raw.shape == (count,):
        return raw[:, None, None] * np.eye(3)[None, :, :]
    if raw.shape == (count, 3, 3):
        return raw
    raise ValueError("invalid transport tensor shape")


def _strain_stress(
    mesh: Mesh,
    flat_displacement: np.ndarray,
    young_modulus: float | np.ndarray,
    poisson_ratio: float | np.ndarray,
    eigenstrain: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    young = np.broadcast_to(np.asarray(young_modulus, dtype=float), (mesh.n_cells,))
    poisson = np.broadcast_to(np.asarray(poisson_ratio, dtype=float), (mesh.n_cells,))
    if eigenstrain is None:
        inherent = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    else:
        inherent = np.broadcast_to(
            np.asarray(eigenstrain, dtype=np.float64), (mesh.n_cells, 6)
        )
    strain = np.empty((mesh.n_cells, 6), dtype=np.float64)
    stress = np.empty_like(strain)
    for element, cell in enumerate(mesh.cells):
        _, gradients = tet4_geometry(mesh.nodes[cell])
        b_matrix = strain_displacement(gradients)
        dofs = (3 * cell[:, None] + np.arange(3)[None, :]).reshape(-1)
        strain[element] = b_matrix @ flat_displacement[dofs]
        stress[element] = isotropic_constitutive(young[element], poisson[element]) @ (
            strain[element] - inherent[element]
        )
    return strain, stress


def _von_mises(stress: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, txz = np.asarray(stress).T
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


__all__ = [
    "ElasticitySolution",
    "LinearElasticityProblem",
    "ScalarFEMProblem",
    "ScalarSolution",
    "ThermalFEM",
    "TransientScalarFEM",
]
