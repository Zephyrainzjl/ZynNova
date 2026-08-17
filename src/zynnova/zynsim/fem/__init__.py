"""Validated Tet4 finite-element kernels and coupled continuum solvers."""

from ._backend import BackendName, native_available
from .assembly import (
    ScalarOperators,
    assemble_eigenstrain_load,
    assemble_elasticity,
    assemble_scalar,
    assemble_scalar_source,
)
from .distributed import (
    DistributedLinearSolveOptions,
    DistributedSolveDiagnostics,
    morton_partition,
    solve_petsc,
)
from .elements import (
    NeoHookeanElementResult,
    compressible_neo_hookean,
    elastic_stiffness,
    scalar_matrices,
    tet4_geometry,
)
from .nonlinear import NeoHookeanProblem, NonlinearElasticitySolution
from .problems import (
    ElasticitySolution,
    LinearElasticityProblem,
    ScalarFEMProblem,
    ScalarSolution,
    ThermalFEM,
    TransientScalarFEM,
)

__all__ = [
    "BackendName",
    "DistributedSolveDiagnostics",
    "DistributedLinearSolveOptions",
    "ElasticitySolution",
    "LinearElasticityProblem",
    "NeoHookeanElementResult",
    "NeoHookeanProblem",
    "NonlinearElasticitySolution",
    "ScalarFEMProblem",
    "ScalarOperators",
    "ScalarSolution",
    "ThermalFEM",
    "TransientScalarFEM",
    "assemble_eigenstrain_load",
    "assemble_elasticity",
    "assemble_scalar",
    "assemble_scalar_source",
    "compressible_neo_hookean",
    "elastic_stiffness",
    "native_available",
    "solve_petsc",
    "morton_partition",
    "scalar_matrices",
    "tet4_geometry",
]
