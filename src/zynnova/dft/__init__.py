"""Quantum mechanics, electronic structure, and ab-initio molecular dynamics.

The native C++ core provides stationary/time-dependent one-dimensional quantum
solvers plus NVE/NVT AIMD integration. Research electronic energies and forces
come from validated external engines through ASE-compatible calculators.
"""

from .api import *  # noqa: F403
from .api import __all__
from .exceptions import (
    AIMDDivergedError,
    AIMDError,
    AIMDRestartError,
    DFTConfigurationError,
    DFTError,
    MissingElectronicBackendError,
    QuantumSolverError,
    SCFConvergenceError,
)
from .results import (
    AIMDResult,
    AIMDThermoSeries,
    ElectronicStructureResult,
    StationaryStates,
    WavefunctionTrajectory,
)

__all__ = [
    *__all__,
    "AIMDDivergedError",
    "AIMDError",
    "AIMDResult",
    "AIMDRestartError",
    "AIMDThermoSeries",
    "DFTConfigurationError",
    "DFTError",
    "ElectronicStructureResult",
    "MissingElectronicBackendError",
    "QuantumSolverError",
    "SCFConvergenceError",
    "StationaryStates",
    "WavefunctionTrajectory",
]
