"""Differentiable multimodal inversion and Bayesian uncertainty propagation."""

from .bayesian import DiagonalLaplace, DiagonalLaplacePosterior, propagate_posterior
from .calibration import (
    JointCalibrationConfig,
    JointCalibrationResult,
    JointInverseProblem,
)
from .differentiable import (
    DifferentiableBatteryParameters,
    DifferentiableBatterySolver,
    DifferentiableSolverConfig,
)
from .implicit import (
    DifferentiableImplicitConfig,
    DifferentiableImplicitSolver,
    ImplicitSolveDiagnostics,
    ResidualFunction,
)
from .mechanisms import MechanismSelectionConfig, MechanismSelector
from .observations import (
    MultimodalObservationSet,
    ObservationModality,
    ObservationSeries,
)

__all__ = [
    "DifferentiableImplicitConfig",
    "DifferentiableImplicitSolver",
    "ImplicitSolveDiagnostics",
    "ResidualFunction",
    "DiagonalLaplace",
    "DiagonalLaplacePosterior",
    "DifferentiableBatteryParameters",
    "DifferentiableBatterySolver",
    "DifferentiableSolverConfig",
    "JointCalibrationConfig",
    "JointCalibrationResult",
    "JointInverseProblem",
    "MechanismSelectionConfig",
    "MechanismSelector",
    "MultimodalObservationSet",
    "ObservationModality",
    "ObservationSeries",
    "propagate_posterior",
]
