"""ZynSim phase-field engine for 1D, 2D, and 3D microstructure evolution.

The module combines a free-energy/kinetics registry with high-accuracy Fourier
ETDRK4 and adaptive IMEX-BDF2, high-order finite differences, optional C++/OpenMP,
Torch/JAX GPU and automatic differentiation, and live/recorded visualization.
"""

from ._backend import jax_available, native_available, torch_available
from .benchmarks import (
    PhaseFieldBenchmark,
    dendrite_growth_2d,
    shrinking_circle_2d,
    spinodal_decomposition_2d,
)
from .config import (
    AdaptiveTimeConfig,
    BoundaryCondition,
    GridSpec,
    SolverBackend,
    SolverConfig,
    TimeScheme,
)
from .diagnostics import (
    StructureFactorResult,
    energy_history,
    interfacial_measure,
    mass,
    phase_fraction,
    structure_factor,
)
from .fields import (
    FieldSpec,
    PhaseFieldDiagnostics,
    PhaseFieldResult,
    PhaseFieldState,
    PhaseFieldTrajectory,
)
from .initial_conditions import (
    from_labels,
    multiple_nuclei,
    planar_interface,
    random_noise,
    spherical_nucleus,
    voronoi_order_parameters,
)
from .io import load_state, load_trajectory, save_state, save_trajectory
from .models import *
from .models import __all__ as _model_exports
from .registry import (
    MODEL_REGISTRY,
    ModelDescriptor,
    PhaseFieldModelRegistry,
    available_models,
    create_model,
    register_model,
)
from .simulation import PhaseFieldSimulation, simulate
from .solvers import (
    FiniteDifferencePhaseFieldSolver,
    JAXSpectralPhaseFieldSolver,
    PhaseFieldCallback,
    PhaseFieldSolver,
    SpectralPhaseFieldSolver,
    TorchSpectralPhaseFieldSolver,
)
from .visualization import (
    LivePhaseFieldViewer,
    PhaseFieldAnimator,
    PyVistaPhaseFieldViewer,
    animate_phase_field,
)

__all__ = [
    "AdaptiveTimeConfig",
    "BoundaryCondition",
    "FieldSpec",
    "FiniteDifferencePhaseFieldSolver",
    "GridSpec",
    "JAXSpectralPhaseFieldSolver",
    "LivePhaseFieldViewer",
    "MODEL_REGISTRY",
    "ModelDescriptor",
    "PhaseFieldAnimator",
    "PhaseFieldBenchmark",
    "PhaseFieldCallback",
    "PhaseFieldDiagnostics",
    "PhaseFieldModelRegistry",
    "PhaseFieldResult",
    "PhaseFieldSimulation",
    "PhaseFieldSolver",
    "PhaseFieldState",
    "PhaseFieldTrajectory",
    "PyVistaPhaseFieldViewer",
    "SolverBackend",
    "SolverConfig",
    "SpectralPhaseFieldSolver",
    "StructureFactorResult",
    "TimeScheme",
    "TorchSpectralPhaseFieldSolver",
    "animate_phase_field",
    "available_models",
    "create_model",
    "dendrite_growth_2d",
    "energy_history",
    "from_labels",
    "interfacial_measure",
    "jax_available",
    "load_state",
    "load_trajectory",
    "mass",
    "multiple_nuclei",
    "native_available",
    "phase_fraction",
    "planar_interface",
    "random_noise",
    "register_model",
    "save_state",
    "save_trajectory",
    "shrinking_circle_2d",
    "simulate",
    "spherical_nucleus",
    "spinodal_decomposition_2d",
    "structure_factor",
    "torch_available",
    "voronoi_order_parameters",
    *_model_exports,
]
