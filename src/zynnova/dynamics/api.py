from __future__ import annotations

from .analysis import summarize_thermo
from .calculators import (
    TorchPotentialCalculator,
    calculator_capabilities,
    create_classical_calculator,
)
from .config import (
    CellMode,
    Ensemble,
    MDConfig,
    OptimizerMethod,
    OutputConfig,
    RelaxationConfig,
    RunConfig,
    SafetyConfig,
    VelocityConfig,
    VelocityMode,
)
from .constraints import fix_atoms, fix_bonds
from .relaxation import relax, staged_relax
from .runner import DynamicsSession, run_md
from .trajectory import iter_trajectory, load_trajectory, write_trajectory
from .workflows import TemperatureStage, WorkflowResult, anneal, equilibrate


def available_ensembles() -> tuple[str, ...]:
    return tuple(item.value for item in Ensemble)


def available_optimizers() -> tuple[str, ...]:
    return tuple(item.value for item in OptimizerMethod)


__all__ = [
    "CellMode",
    "DynamicsSession",
    "Ensemble",
    "MDConfig",
    "OptimizerMethod",
    "OutputConfig",
    "RelaxationConfig",
    "RunConfig",
    "SafetyConfig",
    "TemperatureStage",
    "TorchPotentialCalculator",
    "VelocityConfig",
    "VelocityMode",
    "WorkflowResult",
    "anneal",
    "available_ensembles",
    "available_optimizers",
    "calculator_capabilities",
    "create_classical_calculator",
    "equilibrate",
    "fix_atoms",
    "fix_bonds",
    "iter_trajectory",
    "load_trajectory",
    "relax",
    "run_md",
    "staged_relax",
    "summarize_thermo",
    "write_trajectory",
]
