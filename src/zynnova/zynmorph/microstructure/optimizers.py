"""Optimizer plugin definitions for descriptor-based microstructure reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import OPTIMIZERS


@dataclass(frozen=True, slots=True)
class OptimizerDefinition:
    name: str
    family: str
    implementation: str
    preserves_phase_counts: bool = False


def _add(name: str, family: str, implementation: str, *, aliases=(), preserves=False):
    definition = OptimizerDefinition(name, family, implementation, preserves)
    OPTIMIZERS.register(name, definition, aliases=tuple(aliases))
    return definition


# Tensor/autograd optimizers.
_add("Adam", "torch", "Adam", aliases=("TFOptimizer",))
_add("Adamax", "torch", "Adamax")
_add("Nadam", "torch", "NAdam")
_add("RMSprop", "torch", "RMSprop")
_add("SGD", "torch", "SGD")
_add("Adagrad", "torch", "Adagrad")
_add("Adadelta", "torch", "Adadelta")

# SciPy gradient optimizers.
_add("LBFGSB", "scipy", "L-BFGS-B", aliases=("L-BFGS-B", "SPOptimizer"))
_add("TNC", "scipy", "TNC")

# Discrete Yeong-Torquato style optimizer and postprocessing route.
_add(
    "SimulatedAnnealing",
    "annealing",
    "phase-swap-metropolis",
    aliases=("YT", "YeongTorquato"),
    preserves=True,
)
_add("YTPost", "postprocess", "annealing-postprocess", preserves=True)


def optimizer_definition(name: str) -> OptimizerDefinition:
    return OPTIMIZERS.get(name)


def make_torch_optimizer(name: str, parameters: Any, settings: Any):
    import torch

    definition = optimizer_definition(name)
    if definition.family != "torch":
        raise ValueError(f"optimizer {definition.name} is not a torch optimizer")
    cls = getattr(torch.optim, definition.implementation)
    kwargs: dict[str, Any] = {"lr": settings.learning_rate}
    if definition.name in {"Adam", "Adamax", "Nadam"}:
        kwargs["betas"] = (settings.beta_1, settings.beta_2)
    elif definition.name == "RMSprop":
        kwargs["alpha"] = settings.rho
        kwargs["momentum"] = settings.momentum
    elif definition.name == "SGD":
        kwargs["momentum"] = settings.momentum
    elif definition.name == "Adadelta":
        kwargs["rho"] = settings.rho
    return cls(parameters, **kwargs)


__all__ = ["OptimizerDefinition", "make_torch_optimizer", "optimizer_definition"]
