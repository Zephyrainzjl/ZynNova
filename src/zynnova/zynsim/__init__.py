"""ZynSim: finite-element, battery, multiscale, and safe LLM orchestration.

The package keeps ML checkpoint loading and fine-tuning outside the simulation
kernel. Callers inject calculators, property providers, or remote LLM clients.
"""

from . import (
    battery,
    core,
    digital_twin,
    fem,
    inverse,
    io,
    llm,
    microstructure,
    multiphysics,
    multiscale,
    operators,
    phasefield,
    studies,
    workflows,
)
from .constants import FARADAY, GAS_CONSTANT
from .exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    ConvergenceError,
    LLMProtocolError,
    MeshError,
    PropertyResolutionError,
    ZynSimError,
)

__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "ConvergenceError",
    "FARADAY",
    "GAS_CONSTANT",
    "LLMProtocolError",
    "MeshError",
    "PropertyResolutionError",
    "ZynSimError",
    "battery",
    "core",
    "digital_twin",
    "fem",
    "inverse",
    "io",
    "llm",
    "microstructure",
    "multiphysics",
    "multiscale",
    "operators",
    "phasefield",
    "studies",
    "workflows",
]
