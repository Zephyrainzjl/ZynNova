"""Neural operators and adaptive reduced-order surrogates."""

from .neural_operator import (
    NeuralOperatorConfig,
    ParameterEmbeddedFNO1d,
    SpectralConv1d,
    SurrogateGate,
    physics_informed_operator_loss,
)
from .rom import AdaptiveROM

__all__ = [
    "AdaptiveROM",
    "NeuralOperatorConfig",
    "ParameterEmbeddedFNO1d",
    "SpectralConv1d",
    "SurrogateGate",
    "physics_informed_operator_loss",
]
