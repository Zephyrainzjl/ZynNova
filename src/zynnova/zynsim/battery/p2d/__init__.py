"""Doyle–Fuller–Newman pseudo-two-dimensional battery model."""

from .model import MaterialUpdate, P2DModel
from .parameters import (
    ElectrodeParameters,
    ElectrolyteParameters,
    P2DDiscretization,
    P2DParameters,
    SeparatorParameters,
    ThermalParameters,
    graphite_ocp,
    nmc811_ocp,
    reference_graphite_nmc811_parameters,
)
from .protocol import CurrentSegment, P2DTrajectory
from .state import P2DState, P2DStepDiagnostics

__all__ = [
    "CurrentSegment",
    "ElectrodeParameters",
    "ElectrolyteParameters",
    "MaterialUpdate",
    "P2DDiscretization",
    "P2DModel",
    "P2DParameters",
    "P2DState",
    "P2DStepDiagnostics",
    "P2DTrajectory",
    "SeparatorParameters",
    "ThermalParameters",
    "graphite_ocp",
    "nmc811_ocp",
    "reference_graphite_nmc811_parameters",
]
