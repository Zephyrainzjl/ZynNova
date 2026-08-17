"""Three-dimensional porous-electrode finite-element battery model."""

from .model import (
    Battery3DConfig,
    Battery3DDiagnostics,
    Battery3DState,
    PorousElectrode3D,
    layered_battery_mesh,
)

__all__ = [
    "Battery3DConfig",
    "Battery3DDiagnostics",
    "Battery3DState",
    "PorousElectrode3D",
    "layered_battery_mesh",
]
