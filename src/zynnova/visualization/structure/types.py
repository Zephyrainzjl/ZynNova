from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BackendName = Literal["auto", "py3dmol", "nglview"]
StructureKind = Literal["auto", "molecule", "polymer", "crystal"]
RenderStyle = Literal["ball_and_stick", "stick", "sphere", "line", "surface"]


@dataclass(slots=True)
class ViewerConfig:
    """Common rendering configuration shared by all structure visualizers."""

    width: int = 800
    height: int = 500
    background: str = "white"
    style: RenderStyle = "ball_and_stick"
    backend: BackendName = "auto"
    camera: Literal["perspective", "orthographic"] = "perspective"
    spin: bool = False
    atom_labels: bool = False
    atom_indices: bool = False
    show_hydrogens: bool = True
    show_bonds: bool = True
    show_unit_cell: bool = True
    show_axes: bool = False
    supercell: tuple[int, int, int] = (1, 1, 1)
    sphere_scale: float = 0.28
    stick_radius: float = 0.16
    line_width: float = 1.5
    surface_opacity: float = 0.45
    zoom: float = 1.0
    color_by: Literal["element", "unit", "role", "chain"] = "element"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("viewer width and height must be positive")
        if len(self.supercell) != 3 or any(int(value) < 1 for value in self.supercell):
            raise ValueError("supercell must contain three positive integers")
        if self.sphere_scale <= 0 or self.stick_radius <= 0:
            raise ValueError("sphere_scale and stick_radius must be positive")
        if not 0 <= self.surface_opacity <= 1:
            raise ValueError("surface_opacity must lie in [0, 1]")
        if self.zoom <= 0:
            raise ValueError("zoom must be positive")
