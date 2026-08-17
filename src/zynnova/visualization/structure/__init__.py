"""Interactive molecular, polymer, and crystal visualization for notebooks."""

from .adapters import coerce_structure, coerce_trajectory, infer_structure_kind
from .api import view, visualize, visualize_structure
from .backend import VisualizationDependencyError, available_backends
from .crystal import visualize_crystal
from .molecule import visualize_molecule
from .polymer import visualize_polymer
from .trajectory import visualize_trajectory
from .types import ViewerConfig

__all__ = [
    "ViewerConfig",
    "VisualizationDependencyError",
    "available_backends",
    "coerce_structure",
    "coerce_trajectory",
    "infer_structure_kind",
    "visualize",
    "view",
    "visualize_structure",
    "visualize_molecule",
    "visualize_polymer",
    "visualize_crystal",
    "visualize_trajectory",
]
