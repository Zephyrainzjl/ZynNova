"""ZynForm: high-fidelity image-to-3D assets, DCC export, and FEM meshing."""

from .meshing import FEMMeshingError, tetrahedralize_surface
from .pipeline import run_object
from .registry import OBJECT_BACKENDS, PUBLIC_OBJECT_SOURCES
from .repair import repair_surface_for_fem
from .schema import FEMConfig, FEMMethod, ObjectConfig, ObjectRequest
from .types import ObjectBackendOutput, ObjectResult

__all__ = [
    "FEMConfig",
    "FEMMeshingError",
    "FEMMethod",
    "OBJECT_BACKENDS",
    "ObjectBackendOutput",
    "ObjectConfig",
    "ObjectRequest",
    "ObjectResult",
    "PUBLIC_OBJECT_SOURCES",
    "repair_surface_for_fem",
    "run_object",
    "tetrahedralize_surface",
]
