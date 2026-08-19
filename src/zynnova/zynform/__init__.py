"""ZynForm: high-fidelity image-to-3D assets, DCC export, and FEM meshing."""

from .meshing import FEMMeshingError, tetrahedralize_surface
from .pipeline import run_object
from .quality import SurfaceAudit, audit_surface
from .registry import OBJECT_BACKENDS, PUBLIC_OBJECT_SOURCES
from .repair import repair_surface_for_fem
from .scaling import (
    PhysicalScaleTransform,
    apply_physical_scale,
    compute_physical_scale_transform,
    transform_native_asset,
)
from .schema import FEMConfig, FEMMethod, ObjectConfig, ObjectRequest, PhysicalScaleBasis
from .types import ObjectBackendOutput, ObjectResult

from .external import CommandObjectEngine, GenerativeObjectRequest, ObjectAssetBundle, ObjectEngineProfile
from .model_hub import download_object_model, object_workspace
from .studio import ObjectStudio

__all__ = [
    "object_workspace",
    "download_object_model",
    "ObjectStudio",
    "ObjectEngineProfile",
    "ObjectAssetBundle",
    "GenerativeObjectRequest",
    "CommandObjectEngine",
    "FEMConfig",
    "FEMMeshingError",
    "FEMMethod",
    "OBJECT_BACKENDS",
    "PhysicalScaleBasis",
    "PhysicalScaleTransform",
    "ObjectBackendOutput",
    "ObjectConfig",
    "ObjectRequest",
    "ObjectResult",
    "PUBLIC_OBJECT_SOURCES",
    "SurfaceAudit",
    "apply_physical_scale",
    "audit_surface",
    "compute_physical_scale_transform",
    "repair_surface_for_fem",
    "run_object",
    "tetrahedralize_surface",
    "transform_native_asset",
]
