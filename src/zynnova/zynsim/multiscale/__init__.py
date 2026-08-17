"""Cross-scale material-property resolution and homogenization."""

from .adaptive import (
    AtomisticSurfaceProvider,
    CrossScaleSimulationLoop,
    CrossScaleSimulationResult,
    CrossScaleStepRecord,
    OnDemandAtomisticProvider,
    ParameterEstimateLike,
    ParameterSurfaceLike,
)
from .bindings import full_p2d_bindings, multiphysics_material_bindings
from .coupling import (
    CrossScaleRecord,
    MultiscaleCoordinator,
    PropertyBinding,
    default_p2d_bindings,
)
from .homogenization import (
    VoigtReussHillResult,
    bruggeman_effective,
    homogenize_conductivity_dirichlet,
    maxwell_garnett,
    voigt_reuss_hill,
)
from .jouleweave import JouleWeavePropertyProvider
from .properties import (
    CachingPropertyProvider,
    CompositePropertyProvider,
    MaterialProperty,
    PropertyProvider,
    PropertyRequest,
    TabulatedPropertyProvider,
    convert_property,
)

__all__ = [
    "AtomisticSurfaceProvider",
    "CrossScaleSimulationLoop",
    "CrossScaleSimulationResult",
    "CrossScaleStepRecord",
    "OnDemandAtomisticProvider",
    "ParameterEstimateLike",
    "ParameterSurfaceLike",
    "CachingPropertyProvider",
    "CompositePropertyProvider",
    "CrossScaleRecord",
    "JouleWeavePropertyProvider",
    "MaterialProperty",
    "MultiscaleCoordinator",
    "PropertyBinding",
    "PropertyProvider",
    "PropertyRequest",
    "TabulatedPropertyProvider",
    "VoigtReussHillResult",
    "bruggeman_effective",
    "convert_property",
    "default_p2d_bindings",
    "full_p2d_bindings",
    "multiphysics_material_bindings",
    "homogenize_conductivity_dirichlet",
    "maxwell_garnett",
    "voigt_reuss_hill",
]
