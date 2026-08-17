"""Built-in and public-repository backend registry for ZynMorph."""

from __future__ import annotations

from ..core.backend import BackendDescriptor, BackendRegistry
from .backends.base import MicrostructureBackend
from .backends.spectral import SpectralBackend
from .backends.torch_flow import TorchFlowBackend


BACKENDS: BackendRegistry[MicrostructureBackend] = BackendRegistry("battery-microstructure")
BACKENDS.register(
    BackendDescriptor(
        name="spectral-exact",
        task="battery-microstructure",
        factory=SpectralBackend,
        summary="Exact-composition anisotropic spectral baseline with descriptor refinement.",
        license_id="MIT",
        source="built-in",
        default_rank=50,
    )
)
BACKENDS.register(
    BackendDescriptor(
        name="torch-rectified-flow",
        task="battery-microstructure",
        factory=TorchFlowBackend,
        summary="Trainable conditional 3-D rectified-flow U-Net with exact-count projection.",
        license_id="MIT",
        source="built-in",
        default_rank=20,
        extras=("zynnova-morph-train",),
    )
)

PUBLIC_REFERENCE_BACKENDS = {
    "pores4thought": {
        "source": "https://github.com/agayonlombardo/pores4thought",
        "method": "periodic multiphase GAN baseline for porous microstructures",
        "license": (
            "no standard repository license detected during the 2026-08-17 "
            "audit; reference only unless permission is established"
        ),
    },
    "discrete-spatial-diffusion": {
        "source": "https://github.com/lanl/DiscreteSpatialDiffusion",
        "method": "intensity- and mass-preserving discrete spatial diffusion",
        "license": "LANL repository notice; isolate and review before deployment",
    },
    "microlad-paper": {
        "source": "https://arxiv.org/abs/2508.20138",
        "method": "latent multi-plane 2-D-to-3-D reconstruction and inverse control",
        "license": (
            "paper reference only; no author-maintained source repository was "
            "verified during the 2026-08-17 audit"
        ),
    },
}


__all__ = ["BACKENDS", "PUBLIC_REFERENCE_BACKENDS"]
