"""ZynNova public package.

ZynNova combines materials representation, datasets, machine learning, atomistic
simulation, battery multiphysics, conditional microstructure generation,
image-conditioned 3-D reconstruction, object-to-FEM workflows, and consent-aware
voice conversion behind one extensible Python namespace.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Mapping

try:
    __version__ = version("zynnova")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0.2.0"

from . import core, geometry, zynform, zynmorph, zynsim, zynvista, zynvox
from .structure import GraphData, StructureData


def backend_status() -> Mapping[str, object]:
    """Return side-effect-free diagnostics for every registered difficult-task backend."""

    return {
        "zynnova_version": __version__,
        "zynmorph": zynmorph.BACKENDS.status(),
        "zynvista_reconstruction": zynvista.RECONSTRUCTION_BACKENDS.status(),
        "zynvista_generation": zynvista.GENERATION_BACKENDS.status(),
        "zynvista_style": zynvista.STYLE_BACKENDS.status(),
        "zynform": zynform.OBJECT_BACKENDS.status(),
        "zynvox_conversion": zynvox.VOICE_BACKENDS.status(),
        "zynvox_tts": zynvox.TTS_BACKENDS.status(),
    }


__all__ = [
    "GraphData",
    "StructureData",
    "__version__",
    "backend_status",
    "core",
    "geometry",
    "zynform",
    "zynmorph",
    "zynsim",
    "zynvista",
    "zynvox",
]
