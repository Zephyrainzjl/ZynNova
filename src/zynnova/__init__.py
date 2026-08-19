"""ZynNova public package.

ZynNova combines materials intelligence, simulation, generative 3D/scene workflows,
consent-aware speech, and provider-neutral LLM agents behind one extensible namespace.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path
from typing import Mapping

__path__ = extend_path(__path__, __name__)

try:
    __version__ = version("zynnova")
except PackageNotFoundError:
    __version__ = "0.3.0"

from . import core, geometry, llm, zynform, zynmorph, zynsim, zynvista, zynvox
from .structure import GraphData, StructureData

def backend_status() -> Mapping[str, object]:
    """Return side-effect-free diagnostics for registered difficult-task backends."""
    return {
        "zynnova_version": __version__,
        "zynmorph": zynmorph.BACKENDS.status(),
        "zynvista_reconstruction": zynvista.RECONSTRUCTION_BACKENDS.status(),
        "zynvista_generation": zynvista.GENERATION_BACKENDS.status(),
        "zynvista_style": zynvista.STYLE_BACKENDS.status(),
        "zynform": zynform.OBJECT_BACKENDS.status(),
        "zynvox_conversion": zynvox.VOICE_BACKENDS.status(),
        "zynvox_tts": zynvox.TTS_BACKENDS.status(),
        "llm_frameworks": ("zynastra",),
    }

__all__ = [
    "GraphData", "StructureData", "__version__", "backend_status", "core",
    "geometry", "llm", "zynform", "zynmorph", "zynsim", "zynvista", "zynvox",
]
