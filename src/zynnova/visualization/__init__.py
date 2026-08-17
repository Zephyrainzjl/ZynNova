"""ZynNova visualization namespace.

The result-plotting layer and structure viewers are imported lazily. This keeps
``from zynnova.visualization import results`` usable in lightweight analysis
environments that do not have the structure I/O backends installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_STRUCTURE_EXPORTS = {
    "ViewerConfig",
    "available_backends",
    "view",
    "visualize",
    "visualize_crystal",
    "visualize_molecule",
    "visualize_polymer",
    "visualize_structure",
    "visualize_trajectory",
}


def __getattr__(name: str) -> Any:
    if name == "results":
        value = import_module(f"{__name__}.results")
    elif name == "structure":
        value = import_module(f"{__name__}.structure")
    elif name in _STRUCTURE_EXPORTS:
        module = import_module(f"{__name__}.structure")
        value = getattr(module, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"results", "structure"} | _STRUCTURE_EXPORTS)


__all__ = ["results", "structure", *_STRUCTURE_EXPORTS]
