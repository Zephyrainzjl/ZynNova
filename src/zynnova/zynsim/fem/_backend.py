"""Selection of the native C++ FEM kernels with a NumPy reference fallback."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

BackendName = Literal["auto", "python", "cpp"]


def native_module() -> Any | None:
    try:
        return import_module("zynnova._native._zynsim_fem_native")
    except ImportError:
        return None


def native_available() -> bool:
    return native_module() is not None


def resolve_backend(backend: BackendName) -> Literal["python", "cpp"]:
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be 'auto', 'python', or 'cpp'")
    if backend == "auto":
        return "cpp" if native_available() else "python"
    if backend == "cpp" and not native_available():
        raise RuntimeError(
            "The ZynSim C++ FEM backend is unavailable. Rebuild ZynNova with "
            "ZYNNOVA_BUILD_NATIVE=ON, or select backend='python'."
        )
    return backend


__all__ = ["BackendName", "native_available", "native_module", "resolve_backend"]
