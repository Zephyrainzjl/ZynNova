from __future__ import annotations

from importlib import import_module
from typing import Literal

NativeBackend = Literal["auto", "python", "cpp"]


def native_module():
    try:
        return import_module("zynnova._native._dft_native")
    except ImportError:
        return None


def native_available() -> bool:
    return native_module() is not None


def resolve_native_backend(backend: NativeBackend) -> Literal["python", "cpp"]:
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be 'auto', 'python', or 'cpp'")
    if backend == "auto":
        return "cpp" if native_available() else "python"
    if backend == "cpp" and not native_available():
        raise RuntimeError(
            "The ZynNova DFT C++ extension is unavailable. Reinstall/build the package "
            "with ZYNNOVA_BUILD_NATIVE=ON, or use backend='python'."
        )
    return backend
