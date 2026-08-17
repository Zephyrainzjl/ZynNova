"""Optional native, Torch, and JAX backend discovery."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any


def native_module() -> Any | None:
    try:
        return import_module("zynnova._native._zynsim_phasefield_native")
    except ImportError:
        return None


def native_available() -> bool:
    return native_module() is not None


def torch_available() -> bool:
    return find_spec("torch") is not None


def jax_available() -> bool:
    return find_spec("jax") is not None and find_spec("jaxlib") is not None


__all__ = ["jax_available", "native_available", "native_module", "torch_available"]
