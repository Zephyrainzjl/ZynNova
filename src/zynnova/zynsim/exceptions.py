"""Exceptions raised by zynsim."""

from __future__ import annotations


class ZynSimError(RuntimeError):
    """Base class for simulation failures."""


class ConfigurationError(ZynSimError, ValueError):
    """A model or solver configuration is internally inconsistent."""


class MeshError(ZynSimError, ValueError):
    """A mesh is malformed or unsupported."""


class ConvergenceError(ZynSimError):
    """An iterative numerical method did not meet its convergence criterion."""


class BackendUnavailableError(ZynSimError, ImportError):
    """A requested optional numerical backend is unavailable."""


class PropertyResolutionError(ZynSimError):
    """A cross-scale material property could not be resolved safely."""


class LLMProtocolError(ZynSimError):
    """An LLM provider returned an invalid or unsafe orchestration message."""


__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "ConvergenceError",
    "LLMProtocolError",
    "MeshError",
    "PropertyResolutionError",
    "ZynSimError",
]
