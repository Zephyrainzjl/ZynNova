"""Typed exceptions used by ZynNova without importing heavyweight backends."""

from __future__ import annotations


class ZynNovaError(RuntimeError):
    """Base class for recoverable ZynNova failures."""


class ConfigurationError(ZynNovaError, ValueError):
    """Raised when a workflow configuration is internally inconsistent."""


class BackendUnavailableError(ZynNovaError):
    """Raised when an optional external backend cannot be used."""


class BackendExecutionError(ZynNovaError):
    """Raised when an external backend exits unsuccessfully or returns bad output."""


class LicenseNotAcceptedError(ZynNovaError):
    """Raised before invoking a backend whose terms were not explicitly accepted."""


class GeometryError(ZynNovaError):
    """Raised when a geometry cannot satisfy an export or FEM invariant."""


class ConsentRequiredError(ZynNovaError):
    """Raised when voice-reference consent has not been recorded."""


__all__ = [
    "BackendExecutionError",
    "BackendUnavailableError",
    "ConfigurationError",
    "ConsentRequiredError",
    "GeometryError",
    "LicenseNotAcceptedError",
    "ZynNovaError",
]
