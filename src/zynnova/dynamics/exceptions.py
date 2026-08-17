"""Dynamics-specific exception hierarchy."""


class DynamicsError(RuntimeError):
    """Base exception raised by :mod:`zynnova.dynamics`."""


class ConfigurationError(DynamicsError, ValueError):
    """Raised when a simulation configuration is inconsistent."""


class MissingBackendError(DynamicsError, ImportError):
    """Raised when an optional simulation backend is unavailable."""


class PotentialError(DynamicsError):
    """Raised when a potential cannot provide a required property."""


class SimulationDivergedError(DynamicsError):
    """Raised when numerical safety checks detect a divergent trajectory."""


class RestartError(DynamicsError):
    """Raised when a checkpoint cannot be restored safely."""
