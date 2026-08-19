"""Exception hierarchy for quantum and electronic-structure calculations."""


class DFTError(RuntimeError):
    """Base exception raised by :mod:`zynnova.dft`."""


class DFTConfigurationError(DFTError, ValueError):
    """Raised when a calculation configuration is inconsistent."""


class MissingElectronicBackendError(DFTError, ImportError):
    """Raised when an optional electronic-structure engine is unavailable."""


class QuantumSolverError(DFTError):
    """Raised when a numerical quantum solver fails."""


class SCFConvergenceError(DFTError):
    """Raised when a self-consistent electronic calculation does not converge."""


class AIMDError(DFTError):
    """Base exception for ab-initio molecular-dynamics failures."""


class AIMDDivergedError(AIMDError):
    """Raised when an AIMD safety condition is violated."""


class AIMDRestartError(AIMDError):
    """Raised when an AIMD checkpoint cannot be restored safely."""
