"""Explicit failure types for the ZIVAR electro-spin production path."""

from __future__ import annotations

from typing import Any


class ZIVARError(Exception):
    """Base class for failures raised by the unified ZIVAR core."""


class ZIVARValidationError(ZIVARError, ValueError):
    """An input violates the public tensor or physical-semantics contract."""


class ShapeError(ZIVARValidationError):
    """A tensor has an incompatible rank or dimension."""


class SymmetryError(ZIVARValidationError):
    """A tensor or transformation violates an O(3)/time-reversal contract."""


class NonPositiveDefiniteError(ZIVARValidationError):
    """A claimed convex quadratic contribution is not positive definite."""


class ConstraintError(ZIVARError):
    """A linear constraint is malformed, inconsistent, or unsatisfied."""


class NonFiniteFunctionalError(ZIVARError, FloatingPointError):
    """The unified energy or one of its variational inputs is non-finite."""


class SCFConvergenceError(ZIVARError, RuntimeError):
    """The variational solve terminated without satisfying its convergence gate."""

    def __init__(self, message: str, *, report: Any | None = None) -> None:
        super().__init__(message)
        self.report = report


__all__ = [
    "ConstraintError",
    "NonFiniteFunctionalError",
    "NonPositiveDefiniteError",
    "SCFConvergenceError",
    "ShapeError",
    "SymmetryError",
    "ZIVARError",
    "ZIVARValidationError",
]
