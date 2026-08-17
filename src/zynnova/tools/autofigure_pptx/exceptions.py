from __future__ import annotations


class AutoFigurePPTXError(RuntimeError):
    """Base exception raised by the editable scientific-figure toolkit."""


class MissingDependencyError(AutoFigurePPTXError):
    """Raised when an optional backend required by the requested action is absent."""


class SceneValidationError(AutoFigurePPTXError):
    """Raised when a scene graph cannot be rendered safely."""


class UpstreamAutoFigureUnavailable(AutoFigurePPTXError):
    """Raised when strict upstream AutoFigure compatibility was requested but unavailable."""


class PPTXCompatibilityError(AutoFigurePPTXError):
    """Raised when a generated package violates the selected PowerPoint target."""
