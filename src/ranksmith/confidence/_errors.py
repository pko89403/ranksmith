from __future__ import annotations


class ConfidenceError(Exception):
    """Base error for confidence estimation."""


class ConfidenceDependencyError(ConfidenceError):
    """Raised when an optional confidence dependency is unavailable."""


class ConfidenceInputError(ConfidenceError):
    """Raised when confidence input or estimator configuration is invalid."""


class ConfidenceArtifactError(ConfidenceError):
    """Raised when a confidence scorer artifact is invalid or incompatible."""
