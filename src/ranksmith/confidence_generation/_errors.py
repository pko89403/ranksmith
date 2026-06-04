from __future__ import annotations


class ConfidenceGenerationError(Exception):
    """Base error for confidence generation."""


class ConfidenceGenerationInputError(ConfidenceGenerationError):
    """Raised when confidence generation input or config is invalid."""


class ConfidenceGenerationParseError(ConfidenceGenerationError):
    """Raised when closed model output cannot be parsed."""
