from __future__ import annotations


class RerankError(Exception):
    """Base error for ranksmith."""


class RerankParseError(RerankError):
    """Raised when the LLM response cannot be parsed or validated."""

    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(reason)


class RerankProviderError(RerankError):
    """Raised when the LLM provider request fails."""


class RerankInputError(RerankError):
    """Raised when user input or strategy configuration is invalid."""


class DocumentTooLongError(RerankError):
    """Raised when a document exceeds max_document_chars."""
