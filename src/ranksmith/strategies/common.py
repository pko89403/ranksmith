from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ranksmith.errors import DocumentTooLongError, RerankInputError
from ranksmith.types import Document


def validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k < 0:
        raise RerankInputError("top_k must be greater than or equal to 0")


def validate_documents_max_chars(
    documents: Sequence[Document],
    *,
    max_document_chars: int,
) -> None:
    for index, document in enumerate(documents):
        length = len(document.text)
        if length > max_document_chars:
            message = (
                f"Document at index {index} has {length} characters, exceeding "
                f"max_document_chars={max_document_chars}. Shorten the "
                "document, chunk it before reranking, or increase "
                "max_document_chars."
            )
            raise DocumentTooLongError(message)


def ensure_capability(model_client: object, capability: str, method: str) -> Any:
    if not callable(getattr(model_client, method, None)):
        raise RerankInputError(f"provider must support {capability} {method}()")
    return model_client
