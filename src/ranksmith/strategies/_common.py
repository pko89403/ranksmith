from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from ranksmith.errors import DocumentTooLongError, RerankInputError
from ranksmith.model import AsyncModelClient, ModelClient
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


def ensure_listwise_model_client(model_client: object) -> ModelClient:
    rank = getattr(model_client, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(ModelClient, model_client)


def ensure_async_listwise_model_client(model_client: object) -> AsyncModelClient:
    rank = getattr(model_client, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(AsyncModelClient, model_client)


def ensure_pairwise_model_client(model_client: object) -> ModelClient:
    compare = getattr(model_client, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(ModelClient, model_client)


def ensure_async_pairwise_model_client(model_client: object) -> AsyncModelClient:
    compare = getattr(model_client, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(AsyncModelClient, model_client)


def ensure_selection_model_client(model_client: object) -> ModelClient:
    select = getattr(model_client, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(ModelClient, model_client)


def ensure_async_selection_model_client(model_client: object) -> AsyncModelClient:
    select = getattr(model_client, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(AsyncModelClient, model_client)
