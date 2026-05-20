from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from ranksmith.model import AsyncModelProvider, ModelProvider
from ranksmith.types import Document, RerankResult

__all__ = [
    "AsyncModelProvider",
    "AsyncRerankStrategy",
    "ModelProvider",
    "RerankStrategy",
]

ModelClientT_contra = TypeVar("ModelClientT_contra", contravariant=True)
AsyncModelClientT_contra = TypeVar("AsyncModelClientT_contra", contravariant=True)


class RerankStrategy(Protocol[ModelClientT_contra]):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClientT_contra,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...


class AsyncRerankStrategy(Protocol[AsyncModelClientT_contra]):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClientT_contra,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...
