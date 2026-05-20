from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypeVar

from ranksmith.types import Document, RerankResult

__all__ = [
    "AsyncLLMProvider",
    "AsyncPairwiseLLMProvider",
    "AsyncProvider",
    "AsyncRerankStrategy",
    "LLMProvider",
    "PairwiseLLMProvider",
    "Provider",
    "RerankStrategy",
]

ProviderT_contra = TypeVar("ProviderT_contra", contravariant=True)
AsyncProviderT_contra = TypeVar("AsyncProviderT_contra", contravariant=True)


class LLMProvider(Protocol):
    def rank(self, query: str, documents: list[Document]) -> str:
        """Return a JSON string containing a 1-based ranking permutation."""


class PairwiseLLMProvider(Protocol):
    def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        """Return a JSON string containing a pairwise winner, "A" or "B"."""


class AsyncLLMProvider(Protocol):
    async def rank(self, query: str, documents: list[Document]) -> str:
        """Return a JSON string containing a 1-based ranking asynchronously."""


class AsyncPairwiseLLMProvider(Protocol):
    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        """Return a JSON string containing a pairwise winner asynchronously."""


Provider: TypeAlias = LLMProvider | PairwiseLLMProvider
AsyncProvider: TypeAlias = AsyncLLMProvider | AsyncPairwiseLLMProvider


class RerankStrategy(Protocol[ProviderT_contra]):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: ProviderT_contra,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...


class AsyncRerankStrategy(Protocol[AsyncProviderT_contra]):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: AsyncProviderT_contra,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...
