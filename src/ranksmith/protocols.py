from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypeVar

from ranksmith.types import Document, RerankResult

__all__ = [
    "AsyncLLMProvider",
    "AsyncPairwiseLLMProvider",
    "AsyncProvider",
    "AsyncRerankStrategy",
    "AsyncSelectionLLMProvider",
    "LLMProvider",
    "PairwiseLLMProvider",
    "Provider",
    "RerankStrategy",
    "SelectionLLMProvider",
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


class SelectionLLMProvider(Protocol):
    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        """Return a JSON string containing 1-based selected document indexes."""


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


class AsyncSelectionLLMProvider(Protocol):
    async def select(
        self,
        query: str,
        documents: list[Document],
        top_m: int,
    ) -> str:
        """Return a JSON string containing selected indexes asynchronously."""


Provider: TypeAlias = LLMProvider | PairwiseLLMProvider | SelectionLLMProvider
AsyncProvider: TypeAlias = (
    AsyncLLMProvider | AsyncPairwiseLLMProvider | AsyncSelectionLLMProvider
)


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
