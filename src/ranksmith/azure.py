from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ranksmith._providers import (
    AsyncAzureOpenAIProvider,
    AsyncUsageCallback,
    AzureOpenAIProvider,
    UsageCallback,
)
from ranksmith.errors import RerankError, RerankProviderError, RerankStrategyError
from ranksmith.protocols import (
    AsyncLLMProvider,
    AsyncPairwiseLLMProvider,
    AsyncRerankStrategy,
    AsyncSelectionLLMProvider,
    LLMProvider,
    PairwiseLLMProvider,
    RerankStrategy,
    SelectionLLMProvider,
)
from ranksmith.strategies import (
    AsyncListwiseStrategy,
    AsyncPairwiseStrategy,
    AsyncTourRankStrategy,
    ListwiseStrategy,
    PairwiseStrategy,
    TourRankStrategy,
)
from ranksmith.types import Document, RerankResult


class AzureOpenAIReranker:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str = "2024-08-01-preview",
        strategy: RerankStrategy[Any] | None = None,
        provider: (
            LLMProvider | PairwiseLLMProvider | SelectionLLMProvider | None
        ) = None,
        timeout: float | None = None,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._strategy: RerankStrategy[Any] = strategy or ListwiseStrategy()
        self._provider = provider or AzureOpenAIProvider(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            timeout=timeout,
            on_usage=on_usage,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str | Document],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        normalized_documents = _normalize_documents(documents)
        try:
            return self._strategy.rerank(
                query=query,
                documents=normalized_documents,
                provider=self._provider,
                top_k=top_k,
            )
        except RerankError:
            raise
        except Exception as exc:
            if _is_builtin_sync_strategy(self._strategy):
                raise RerankProviderError(str(exc)) from exc
            raise RerankStrategyError(str(exc)) from exc


class AsyncAzureOpenAIReranker:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str = "2024-08-01-preview",
        strategy: AsyncRerankStrategy[Any] | None = None,
        provider: (
            AsyncLLMProvider
            | AsyncPairwiseLLMProvider
            | AsyncSelectionLLMProvider
            | None
        ) = None,
        timeout: float | None = None,
        on_usage: AsyncUsageCallback | None = None,
    ) -> None:
        self._strategy: AsyncRerankStrategy[Any] = strategy or AsyncListwiseStrategy()
        self._provider = provider or AsyncAzureOpenAIProvider(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            timeout=timeout,
            on_usage=on_usage,
        )

    async def rerank(
        self,
        query: str,
        documents: Sequence[str | Document],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        normalized_documents = _normalize_documents(documents)
        try:
            return await self._strategy.rerank(
                query=query,
                documents=normalized_documents,
                provider=self._provider,
                top_k=top_k,
            )
        except RerankError:
            raise
        except Exception as exc:
            if _is_builtin_async_strategy(self._strategy):
                raise RerankProviderError(str(exc)) from exc
            raise RerankStrategyError(str(exc)) from exc


def _is_builtin_sync_strategy(strategy: object) -> bool:
    return type(strategy) in {ListwiseStrategy, PairwiseStrategy, TourRankStrategy}


def _is_builtin_async_strategy(strategy: object) -> bool:
    return type(strategy) in {
        AsyncListwiseStrategy,
        AsyncPairwiseStrategy,
        AsyncTourRankStrategy,
    }


def _normalize_documents(documents: Sequence[str | Document]) -> list[Document]:
    normalized: list[Document] = []
    for document in documents:
        if isinstance(document, str):
            normalized.append(Document(text=document))
        elif isinstance(document, Document):
            normalized.append(document)
        else:
            typename = type(document).__name__
            raise TypeError(f"documents must contain str or Document, got {typename}")
    return normalized
