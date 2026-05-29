from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ranksmith._providers import (
    AsyncAzureAOAIProvider,
    AzureAOAIProvider,
)
from ranksmith.errors import RerankError, RerankProviderError, RerankStrategyError
from ranksmith.model import (
    AsyncModelClient,
    AsyncModelProvider,
    AsyncUsageCallback,
    ModelClient,
    ModelProvider,
    UsageCallback,
)
from ranksmith.protocols import (
    AsyncRerankStrategy,
    RerankStrategy,
)
from ranksmith.strategies import (
    AcuRankStrategy,
    AsyncAcuRankStrategy,
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
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str = "2024-08-01-preview",
        strategy: RerankStrategy[Any] | None = None,
        provider: ModelProvider | None = None,
        model_client: ModelClient | None = None,
        timeout: float | None = None,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._strategy: RerankStrategy[Any] = strategy or ListwiseStrategy()
        if provider is not None and model_client is not None:
            raise ValueError("provider and model_client cannot both be provided")
        if model_client is not None:
            self._model_client = model_client
        else:
            if api_key is None or azure_endpoint is None or azure_deployment is None:
                raise ValueError(
                    "api_key, azure_endpoint, and azure_deployment are required "
                    "when model_client is not provided"
                )
            model_provider = provider or AzureAOAIProvider(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                timeout=timeout,
            )
            self._model_client = ModelClient(
                provider=model_provider,
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
                model_client=self._model_client,
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
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str = "2024-08-01-preview",
        strategy: AsyncRerankStrategy[Any] | None = None,
        provider: AsyncModelProvider | None = None,
        model_client: AsyncModelClient | None = None,
        timeout: float | None = None,
        on_usage: AsyncUsageCallback | None = None,
    ) -> None:
        self._strategy: AsyncRerankStrategy[Any] = strategy or AsyncListwiseStrategy()
        if provider is not None and model_client is not None:
            raise ValueError("provider and model_client cannot both be provided")
        if model_client is not None:
            self._model_client = model_client
        else:
            if api_key is None or azure_endpoint is None or azure_deployment is None:
                raise ValueError(
                    "api_key, azure_endpoint, and azure_deployment are required "
                    "when model_client is not provided"
                )
            model_provider = provider or AsyncAzureAOAIProvider(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                timeout=timeout,
            )
            self._model_client = AsyncModelClient(
                provider=model_provider,
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
                model_client=self._model_client,
                top_k=top_k,
            )
        except RerankError:
            raise
        except Exception as exc:
            if _is_builtin_async_strategy(self._strategy):
                raise RerankProviderError(str(exc)) from exc
            raise RerankStrategyError(str(exc)) from exc


def _is_builtin_sync_strategy(strategy: object) -> bool:
    return type(strategy) in {
        AcuRankStrategy,
        ListwiseStrategy,
        PairwiseStrategy,
        TourRankStrategy,
    }


def _is_builtin_async_strategy(strategy: object) -> bool:
    return type(strategy) in {
        AsyncAcuRankStrategy,
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
