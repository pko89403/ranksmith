from __future__ import annotations

from collections.abc import Sequence

from ranksmith._providers import AzureOpenAIProvider, LLMProvider
from ranksmith.errors import RerankError, RerankProviderError
from ranksmith.strategies import ListwiseStrategy
from ranksmith.types import Document, RerankResult


class AzureOpenAIReranker:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str = "2024-08-01-preview",
        strategy: ListwiseStrategy | None = None,
        provider: LLMProvider | None = None,
        timeout: float | None = None,
    ) -> None:
        self._strategy = strategy or ListwiseStrategy()
        self._provider = provider or AzureOpenAIProvider(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            timeout=timeout,
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
            raise RerankProviderError(str(exc)) from exc


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
