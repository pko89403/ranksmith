from __future__ import annotations

import asyncio
import json

import pytest

from ranksmith import (
    AsyncAzureOpenAIReranker,
    AsyncSetwiseStrategy,
    AzureOpenAIReranker,
    Document,
    DocumentTooLongError,
    RerankInputError,
    RerankParseError,
    SetwiseStrategy,
)


class SelectionByScoreProvider:
    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query
        self.calls.append([document.id or "" for document in documents])
        selected = sorted(
            range(len(documents)),
            key=lambda index: (-self.scores.get(documents[index].id or "", 0), index),
        )[:top_m]
        return json.dumps({"selected": [index + 1 for index in selected]})


class AsyncSelectionByScoreProvider:
    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    async def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query
        await asyncio.sleep(0)
        self.calls.append([document.id or "" for document in documents])
        selected = sorted(
            range(len(documents)),
            key=lambda index: (-self.scores.get(documents[index].id or "", 0), index),
        )[:top_m]
        return json.dumps({"selected": [index + 1 for index in selected]})


class InvalidSelectionProvider:
    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query, documents, top_m
        return '{"selected": [1, 1]}'


class MissingSelectionProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        return '{"ranking": [1, 2]}'


def _documents() -> list[Document]:
    return [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
        Document(id="d", text="delta"),
        Document(id="e", text="epsilon"),
        Document(id="f", text="zeta"),
        Document(id="g", text="eta"),
    ]


def _scores() -> dict[str, int]:
    return {"a": 1, "b": 7, "c": 3, "d": 6, "e": 5, "f": 2, "g": 4}


def test_setwise_strategy_defaults_to_setwise_heapsort() -> None:
    strategy = SetwiseStrategy()

    assert strategy.set_size == 3


def test_setwise_heapsort_reranks_top_k_and_preserves_indexes() -> None:
    provider = SelectionByScoreProvider(_scores())
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=SetwiseStrategy(set_size=3),
    )

    results = reranker.rerank("query", _documents(), top_k=3)

    assert [result.document.id for result in results] == ["b", "d", "e"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [1, 3, 4]
    assert [dict(result.metadata) for result in results] == [
        {"strategy": "setwise", "algorithm": "setwise_heapsort", "set_size": 3},
        {"strategy": "setwise", "algorithm": "setwise_heapsort", "set_size": 3},
        {"strategy": "setwise", "algorithm": "setwise_heapsort", "set_size": 3},
    ]


def test_setwise_heapsort_without_top_k_returns_full_order() -> None:
    provider = SelectionByScoreProvider(_scores())
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=SetwiseStrategy(set_size=4),
    )

    results = reranker.rerank("query", _documents())

    assert [result.document.id for result in results] == [
        "b",
        "d",
        "e",
        "g",
        "c",
        "f",
        "a",
    ]


def test_setwise_top_k_zero_returns_empty_without_provider_call() -> None:
    provider = SelectionByScoreProvider(_scores())
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=SetwiseStrategy(),
    )

    assert reranker.rerank("query", _documents(), top_k=0) == []
    assert provider.calls == []


def test_setwise_top_k_stops_after_requested_extractions() -> None:
    top_one_provider = SelectionByScoreProvider(_scores())
    full_provider = SelectionByScoreProvider(_scores())
    top_one_reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=top_one_provider,
        strategy=SetwiseStrategy(),
    )
    full_reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=full_provider,
        strategy=SetwiseStrategy(),
    )

    top_one_reranker.rerank("query", _documents(), top_k=1)
    full_reranker.rerank("query", _documents())

    assert len(top_one_provider.calls) < len(full_provider.calls)


def test_setwise_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="set_size"):
        SetwiseStrategy(set_size=2)


def test_setwise_rejects_provider_without_select() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=MissingSelectionProvider(),
        strategy=SetwiseStrategy(),
    )

    with pytest.raises(RerankInputError, match="selection select"):
        reranker.rerank("query", _documents())


def test_setwise_invalid_selection_fast_fails() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=InvalidSelectionProvider(),
        strategy=SetwiseStrategy(),
    )

    with pytest.raises(RerankParseError):
        reranker.rerank("query", _documents())


def test_setwise_long_document_fast_fails_before_provider_call() -> None:
    provider = SelectionByScoreProvider(_scores())
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=SetwiseStrategy(max_document_chars=3),
    )

    with pytest.raises(DocumentTooLongError):
        reranker.rerank("query", _documents())

    assert provider.calls == []


@pytest.mark.asyncio
async def test_async_setwise_heapsort_reranks_top_k() -> None:
    provider = AsyncSelectionByScoreProvider(_scores())
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=AsyncSetwiseStrategy(set_size=3),
    )

    results = await reranker.rerank("query", _documents(), top_k=3)

    assert [result.document.id for result in results] == ["b", "d", "e"]
    assert provider.calls


def test_async_setwise_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="set_size"):
        AsyncSetwiseStrategy(set_size=2)
