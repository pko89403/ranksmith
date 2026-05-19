import asyncio
from typing import Any, cast

import pytest

from ranksmith import (
    AsyncAzureOpenAIReranker,
    AsyncListwiseStrategy,
    AsyncPairwiseStrategy,
    Document,
    RerankInputError,
    RerankParseError,
    RerankProviderError,
)


class AsyncFakeProvider:
    def __init__(self, responses: list[str], fail: Exception | None = None) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[list[str]] = []

    async def rank(self, query: str, documents: list[Document]) -> str:
        if self.fail is not None:
            raise self.fail
        self.calls.append([document.text for document in documents])
        return self.responses.pop(0)


class AsyncFakePairwiseProvider:
    def __init__(self, responses: list[str], fail: Exception | None = None) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        if self.fail is not None:
            raise self.fail
        self.calls.append((document_a.text, document_b.text))
        return self.responses.pop(0)


class AsyncBlockingPairwiseProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        del query
        self.calls.append((document_a.text, document_b.text))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        winner = "B" if document_a.text == "a" and document_b.text == "b" else "A"
        return f'{{"winner": "{winner}"}}'


@pytest.mark.asyncio
async def test_async_reranks_string_documents_and_preserves_indexes() -> None:
    provider = AsyncFakeProvider(['{"ranking": [3, 1, 2]}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    results = await reranker.rerank("query", ["alpha", "beta", "gamma"])

    assert [result.document.text for result in results] == ["gamma", "alpha", "beta"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [2, 0, 1]


@pytest.mark.asyncio
async def test_async_reranks_document_objects() -> None:
    provider = AsyncFakeProvider(['{"ranking": [2, 1]}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )
    documents = [
        Document(id="a", text="first", metadata={"source": "one"}),
        Document(id="b", text="second", metadata={"source": "two"}),
    ]

    results = await reranker.rerank("query", documents, top_k=1)

    assert len(results) == 1
    assert results[0].document.id == "b"
    assert results[0].document.metadata == {"source": "two"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"ranking": [1, 1]}',
        '{"ranking": [1]}',
        '{"ranking": [1, 3]}',
        '{"ranking": ["1", "2"]}',
    ],
)
async def test_async_invalid_llm_ranking_fast_fails(response: str) -> None:
    provider = AsyncFakeProvider([response])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankParseError):
        await reranker.rerank("query", ["alpha", "beta"])


@pytest.mark.asyncio
async def test_async_provider_errors_are_wrapped() -> None:
    provider = AsyncFakeProvider([], fail=RuntimeError("timeout"))
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankProviderError) as error:
        await reranker.rerank("query", ["alpha"])

    assert "timeout" in str(error.value)


def test_async_listwise_strategy_defaults_to_rankgpt_sliding_window() -> None:
    strategy = AsyncListwiseStrategy()

    assert strategy.algorithm == "rankgpt_sliding_window"


def test_async_listwise_strategy_rejects_removed_sliding_window_algorithm() -> None:
    with pytest.raises(ValueError, match='algorithm must be "rankgpt_sliding_window"'):
        AsyncListwiseStrategy(algorithm=cast(Any, "sliding_window"))


def test_async_listwise_strategy_rejects_removed_direct_algorithm() -> None:
    with pytest.raises(ValueError, match='algorithm must be "rankgpt_sliding_window"'):
        AsyncListwiseStrategy(algorithm=cast(Any, "direct"))


@pytest.mark.asyncio
async def test_async_rankgpt_sliding_window_bubbles_top_document_up() -> None:
    provider = AsyncFakeProvider(
        [
            '{"ranking": [3, 1, 2]}',
            '{"ranking": [3, 1, 2]}',
        ]
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncListwiseStrategy(
            algorithm="rankgpt_sliding_window", window_size=3, stride=2
        ),
    )

    results = await reranker.rerank("query", ["a", "b", "c", "d", "e"])

    assert [result.document.text for result in results] == ["e", "a", "b", "c", "d"]
    assert provider.calls == [["c", "d", "e"], ["a", "b", "e"]]


@pytest.mark.asyncio
async def test_async_pairwise_prp_sliding_k_compares_and_swaps() -> None:
    provider = AsyncFakePairwiseProvider(
        [
            '{"winner": "B"}',
            '{"winner": "A"}',
            '{"winner": "B"}',
            '{"winner": "A"}',
        ]
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncPairwiseStrategy(passes=1),
    )

    results = await reranker.rerank("query", ["a", "b", "c"])

    assert [result.document.text for result in results] == ["c", "a", "b"]
    assert [result.original_index for result in results] == [2, 0, 1]
    assert provider.calls == [("b", "c"), ("c", "b"), ("a", "c"), ("c", "a")]


@pytest.mark.asyncio
async def test_async_pairwise_compares_pair_orders_concurrently() -> None:
    provider = AsyncBlockingPairwiseProvider()
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncPairwiseStrategy(passes=1),
    )

    results = await reranker.rerank("query", ["a", "b"])

    assert [result.document.text for result in results] == ["b", "a"]
    assert set(provider.calls) == {("a", "b"), ("b", "a")}
    assert provider.max_in_flight == 2


@pytest.mark.asyncio
async def test_async_pairwise_can_disable_pair_order_parallelism() -> None:
    provider = AsyncBlockingPairwiseProvider()
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncPairwiseStrategy(passes=1, pair_order_parallelism=1),
    )

    results = await reranker.rerank("query", ["a", "b"])

    assert [result.document.text for result in results] == ["b", "a"]
    assert provider.calls == [("a", "b"), ("b", "a")]
    assert provider.max_in_flight == 1


def test_async_pairwise_rejects_invalid_pair_order_parallelism() -> None:
    with pytest.raises(ValueError, match="pair_order_parallelism"):
        AsyncPairwiseStrategy(pair_order_parallelism=3)


@pytest.mark.asyncio
async def test_async_pairwise_invalid_winner_fast_fails() -> None:
    provider = AsyncFakePairwiseProvider(["not json", '{"winner": "A"}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncPairwiseStrategy(passes=1),
    )

    with pytest.raises(RerankParseError):
        await reranker.rerank("query", ["a", "b"])


@pytest.mark.asyncio
async def test_async_pairwise_strategy_rejects_provider_without_compare() -> None:
    provider = AsyncFakeProvider(['{"ranking": [1, 2]}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncPairwiseStrategy(passes=1),
    )

    with pytest.raises(RerankInputError):
        await reranker.rerank("query", ["a", "b"])

    assert provider.calls == []
