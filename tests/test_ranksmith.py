from __future__ import annotations

from typing import Any, cast

import pytest

from ranksmith import (
    AzureOpenAIReranker,
    Document,
    DocumentTooLongError,
    ListwiseStrategy,
    PairwiseStrategy,
    RerankInputError,
    RerankParseError,
    RerankProviderError,
)


class FakeProvider:
    def __init__(self, responses: list[str], fail: Exception | None = None) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[list[str]] = []

    def rank(self, query: str, documents: list[Document]) -> str:
        if self.fail is not None:
            raise self.fail
        self.calls.append([document.text for document in documents])
        return self.responses.pop(0)


class FakePairwiseProvider:
    def __init__(self, responses: list[str], fail: Exception | None = None) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def compare(self, query: str, document_a: Document, document_b: Document) -> str:
        if self.fail is not None:
            raise self.fail
        self.calls.append((document_a.text, document_b.text))
        return self.responses.pop(0)


def test_reranks_string_documents_and_preserves_original_indexes() -> None:
    provider = FakeProvider(['{"ranking": [3, 1, 2]}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    results = reranker.rerank("query", ["alpha", "beta", "gamma"])

    assert [result.document.text for result in results] == ["gamma", "alpha", "beta"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [2, 0, 1]


def test_reranks_document_objects_and_preserves_metadata() -> None:
    provider = FakeProvider(['{"ranking": [2, 1]}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )
    documents = [
        Document(id="a", text="first", metadata={"source": "one"}),
        Document(id="b", text="second", metadata={"source": "two"}),
    ]

    results = reranker.rerank("query", documents, top_k=1)

    assert len(results) == 1
    assert results[0].document.id == "b"
    assert results[0].document.metadata == {"source": "two"}
    assert results[0].rank == 1
    assert results[0].original_index == 1


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
def test_invalid_llm_ranking_fast_fails(response: str) -> None:
    provider = FakeProvider([response])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankParseError):
        reranker.rerank("query", ["alpha", "beta"])


def test_long_document_fast_fails_without_truncating() -> None:
    provider = FakeProvider(['{"ranking": [1]}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=ListwiseStrategy(max_document_chars=3),
    )

    with pytest.raises(DocumentTooLongError) as error:
        reranker.rerank("query", ["four"])

    assert "index 0" in str(error.value)
    assert provider.calls == []


def test_listwise_strategy_defaults_to_rankgpt_sliding_window() -> None:
    strategy = ListwiseStrategy()

    assert strategy.algorithm == "rankgpt_sliding_window"


def test_listwise_strategy_rejects_removed_sliding_window_algorithm() -> None:
    with pytest.raises(ValueError, match='algorithm must be "rankgpt_sliding_window"'):
        ListwiseStrategy(algorithm=cast(Any, "sliding_window"))


def test_listwise_strategy_rejects_removed_direct_algorithm() -> None:
    with pytest.raises(ValueError, match='algorithm must be "rankgpt_sliding_window"'):
        ListwiseStrategy(algorithm=cast(Any, "direct"))


def test_negative_top_k_is_input_error_not_provider_error() -> None:
    provider = FakeProvider(['{"ranking": [1]}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankInputError):
        reranker.rerank("query", ["alpha"], top_k=-1)


def test_provider_errors_are_wrapped() -> None:
    provider = FakeProvider([], fail=RuntimeError("timeout"))
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankProviderError) as error:
        reranker.rerank("query", ["alpha"])

    assert "timeout" in str(error.value)


def test_rankgpt_sliding_window_bubbles_top_document_up() -> None:
    provider = FakeProvider(
        [
            '{"ranking": [3, 1, 2]}',
            '{"ranking": [3, 1, 2]}',
        ]
    )
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=ListwiseStrategy(
            algorithm="rankgpt_sliding_window", window_size=3, stride=2
        ),
    )

    results = reranker.rerank("query", ["a", "b", "c", "d", "e"])

    assert [result.document.text for result in results] == ["e", "a", "b", "c", "d"]
    assert provider.calls == [["c", "d", "e"], ["a", "b", "e"]]


def test_rankgpt_sliding_window_rejects_stride_larger_than_window_size() -> None:
    with pytest.raises(RerankInputError):
        ListwiseStrategy(algorithm="rankgpt_sliding_window", window_size=3, stride=4)


def test_pairwise_strategy_defaults_to_prp_sliding_10() -> None:
    strategy = PairwiseStrategy()

    assert strategy.algorithm == "prp_sliding_k"
    assert strategy.passes == 10


def test_pairwise_prp_sliding_k_compares_from_bottom_and_swaps() -> None:
    provider = FakePairwiseProvider(
        [
            '{"winner": "B"}',
            '{"winner": "A"}',
            '{"winner": "B"}',
            '{"winner": "A"}',
        ]
    )
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=PairwiseStrategy(passes=1),
    )

    results = reranker.rerank("query", ["a", "b", "c"])

    assert [result.document.text for result in results] == ["c", "a", "b"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [2, 0, 1]
    assert [dict(result.metadata) for result in results] == [
        {"strategy": "pairwise", "algorithm": "prp_sliding_k"},
        {"strategy": "pairwise", "algorithm": "prp_sliding_k"},
        {"strategy": "pairwise", "algorithm": "prp_sliding_k"},
    ]
    assert provider.calls == [("b", "c"), ("c", "b"), ("a", "c"), ("c", "a")]


def test_pairwise_prp_sliding_k_keeps_order_on_conflicting_preference() -> None:
    provider = FakePairwiseProvider(['{"winner": "A"}', '{"winner": "A"}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=PairwiseStrategy(passes=1),
    )

    results = reranker.rerank("query", ["a", "b"])

    assert [result.document.text for result in results] == ["a", "b"]
    assert provider.calls == [("a", "b"), ("b", "a")]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "{}",
        '{"winner": "C"}',
        '{"winner": "a"}',
        '{"winner": 1}',
    ],
)
def test_pairwise_invalid_winner_fast_fails(response: str) -> None:
    provider = FakePairwiseProvider([response])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=PairwiseStrategy(passes=1),
    )

    with pytest.raises(RerankParseError):
        reranker.rerank("query", ["a", "b"])


def test_pairwise_strategy_rejects_provider_without_compare() -> None:
    provider = FakeProvider(['{"ranking": [1, 2]}'])
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=PairwiseStrategy(passes=1),
    )

    with pytest.raises(RerankInputError):
        reranker.rerank("query", ["a", "b"])

    assert provider.calls == []
