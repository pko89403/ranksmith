from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, cast

import pytest

from ranksmith import (
    AsyncAzureOpenAIReranker,
    AsyncTourRankStrategy,
    AzureOpenAIReranker,
    Document,
    RerankInputError,
    RerankParseError,
    RerankProviderError,
    TourRankStageConfig,
    TourRankStrategy,
    parse_selection_response,
)

SMALL_STAGES = (TourRankStageConfig(group_count=2, group_size=3, selected_count=1),)


class SelectionByScoreProvider:
    def __init__(self, scores: dict[str, int], fail: Exception | None = None) -> None:
        self.scores = scores
        self.fail = fail
        self.calls: list[list[str]] = []

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query
        if self.fail is not None:
            raise self.fail
        self.calls.append([document.id or "" for document in documents])
        selected = sorted(
            range(len(documents)),
            key=lambda index: (-self.scores.get(documents[index].id or "", 0), index),
        )[:top_m]
        return json.dumps({"selected": [index + 1 for index in selected]})


class InvalidSelectionProvider:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query, documents, top_m
        self.calls += 1
        return '{"selected": [1, 1]}'


class MissingSelectionProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        return '{"ranking": [1, 2]}'


class BlockingSelectionProvider(SelectionByScoreProvider):
    def __init__(self, scores: dict[str, int]) -> None:
        super().__init__(scores)
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(0.01)
        try:
            return super().select(query, documents, top_m)
        finally:
            with self._lock:
                self.in_flight -= 1


class AsyncSelectionByScoreProvider:
    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query
        self.calls.append([document.id or "" for document in documents])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        selected = sorted(
            range(len(documents)),
            key=lambda index: (-self.scores.get(documents[index].id or "", 0), index),
        )[:top_m]
        return json.dumps({"selected": [index + 1 for index in selected]})


def _documents() -> list[Document]:
    return [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
        Document(id="d", text="delta"),
        Document(id="e", text="epsilon"),
        Document(id="f", text="zeta"),
    ]


def test_selection_parser_rejects_boolean_indexes() -> None:
    with pytest.raises(RerankParseError):
        parse_selection_response(
            '{"selected": [true]}',
            expected_count=2,
            selected_count=1,
        )


def test_tourrank_strategy_defaults_to_tourrank_2() -> None:
    strategy = TourRankStrategy()

    assert strategy.algorithm == "tourrank_r"
    assert strategy.rounds == 2
    assert strategy.group_parallelism == 1
    assert strategy.stage_configs == (
        TourRankStageConfig(group_count=5, group_size=20, selected_count=10),
        TourRankStageConfig(group_count=5, group_size=10, selected_count=4),
        TourRankStageConfig(group_count=1, group_size=20, selected_count=10),
        TourRankStageConfig(group_count=1, group_size=10, selected_count=5),
        TourRankStageConfig(group_count=1, group_size=5, selected_count=2),
    )


def test_tourrank_selects_by_accumulated_points_and_preserves_indexes() -> None:
    provider = SelectionByScoreProvider(
        {"a": 1, "b": 9, "c": 2, "d": 8, "e": 3, "f": 7}
    )
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(rounds=2, stage_configs=SMALL_STAGES),
    )

    results = reranker.rerank("query", _documents(), top_k=3)

    assert [result.document.id for result in results] == ["b", "e", "a"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [1, 4, 0]
    assert [dict(result.metadata) for result in results] == [
        {"strategy": "tourrank", "algorithm": "tourrank_r", "rounds": 2, "score": 2},
        {"strategy": "tourrank", "algorithm": "tourrank_r", "rounds": 2, "score": 2},
        {"strategy": "tourrank", "algorithm": "tourrank_r", "rounds": 2, "score": 0},
    ]
    assert len(provider.calls) == 4


def test_tourrank_default_group_parallelism_is_serial() -> None:
    provider = BlockingSelectionProvider(
        {"a": 1, "b": 9, "c": 2, "d": 8, "e": 3, "f": 7}
    )
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    reranker.rerank("query", _documents())

    assert provider.max_in_flight == 1


def test_tourrank_can_run_stage_groups_in_parallel() -> None:
    provider = BlockingSelectionProvider(
        {"a": 1, "b": 9, "c": 2, "d": 8, "e": 3, "f": 7}
    )
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(
            rounds=1,
            stage_configs=SMALL_STAGES,
            group_parallelism=2,
        ),
    )

    reranker.rerank("query", _documents())

    assert provider.max_in_flight == 2


def test_tourrank_rejects_invalid_group_parallelism() -> None:
    with pytest.raises(ValueError, match="group_parallelism"):
        TourRankStrategy(group_parallelism=0)
    with pytest.raises(ValueError, match="group_parallelism"):
        TourRankStrategy(group_parallelism=cast(Any, None))


def test_tourrank_fast_fails_when_stage_does_not_match_document_count() -> None:
    provider = SelectionByScoreProvider({"a": 1})
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    with pytest.raises(RerankInputError, match="stage_configs"):
        reranker.rerank("query", _documents()[:5])

    assert provider.calls == []


def test_tourrank_rejects_provider_without_select() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=MissingSelectionProvider(),
        strategy=TourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    with pytest.raises(RerankInputError, match="selection select"):
        reranker.rerank("query", _documents())


def test_tourrank_provider_error_is_preserved() -> None:
    provider = SelectionByScoreProvider({}, fail=RerankProviderError("timeout"))
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    with pytest.raises(RerankProviderError, match="timeout"):
        reranker.rerank("query", _documents())


def test_tourrank_invalid_selection_fast_fails() -> None:
    provider = InvalidSelectionProvider()
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=TourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    with pytest.raises(RerankParseError):
        reranker.rerank("query", _documents())


@pytest.mark.asyncio
async def test_async_tourrank_selects_groups_concurrently() -> None:
    provider = AsyncSelectionByScoreProvider(
        {"a": 1, "b": 9, "c": 2, "d": 8, "e": 3, "f": 7}
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=AsyncTourRankStrategy(rounds=1, stage_configs=SMALL_STAGES),
    )

    results = await reranker.rerank("query", _documents())

    assert [result.document.id for result in results] == ["b", "e", "a", "c", "d", "f"]
    assert [result.original_index for result in results] == [1, 4, 0, 2, 3, 5]
    assert provider.max_in_flight == 2


@pytest.mark.asyncio
async def test_async_tourrank_can_limit_group_parallelism() -> None:
    provider = AsyncSelectionByScoreProvider(
        {"a": 1, "b": 9, "c": 2, "d": 8, "e": 3, "f": 7}
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=provider,
        strategy=AsyncTourRankStrategy(
            rounds=1,
            stage_configs=SMALL_STAGES,
            group_parallelism=1,
        ),
    )

    await reranker.rerank("query", _documents())

    assert provider.max_in_flight == 1


def test_async_tourrank_rejects_invalid_group_parallelism() -> None:
    with pytest.raises(ValueError, match="group_parallelism"):
        AsyncTourRankStrategy(group_parallelism=0)
