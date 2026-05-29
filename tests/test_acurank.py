from __future__ import annotations

import asyncio
import json

import pytest

from ranksmith import (
    AcuRankStrategy,
    AsyncAcuRankStrategy,
    AzureOpenAIReranker,
    Document,
    RerankInputError,
    RerankParseError,
)


class RankByIdProvider:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls: list[list[str]] = []

    def rank(self, query: str, documents: list[Document]) -> str:
        del query
        ids = [document.id or "" for document in documents]
        self.calls.append(ids)
        ranking = sorted(
            range(len(documents)),
            key=lambda index: self.order.index(ids[index]),
        )
        return json.dumps({"ranking": [index + 1 for index in ranking]})


class AsyncRankByIdProvider:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls: list[list[str]] = []

    async def rank(self, query: str, documents: list[Document]) -> str:
        del query
        ids = [document.id or "" for document in documents]
        self.calls.append(ids)
        ranking = sorted(
            range(len(documents)),
            key=lambda index: self.order.index(ids[index]),
        )
        return json.dumps({"ranking": [index + 1 for index in ranking]})


class AsyncBlockingRankProvider:
    def __init__(self) -> None:
        self.started = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._release = asyncio.Event()

    async def rank(self, query: str, documents: list[Document]) -> str:
        del query
        self.started += 1
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        if self.started == 2:
            self._release.set()
        await self._release.wait()
        self._in_flight -= 1
        return json.dumps({"ranking": list(range(1, len(documents) + 1))})


class InvalidRankProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        return '{"ranking": [1, 1]}'


class FailingRankProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        raise TimeoutError("provider timeout")


class MissingRankProvider:
    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query, documents, top_m
        return '{"selected": [1]}'


def test_acurank_strategy_defaults_to_acurank() -> None:
    strategy = AcuRankStrategy()

    assert strategy.algorithm == "acurank"
    assert strategy.target_rank == 10
    assert strategy.window_size == 20
    assert strategy.tolerance == 0.01
    assert strategy.uncertain_threshold == 10
    assert strategy.max_adaptive_reranker_calls is None
    assert strategy.batch_parallelism == 1
    assert strategy.initial_pass is True
    assert strategy.score_metadata_key == "score"
    assert strategy.max_document_chars == 4000


def test_async_acurank_strategy_defaults_to_acurank() -> None:
    strategy = AsyncAcuRankStrategy()

    assert strategy.algorithm == "acurank"
    assert strategy.target_rank == 10
    assert strategy.window_size == 20
    assert strategy.tolerance == 0.01
    assert strategy.uncertain_threshold == 10
    assert strategy.max_adaptive_reranker_calls is None
    assert strategy.batch_parallelism == 1
    assert strategy.initial_pass is True
    assert strategy.score_metadata_key == "score"
    assert strategy.max_document_chars == 4000


def test_acurank_uses_listwise_evidence_to_update_final_mu_order() -> None:
    provider = RankByIdProvider(["b", "a", "c"])
    reranker = AzureOpenAIReranker(
        model_client=provider,
        strategy=AcuRankStrategy(
            target_rank=2,
            window_size=3,
            uncertain_threshold=1,
            max_adaptive_reranker_calls=0,
        ),
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
    ]

    results = reranker.rerank("query", documents)

    assert [result.document.id for result in results] == ["b", "a", "c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [1, 0, 2]
    assert len(provider.calls) == 1
    assert provider.calls[0] == ["a", "b", "c"]
    assert all(result.metadata["strategy"] == "acurank" for result in results)
    assert all(result.metadata["algorithm"] == "acurank" for result in results)
    assert results[0].metadata["mu"] > results[1].metadata["mu"]
    assert results[1].metadata["mu"] > results[2].metadata["mu"]
    assert all(result.metadata["reranker_calls"] == 1 for result in results)


def test_acurank_default_target_rank_clips_to_small_document_count() -> None:
    provider = RankByIdProvider(["b", "a"])
    reranker = AzureOpenAIReranker(
        model_client=provider,
        strategy=AcuRankStrategy(window_size=2, max_adaptive_reranker_calls=0),
    )

    results = reranker.rerank(
        "query",
        [Document(id="a", text="alpha"), Document(id="b", text="beta")],
    )

    assert [result.document.id for result in results] == ["b", "a"]
    assert len(provider.calls) == 1


def test_acurank_uses_score_metadata_for_prior() -> None:
    provider = RankByIdProvider(["b", "a", "c"])
    reranker = AzureOpenAIReranker(
        model_client=provider,
        strategy=AcuRankStrategy(
            target_rank=2,
            window_size=3,
            uncertain_threshold=10,
            max_adaptive_reranker_calls=0,
            initial_pass=False,
        ),
    )
    documents = [
        Document(id="a", text="alpha", metadata={"score": 3.0}),
        Document(id="b", text="beta", metadata={"score": 9.0}),
        Document(id="c", text="gamma", metadata={"score": 1.0}),
    ]

    results = reranker.rerank("query", documents)

    assert [result.document.id for result in results] == ["b", "a", "c"]
    assert provider.calls == []
    assert results[0].metadata["mu"] == pytest.approx(9.0)
    assert results[0].metadata["sigma"] == pytest.approx(3.0)


def test_acurank_rejects_partial_or_invalid_score_metadata() -> None:
    partial = AzureOpenAIReranker(
        model_client=RankByIdProvider(["a", "b"]),
        strategy=AcuRankStrategy(target_rank=1, initial_pass=False),
    )
    with pytest.raises(RerankInputError, match="score metadata"):
        partial.rerank(
            "query",
            [
                Document(id="a", text="alpha", metadata={"score": 1.0}),
                Document(id="b", text="beta"),
            ],
        )

    invalid = AzureOpenAIReranker(
        model_client=RankByIdProvider(["a", "b"]),
        strategy=AcuRankStrategy(target_rank=1, initial_pass=False),
    )
    with pytest.raises(RerankInputError, match="numeric"):
        invalid.rerank(
            "query",
            [
                Document(id="a", text="alpha", metadata={"score": "high"}),
                Document(id="b", text="beta", metadata={"score": 1.0}),
            ],
        )

    boolean_score = AzureOpenAIReranker(
        model_client=RankByIdProvider(["a", "b"]),
        strategy=AcuRankStrategy(target_rank=1, initial_pass=False),
    )
    with pytest.raises(RerankInputError, match="numeric"):
        boolean_score.rerank(
            "query",
            [
                Document(id="a", text="alpha", metadata={"score": True}),
                Document(id="b", text="beta", metadata={"score": 1.0}),
            ],
        )


def test_acurank_fast_fails_invalid_provider_contract_and_ranking() -> None:
    missing = AzureOpenAIReranker(
        model_client=MissingRankProvider(),
        strategy=AcuRankStrategy(target_rank=1, window_size=2, uncertain_threshold=1),
    )
    with pytest.raises(RerankInputError, match="listwise rank"):
        missing.rerank("query", [Document(text="alpha"), Document(text="beta")])

    invalid = AzureOpenAIReranker(
        model_client=InvalidRankProvider(),
        strategy=AcuRankStrategy(target_rank=1, window_size=2, uncertain_threshold=1),
    )
    with pytest.raises(RerankParseError):
        invalid.rerank("query", [Document(text="alpha"), Document(text="beta")])


@pytest.mark.asyncio
async def test_async_acurank_uses_listwise_evidence() -> None:
    from ranksmith import AsyncAzureOpenAIReranker  # noqa: PLC0415

    provider = AsyncRankByIdProvider(["b", "a", "c"])
    reranker = AsyncAzureOpenAIReranker(
        model_client=provider,
        strategy=AsyncAcuRankStrategy(
            target_rank=2,
            window_size=3,
            uncertain_threshold=1,
            max_adaptive_reranker_calls=0,
        ),
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
    ]

    results = await reranker.rerank("query", documents)

    assert [result.document.id for result in results] == ["b", "a", "c"]
    assert len(provider.calls) == 1


def test_acurank_builtin_provider_errors_are_preserved() -> None:
    from ranksmith import RerankProviderError  # noqa: PLC0415

    reranker = AzureOpenAIReranker(
        model_client=FailingRankProvider(),
        strategy=AcuRankStrategy(target_rank=1, window_size=2),
    )

    with pytest.raises(RerankProviderError, match="provider timeout"):
        reranker.rerank("query", [Document(text="alpha"), Document(text="beta")])


def test_acurank_adaptive_budget_does_not_limit_initial_pass() -> None:
    provider = RankByIdProvider(["b", "a", "d", "c", "e"])
    reranker = AzureOpenAIReranker(
        model_client=provider,
        strategy=AcuRankStrategy(
            target_rank=2,
            window_size=2,
            uncertain_threshold=1,
            max_adaptive_reranker_calls=0,
        ),
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
        Document(id="d", text="delta"),
        Document(id="e", text="epsilon"),
    ]

    reranker.rerank("query", documents)

    assert provider.calls == [["a", "b"], ["c", "d"]]


def test_acurank_runs_final_refinement_below_uncertain_threshold() -> None:
    provider = RankByIdProvider(["c", "b", "a"])
    reranker = AzureOpenAIReranker(
        model_client=provider,
        strategy=AcuRankStrategy(
            target_rank=2,
            window_size=3,
            uncertain_threshold=10,
            max_adaptive_reranker_calls=1,
            initial_pass=False,
        ),
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
    ]

    results = reranker.rerank("query", documents)

    assert provider.calls == [["a", "b", "c"]]
    assert [result.document.id for result in results] == ["c", "b", "a"]


@pytest.mark.asyncio
async def test_async_acurank_parallelizes_independent_initial_batches() -> None:
    from ranksmith import AsyncAzureOpenAIReranker  # noqa: PLC0415

    provider = AsyncBlockingRankProvider()
    reranker = AsyncAzureOpenAIReranker(
        model_client=provider,
        strategy=AsyncAcuRankStrategy(
            target_rank=2,
            window_size=2,
            max_adaptive_reranker_calls=0,
            batch_parallelism=2,
        ),
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
        Document(id="d", text="delta"),
    ]

    await reranker.rerank("query", documents)

    assert provider.started == 2
    assert provider.max_in_flight == 2


def test_acurank_rejects_invalid_batch_parallelism() -> None:
    with pytest.raises(ValueError, match="batch_parallelism"):
        AcuRankStrategy(batch_parallelism=0)

    with pytest.raises(ValueError, match="batch_parallelism"):
        AsyncAcuRankStrategy(batch_parallelism=0)
