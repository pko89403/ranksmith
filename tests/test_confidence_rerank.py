from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ranksmith import (
    AnswerConfidenceRerankStrategy,
    AsyncAnswerConfidenceRerankStrategy,
    Document,
)
from ranksmith.confidence.types import AnswerConfidenceInput
from ranksmith.errors import RerankInputError


@dataclass
class _Result:
    score: float


class FakeEstimator:
    """Answer-confidence estimator with scripted per-context scores."""

    task_type = "answer_confidence"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, item: AnswerConfidenceInput) -> _Result:
        return _Result(score=self._scores[item.context])


class FakeAnswerClient:
    """Model client that returns scripted answers per context."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers
        self.calls: list[str] = []

    def answer(self, query: str, context: str) -> str:
        self.calls.append(context)
        return json.dumps({"answer": self._answers[context]})


class AsyncFakeAnswerClient(FakeAnswerClient):
    async def answer(self, query: str, context: str) -> str:  # type: ignore[override]
        self.calls.append(context)
        return json.dumps({"answer": self._answers[context]})


DOCS = [Document(text="a"), Document(text="b"), Document(text="c"), Document(text="d")]
ANSWERS = {"a": "A1", "b": "B1", "c": "C1", "d": "D1"}


def _strategy() -> AnswerConfidenceRerankStrategy:
    estimator = FakeEstimator({"a": 0.9, "b": 0.3, "c": 0.7, "d": 0.1})
    return AnswerConfidenceRerankStrategy(estimator=estimator)


def test_orders_by_answer_confidence_descending() -> None:
    client = FakeAnswerClient(ANSWERS)
    results = _strategy().rerank(query="q", documents=DOCS, model_client=client)

    assert [r.document.text for r in results] == ["a", "c", "b", "d"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert results[0].metadata["answer_confidence"] == pytest.approx(0.9)
    assert results[3].metadata["answer_confidence"] == pytest.approx(0.1)
    # one answer call per document, no repeated sampling
    assert client.calls == ["a", "b", "c", "d"]


def test_ties_preserve_original_order() -> None:
    client = FakeAnswerClient(ANSWERS)
    estimator = FakeEstimator({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    results = AnswerConfidenceRerankStrategy(estimator=estimator).rerank(
        query="q", documents=DOCS, model_client=client
    )
    assert [r.document.text for r in results] == ["a", "b", "c", "d"]


def test_top_k_slices_after_ordering() -> None:
    client = FakeAnswerClient(ANSWERS)
    results = _strategy().rerank(
        query="q", documents=DOCS, model_client=client, top_k=2
    )
    assert [r.document.text for r in results] == ["a", "c"]


def test_empty_documents_returns_empty() -> None:
    client = FakeAnswerClient({})
    assert _strategy().rerank(query="q", documents=[], model_client=client) == []


def test_rejects_non_answer_estimator() -> None:
    class JudgmentEstimator:
        task_type = "judgment_confidence"

        def score(self, item: AnswerConfidenceInput) -> _Result:
            return _Result(score=0.5)

    with pytest.raises(RerankInputError, match="answer_confidence"):
        AnswerConfidenceRerankStrategy(estimator=JudgmentEstimator())


def test_rejects_client_without_answer() -> None:
    class NoAnswerClient:
        def rank(self, query: str, documents: list[Document]) -> str:
            return "{}"

    with pytest.raises(RerankInputError, match="answer"):
        _strategy().rerank(query="q", documents=DOCS, model_client=NoAnswerClient())


def test_invalid_answer_json_fast_fails() -> None:
    from ranksmith.errors import RerankParseError

    class BadClient(FakeAnswerClient):
        def answer(self, query: str, context: str) -> str:
            return "not json"

    with pytest.raises(RerankParseError):
        _strategy().rerank(query="q", documents=DOCS, model_client=BadClient(ANSWERS))


@pytest.mark.asyncio
async def test_async_orders_by_answer_confidence() -> None:
    client = AsyncFakeAnswerClient(ANSWERS)
    estimator = FakeEstimator({"a": 0.9, "b": 0.3, "c": 0.7, "d": 0.1})
    results = await AsyncAnswerConfidenceRerankStrategy(estimator=estimator).rerank(
        query="q", documents=DOCS, model_client=client
    )
    assert [r.document.text for r in results] == ["a", "c", "b", "d"]
    assert client.calls == ["a", "b", "c", "d"]


def test_metadata_carries_strategy_and_algorithm_keys() -> None:
    client = FakeAnswerClient(ANSWERS)
    results = _strategy().rerank(query="q", documents=DOCS, model_client=client)

    assert results[0].metadata["strategy"] == "answer_confidence"
    assert results[0].metadata["algorithm"] == "answer_confidence"


def test_blank_document_fast_fails_before_any_answer_call() -> None:
    client = FakeAnswerClient(ANSWERS)
    documents = [Document(text="a"), Document(text="   ")]

    with pytest.raises(RerankInputError, match="documents\\[1\\]"):
        _strategy().rerank(query="q", documents=documents, model_client=client)
    assert client.calls == []


def test_rejects_negative_top_k() -> None:
    client = FakeAnswerClient(ANSWERS)
    with pytest.raises(RerankInputError, match="top_k"):
        _strategy().rerank(query="q", documents=DOCS, model_client=client, top_k=-1)


@pytest.mark.asyncio
async def test_async_invalid_answer_json_fast_fails() -> None:
    from ranksmith.errors import RerankParseError

    class AsyncBadClient(AsyncFakeAnswerClient):
        async def answer(self, query: str, context: str) -> str:
            return "not json"

    estimator = FakeEstimator({"a": 0.9, "b": 0.3, "c": 0.7, "d": 0.1})
    with pytest.raises(RerankParseError):
        await AsyncAnswerConfidenceRerankStrategy(estimator=estimator).rerank(
            query="q", documents=DOCS, model_client=AsyncBadClient(ANSWERS)
        )


@pytest.mark.asyncio
async def test_async_rejects_client_without_answer() -> None:
    class AsyncNoAnswerClient:
        async def rank(self, query: str, documents: list[Document]) -> str:
            return "{}"

    estimator = FakeEstimator({"a": 0.9})
    with pytest.raises(RerankInputError, match="answer"):
        await AsyncAnswerConfidenceRerankStrategy(estimator=estimator).rerank(
            query="q", documents=DOCS, model_client=AsyncNoAnswerClient()
        )


@pytest.mark.asyncio
async def test_async_blank_document_fast_fails_before_any_answer_call() -> None:
    client = AsyncFakeAnswerClient(ANSWERS)
    estimator = FakeEstimator({"a": 0.9})
    documents = [Document(text="a"), Document(text="")]

    with pytest.raises(RerankInputError, match="documents\\[1\\]"):
        await AsyncAnswerConfidenceRerankStrategy(estimator=estimator).rerank(
            query="q", documents=documents, model_client=client
        )
    assert client.calls == []
