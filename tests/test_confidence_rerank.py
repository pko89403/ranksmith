from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ranksmith import (
    AsyncConfidenceRerankStrategy,
    ConfidenceRerankStrategy,
    Document,
)
from ranksmith.confidence.types import JudgmentConfidenceInput
from ranksmith.errors import RerankInputError


@dataclass
class _Result:
    score: float


class FakeEstimator:
    """Judgment-confidence estimator with scripted per-document scores."""

    task_type = "judgment_confidence"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, item: JudgmentConfidenceInput) -> _Result:
        return _Result(score=self._scores[item.document])


class FakeJudgeClient:
    """Model client that returns scripted relevance judgments per document."""

    def __init__(self, judgments: dict[str, str]) -> None:
        self._judgments = judgments
        self.calls: list[str] = []

    def judge(self, query: str, document: Document) -> str:
        self.calls.append(document.text)
        return json.dumps({"judgment": self._judgments[document.text]})


class AsyncFakeJudgeClient(FakeJudgeClient):
    async def judge(self, query: str, document: Document) -> str:  # type: ignore[override]
        self.calls.append(document.text)
        return json.dumps({"judgment": self._judgments[document.text]})


DOCS = [Document(text="a"), Document(text="b"), Document(text="c"), Document(text="d")]


def _strategy() -> ConfidenceRerankStrategy:
    estimator = FakeEstimator({"a": 0.9, "b": 0.4, "c": 0.4, "d": 0.9})
    return ConfidenceRerankStrategy(estimator=estimator)


def test_signed_confidence_orders_relevant_high_to_not_relevant_high() -> None:
    # a: relevant/0.9 -> +0.9 ; b: relevant/0.4 -> +0.4 ;
    # c: not_relevant/0.4 -> -0.4 ; d: not_relevant/0.9 -> -0.9
    client = FakeJudgeClient(
        {"a": "relevant", "b": "relevant", "c": "not_relevant", "d": "not_relevant"}
    )
    results = _strategy().rerank(query="q", documents=DOCS, model_client=client)

    assert [r.document.text for r in results] == ["a", "b", "c", "d"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert results[0].metadata["signed_confidence"] == pytest.approx(0.9)
    assert results[3].metadata["signed_confidence"] == pytest.approx(-0.9)
    assert results[0].metadata["judgment"] == "relevant"
    assert results[3].metadata["judgment"] == "not_relevant"
    # one judge call per document, no repeated sampling
    assert client.calls == ["a", "b", "c", "d"]


def test_ties_preserve_original_order() -> None:
    client = FakeJudgeClient(
        {"a": "relevant", "b": "relevant", "c": "not_relevant", "d": "not_relevant"}
    )
    estimator = FakeEstimator({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    results = ConfidenceRerankStrategy(estimator=estimator).rerank(
        query="q", documents=DOCS, model_client=client
    )
    # a,b (+0.5) keep input order before c,d (-0.5)
    assert [r.document.text for r in results] == ["a", "b", "c", "d"]


def test_top_k_slices_after_ordering() -> None:
    client = FakeJudgeClient(
        {"a": "relevant", "b": "relevant", "c": "not_relevant", "d": "not_relevant"}
    )
    results = _strategy().rerank(
        query="q", documents=DOCS, model_client=client, top_k=2
    )
    assert [r.document.text for r in results] == ["a", "b"]


def test_empty_documents_returns_empty() -> None:
    client = FakeJudgeClient({})
    assert _strategy().rerank(query="q", documents=[], model_client=client) == []


def test_rejects_non_judgment_estimator() -> None:
    class AnswerEstimator:
        task_type = "answer_confidence"

        def score(self, item: JudgmentConfidenceInput) -> _Result:
            return _Result(score=0.5)

    with pytest.raises(RerankInputError, match="judgment_confidence"):
        ConfidenceRerankStrategy(estimator=AnswerEstimator())


def test_rejects_client_without_judge() -> None:
    class NoJudgeClient:
        def rank(self, query: str, documents: list[Document]) -> str:
            return "{}"

    with pytest.raises(RerankInputError, match="judge"):
        _strategy().rerank(query="q", documents=DOCS, model_client=NoJudgeClient())


def test_invalid_judgment_json_fast_fails() -> None:
    client = FakeJudgeClient({"a": "maybe", "b": "relevant", "c": "x", "d": "y"})
    from ranksmith.errors import RerankParseError

    with pytest.raises(RerankParseError):
        _strategy().rerank(query="q", documents=DOCS, model_client=client)


@pytest.mark.asyncio
async def test_async_signed_confidence_orders_documents() -> None:
    client = AsyncFakeJudgeClient(
        {"a": "relevant", "b": "relevant", "c": "not_relevant", "d": "not_relevant"}
    )
    estimator = FakeEstimator({"a": 0.9, "b": 0.4, "c": 0.4, "d": 0.9})
    results = await AsyncConfidenceRerankStrategy(estimator=estimator).rerank(
        query="q", documents=DOCS, model_client=client
    )
    assert [r.document.text for r in results] == ["a", "b", "c", "d"]
    assert client.calls == ["a", "b", "c", "d"]
