from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, cast

import pytest

from ranksmith import AzureOpenAIReranker
from ranksmith.confidence import StructuralConfidenceResult, TaskType
from ranksmith.errors import (
    DocumentTooLongError,
    RerankInputError,
    RerankProviderError,
    RerankStrategyError,
)
from ranksmith.strategies import CBDRStrategy
from ranksmith.types import Document


@dataclass
class FakeEstimator:
    task_type: TaskType
    scores: list[Any]
    calls: list[object] | None = None

    def score(self, item: object) -> StructuralConfidenceResult:
        if self.calls is not None:
            self.calls.append(item)
        value = self.scores.pop(0)
        if isinstance(value, BaseException):
            raise value
        return StructuralConfidenceResult(
            score=value,
            task_type=self.task_type,
            feature_schema_version="structural-v1",
        )


class FakeGenerator:
    def __init__(
        self,
        *,
        base_answer: object = "base answer",
        context_answers: list[object] | None = None,
    ) -> None:
        self.base_answer = base_answer
        self.context_answers = context_answers or []
        self.query_calls: list[str] = []
        self.context_calls: list[tuple[str, str]] = []

    def answer_query(self, query: str) -> Any:
        self.query_calls.append(query)
        if isinstance(self.base_answer, BaseException):
            raise self.base_answer
        return self.base_answer

    def answer_with_context(self, query: str, context: str) -> Any:
        self.context_calls.append((query, context))
        answer = self.context_answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _strategy(
    *,
    base_scores: list[Any] | None = None,
    context_scores: list[Any] | None = None,
    generator: FakeGenerator | None = None,
    skip_threshold: float = 0.8,
    max_document_chars: int = 4000,
) -> CBDRStrategy:
    return CBDRStrategy(
        base_estimator=FakeEstimator(
            task_type="query_answerability_confidence",
            scores=base_scores or [0.2],
        ),
        context_estimator=FakeEstimator(
            task_type="query_context_answerability_confidence",
            scores=context_scores or [0.7],
        ),
        answer_generator=generator or FakeGenerator(context_answers=["context answer"]),
        skip_threshold=skip_threshold,
        max_document_chars=max_document_chars,
    )


def _unused_model_client() -> Any:
    return object()


def test_cbdr_exports_are_submodule_only() -> None:
    strategies = importlib.import_module("ranksmith.strategies")
    root = importlib.import_module("ranksmith")

    assert strategies.CBDRStrategy is not None
    assert not hasattr(root, "CBDRStrategy")


def test_cbdr_empty_documents_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(generator=generator)

    assert (
        strategy.rerank(
            query="Who?",
            documents=[],
            model_client=object(),
        )
        == []
    )
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_cbdr_top_k_zero_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=["a"])
    strategy = _strategy(generator=generator)

    assert (
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
            top_k=0,
        )
        == []
    )
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_cbdr_skip_path_preserves_original_order_and_metadata() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(base_scores=[0.91], generator=generator, skip_threshold=0.8)
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
    ]

    results = strategy.rerank(query="Who?", documents=documents, model_client=object())

    assert [result.document.id for result in results] == ["a", "b"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [0, 1]
    assert [dict(result.metadata) for result in results] == [
        {
            "strategy": "cbdr",
            "algorithm": "cbdr",
            "cbdr_skipped": True,
            "base_confidence": 0.91,
            "skip_threshold": 0.8,
            "context_confidence": None,
            "confidence_gain": None,
        },
        {
            "strategy": "cbdr",
            "algorithm": "cbdr",
            "cbdr_skipped": True,
            "base_confidence": 0.91,
            "skip_threshold": 0.8,
            "context_confidence": None,
            "confidence_gain": None,
        },
    ]
    assert generator.query_calls == ["Who?"]
    assert generator.context_calls == []


def test_cbdr_skip_path_applies_top_k_after_original_order() -> None:
    strategy = _strategy(base_scores=[0.9], skip_threshold=0.8)

    results = strategy.rerank(
        query="Who?",
        documents=[
            Document(id="a", text="alpha"),
            Document(id="b", text="beta"),
        ],
        model_client=object(),
        top_k=1,
    )

    assert [result.document.id for result in results] == ["a"]
    assert [result.rank for result in results] == [1]
    assert [result.original_index for result in results] == [0]


def test_cbdr_skip_path_does_not_validate_long_documents() -> None:
    strategy = _strategy(
        base_scores=[0.9],
        context_scores=[],
        generator=FakeGenerator(context_answers=[]),
        skip_threshold=0.8,
        max_document_chars=3,
    )

    results = strategy.rerank(
        query="Who?",
        documents=[Document(text="abcdef")],
        model_client=object(),
    )

    assert len(results) == 1
    assert results[0].metadata["cbdr_skipped"] is True


def test_cbdr_rerank_path_sorts_by_gain_and_preserves_ties() -> None:
    generator = FakeGenerator(context_answers=["answer a", "answer b", "answer c"])
    strategy = _strategy(
        base_scores=[0.4],
        context_scores=[0.6, 0.8, 0.8],
        generator=generator,
        skip_threshold=0.9,
    )

    results = strategy.rerank(
        query="Who?",
        documents=[
            Document(id="a", text="alpha"),
            Document(id="b", text="beta"),
            Document(id="c", text="gamma"),
        ],
        model_client=object(),
        top_k=2,
    )

    assert [result.document.id for result in results] == ["b", "c"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [1, 2]
    assert [result.metadata["cbdr_skipped"] for result in results] == [False, False]
    assert [result.metadata["base_confidence"] for result in results] == [0.4, 0.4]
    assert [result.metadata["context_confidence"] for result in results] == [0.8, 0.8]
    assert [result.metadata["confidence_gain"] for result in results] == pytest.approx(
        [0.4, 0.4]
    )
    assert generator.context_calls == [
        ("Who?", "alpha"),
        ("Who?", "beta"),
        ("Who?", "gamma"),
    ]


@pytest.mark.parametrize("skip_threshold", [math.nan, math.inf, -0.1, 1.1, True])
def test_cbdr_invalid_skip_threshold_fails(skip_threshold: object) -> None:
    with pytest.raises(ValueError, match="skip_threshold"):
        CBDRStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
            skip_threshold=cast(float, skip_threshold),
        )


def test_cbdr_threshold_zero_always_skips_non_empty_documents() -> None:
    strategy = _strategy(base_scores=[0.0], skip_threshold=0.0)

    results = strategy.rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )

    assert results[0].metadata["cbdr_skipped"] is True


def test_cbdr_threshold_one_skips_only_at_exact_one() -> None:
    skipped = _strategy(base_scores=[1.0], skip_threshold=1.0).rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )
    reranked = _strategy(
        base_scores=[0.999],
        context_scores=[1.0],
        generator=FakeGenerator(context_answers=["a"]),
        skip_threshold=1.0,
    ).rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )

    assert skipped[0].metadata["cbdr_skipped"] is True
    assert reranked[0].metadata["cbdr_skipped"] is False


def test_cbdr_rerank_path_validates_long_documents() -> None:
    strategy = _strategy(base_scores=[0.2], skip_threshold=0.8, max_document_chars=3)

    with pytest.raises(DocumentTooLongError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="abcdef")],
            model_client=object(),
        )


def test_cbdr_empty_query_fails() -> None:
    with pytest.raises(RerankInputError, match="query"):
        _strategy().rerank(
            query="  ",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_negative_top_k_fails() -> None:
    with pytest.raises(RerankInputError, match="top_k"):
        _strategy().rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
            top_k=-1,
        )


def test_cbdr_invalid_task_types_fail() -> None:
    with pytest.raises(RerankInputError, match="base_estimator"):
        CBDRStrategy(
            base_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
        )

    with pytest.raises(RerankInputError, match="context_estimator"):
        CBDRStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_cbdr_invalid_base_score_fails(score: object) -> None:
    with pytest.raises(RerankStrategyError):
        _strategy(base_scores=[score]).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_cbdr_invalid_context_score_fails(score: object) -> None:
    with pytest.raises(RerankStrategyError):
        _strategy(
            base_scores=[0.1],
            context_scores=[score],
            skip_threshold=0.8,
        ).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_answer_generator_empty_output_fails() -> None:
    with pytest.raises(RerankProviderError):
        _strategy(generator=FakeGenerator(base_answer="")).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_generator_unexpected_exception_wraps_provider_error() -> None:
    with pytest.raises(RerankProviderError) as exc_info:
        _strategy(generator=FakeGenerator(base_answer=RuntimeError("boom"))).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_cbdr_direct_estimator_unexpected_exception_propagates() -> None:
    error = RuntimeError("confidence failed")

    with pytest.raises(RuntimeError, match="confidence failed"):
        _strategy(base_scores=[error]).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_facade_wraps_unexpected_estimator_error() -> None:
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(base_scores=[RuntimeError("confidence failed")]),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
