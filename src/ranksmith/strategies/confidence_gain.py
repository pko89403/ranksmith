from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

from ranksmith.confidence import (
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    StructuralConfidenceResult,
)
from ranksmith.errors import (
    RerankInputError,
    RerankProviderError,
    RerankStrategyError,
)
from ranksmith.types import Document, RerankResult

from .common import validate_documents_max_chars, validate_top_k

ConfidenceGainAlgorithm = Literal["confidence_gain"]
QUERY_ANSWERABILITY_TASK = "query_answerability_confidence"
QUERY_CONTEXT_ANSWERABILITY_TASK = "query_context_answerability_confidence"
AnswerabilityConfidenceInput = (
    QueryAnswerabilityConfidenceInput | QueryContextAnswerabilityConfidenceInput
)


class AnswerGenerator(Protocol):
    def answer_query(self, query: str) -> str: ...

    def answer_with_context(self, query: str, context: str) -> str: ...


class ConfidenceEstimator(Protocol):
    @property
    def task_type(self) -> str: ...

    def score(
        self,
        item: AnswerabilityConfidenceInput,
    ) -> StructuralConfidenceResult: ...


@dataclass(frozen=True)
class ConfidenceGainResult:
    base_score: float
    context_score: float
    gain: float
    base_result: StructuralConfidenceResult
    context_result: StructuralConfidenceResult


@dataclass(frozen=True)
class ConfidenceGainStrategy:
    base_estimator: ConfidenceEstimator
    context_estimator: ConfidenceEstimator
    answer_generator: AnswerGenerator
    max_document_chars: int = 4000
    algorithm: ConfidenceGainAlgorithm = "confidence_gain"

    def __post_init__(self) -> None:
        if self.algorithm != "confidence_gain":
            raise ValueError('algorithm must be "confidence_gain"')
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")
        _validate_estimator_tasks(
            base_estimator=self.base_estimator,
            context_estimator=self.context_estimator,
        )

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del model_client
        validate_top_k(top_k)
        if query.strip() == "":
            raise RerankInputError("query must not be empty")
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )
        if not documents:
            return []

        base_answer = _call_answer_query(self.answer_generator, query)
        base_result = _score_base_answerability(
            estimator=self.base_estimator,
            query=query,
            answer=base_answer,
        )
        base_score = _validate_confidence_score(base_result.score, "base")

        scored: list[tuple[int, ConfidenceGainResult]] = []
        for original_index, document in enumerate(documents):
            context_answer = _call_answer_with_context(
                self.answer_generator,
                query,
                document.text,
            )
            context_result = _score_context_answerability(
                estimator=self.context_estimator,
                query=query,
                context=document.text,
                answer=context_answer,
            )
            context_score = _validate_confidence_score(
                context_result.score,
                "context",
            )
            gain = _confidence_gain(
                base_score=base_score,
                context_score=context_score,
            )
            scored.append(
                (
                    original_index,
                    ConfidenceGainResult(
                        base_score=base_score,
                        context_score=context_score,
                        gain=gain,
                        base_result=base_result,
                        context_result=context_result,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[1].gain, item[0]))
        if top_k is not None:
            scored = scored[:top_k]

        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "confidence_gain",
                    "algorithm": self.algorithm,
                    "base_confidence": result.base_score,
                    "context_confidence": result.context_score,
                    "confidence_gain": result.gain,
                },
            )
            for rank, (original_index, result) in enumerate(scored, start=1)
        ]


def _validate_estimator_tasks(
    *,
    base_estimator: ConfidenceEstimator,
    context_estimator: ConfidenceEstimator,
) -> None:
    if base_estimator.task_type != QUERY_ANSWERABILITY_TASK:
        raise RerankInputError(
            f"base_estimator task_type must be {QUERY_ANSWERABILITY_TASK!r}"
        )
    if context_estimator.task_type != QUERY_CONTEXT_ANSWERABILITY_TASK:
        raise RerankInputError(
            f"context_estimator task_type must be {QUERY_CONTEXT_ANSWERABILITY_TASK!r}"
        )


def _call_answer_query(generator: AnswerGenerator, query: str) -> str:
    try:
        answer = generator.answer_query(query)
    except RerankProviderError:
        raise
    except Exception as exc:
        raise RerankProviderError(str(exc)) from exc
    return _validate_answer(answer, "answer_query")


def _call_answer_with_context(
    generator: AnswerGenerator,
    query: str,
    context: str,
) -> str:
    try:
        answer = generator.answer_with_context(query, context)
    except RerankProviderError:
        raise
    except Exception as exc:
        raise RerankProviderError(str(exc)) from exc
    return _validate_answer(answer, "answer_with_context")


def _validate_answer(answer: object, method_name: str) -> str:
    if not isinstance(answer, str) or answer.strip() == "":
        raise RerankProviderError(f"{method_name} must return a non-empty string")
    return answer


def _score_base_answerability(
    *,
    estimator: ConfidenceEstimator,
    query: str,
    answer: str,
) -> StructuralConfidenceResult:
    return estimator.score(
        QueryAnswerabilityConfidenceInput(query=query, answer=answer)
    )


def _score_context_answerability(
    *,
    estimator: ConfidenceEstimator,
    query: str,
    context: str,
    answer: str,
) -> StructuralConfidenceResult:
    return estimator.score(
        QueryContextAnswerabilityConfidenceInput(
            query=query,
            context=context,
            answer=answer,
        )
    )


def _validate_confidence_score(score: object, label: str) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise RerankStrategyError(f"{label} confidence score must be numeric")
    value = float(score)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RerankStrategyError(f"{label} confidence score must be finite in [0, 1]")
    return value


def _confidence_gain(
    *,
    base_score: float,
    context_score: float,
) -> float:
    gain = context_score - base_score
    if not math.isfinite(gain) or gain < -1.0 or gain > 1.0:
        raise RerankStrategyError("confidence gain must be finite in [-1, 1]")
    return gain
