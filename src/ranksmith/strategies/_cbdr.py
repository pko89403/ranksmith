from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from ranksmith.errors import RerankInputError
from ranksmith.types import Document, RerankResult

from ._common import validate_documents_max_chars, validate_top_k
from ._confidence_gain import (
    AnswerGenerator,
    ConfidenceEstimator,
    _call_answer_query,
    _call_answer_with_context,
    _confidence_gain,
    _score_base_answerability,
    _score_context_answerability,
    _validate_confidence_score,
    _validate_estimator_tasks,
)

CBDRAlgorithm = Literal["cbdr"]


@dataclass(frozen=True)
class CBDRStrategy:
    base_estimator: ConfidenceEstimator
    context_estimator: ConfidenceEstimator
    answer_generator: AnswerGenerator
    skip_threshold: float = 0.8
    max_document_chars: int = 4000
    algorithm: CBDRAlgorithm = "cbdr"

    def __post_init__(self) -> None:
        if self.algorithm != "cbdr":
            raise ValueError('algorithm must be "cbdr"')
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")
        _validate_probability_config(self.skip_threshold, "skip_threshold")
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
        if not documents or top_k == 0:
            return []

        base_answer = _call_answer_query(self.answer_generator, query)
        base_result = _score_base_answerability(
            estimator=self.base_estimator,
            query=query,
            answer=base_answer,
        )
        base_score = _validate_confidence_score(base_result.score, "base")

        if base_score >= self.skip_threshold:
            return _original_order_results(
                documents=documents,
                top_k=top_k,
                algorithm=self.algorithm,
                base_score=base_score,
                skip_threshold=self.skip_threshold,
            )

        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )
        scored = []
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
            scored.append(
                (
                    original_index,
                    context_score,
                    _confidence_gain(
                        base_score=base_score,
                        context_score=context_score,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[2], item[0]))
        if top_k is not None:
            scored = scored[:top_k]

        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "cbdr",
                    "algorithm": self.algorithm,
                    "cbdr_skipped": False,
                    "base_confidence": base_score,
                    "skip_threshold": self.skip_threshold,
                    "context_confidence": context_score,
                    "confidence_gain": gain,
                },
            )
            for rank, (original_index, context_score, gain) in enumerate(
                scored,
                start=1,
            )
        ]


def _validate_probability_config(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


def _original_order_results(
    *,
    documents: Sequence[Document],
    top_k: int | None,
    algorithm: CBDRAlgorithm,
    base_score: float,
    skip_threshold: float,
) -> list[RerankResult]:
    indexed = list(enumerate(documents))
    if top_k is not None:
        indexed = indexed[:top_k]
    return [
        RerankResult(
            document=document,
            rank=rank,
            original_index=original_index,
            metadata={
                "strategy": "cbdr",
                "algorithm": algorithm,
                "cbdr_skipped": True,
                "base_confidence": base_score,
                "skip_threshold": skip_threshold,
                "context_confidence": None,
                "confidence_gain": None,
            },
        )
        for rank, (original_index, document) in enumerate(indexed, start=1)
    ]
