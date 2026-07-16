from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ranksmith.confidence.types import AnswerConfidenceInput
from ranksmith.errors import RerankInputError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_answer_response
from ranksmith.types import Document, RerankResult

from .common import (
    ensure_capability,
    validate_documents_max_chars,
    validate_top_k,
)


class _ConfidenceResult(Protocol):
    @property
    def score(self) -> float: ...


class AnswerConfidenceScorer(Protocol):
    # Read-only property members so frozen estimators (for example
    # StructuralConfidenceEstimator with its Literal task_type) satisfy the
    # protocol; plain attribute members would demand settable fields.
    @property
    def task_type(self) -> str: ...

    def score(self, item: AnswerConfidenceInput) -> _ConfidenceResult: ...


@dataclass(frozen=True)
class _AnswerConfidenceRerankConfigMixin:
    """Confidence-change reranking (CBDR-style), black-box via structural confidence.

    For each candidate the model answers the query from that document, and the
    document's confidence is the local structural confidence that the answer is
    correct. Documents are ordered by that confidence, descending.

    This ranks by confidence change: CBDR ranks by Inc = C(query+doc) - C(query).
    Within a single query the baseline C(query) is a constant, so it does not
    affect the order — and answer_confidence cannot score an empty context
    anyway — so the reranker ranks by C(query+doc) directly.
    """

    estimator: AnswerConfidenceScorer
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.estimator.task_type != "answer_confidence":
            raise RerankInputError('estimator.task_type must be "answer_confidence"')
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )
        # answer_confidence cannot score an empty context; fail before any
        # paid answer() call instead of surfacing a non-RerankError from the
        # estimator afterwards.
        for index, document in enumerate(documents):
            if document.text.strip() == "":
                raise RerankInputError(
                    f"documents[{index}].text must not be empty or "
                    "whitespace-only for answer_confidence."
                )

    def _confidence(self, *, document: Document, answer: str) -> float:
        return self.estimator.score(
            AnswerConfidenceInput(context=document.text, answer=answer)
        ).score

    def _results_from_confidence(
        self,
        documents: Sequence[Document],
        confidences: Sequence[float],
        top_k: int | None,
    ) -> list[RerankResult]:
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-confidences[index], index),
        )
        if top_k is not None:
            ordered_indexes = ordered_indexes[:top_k]
        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "answer_confidence",
                    "algorithm": "answer_confidence",
                    "answer_confidence": confidences[original_index],
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]


@dataclass(frozen=True)
class AnswerConfidenceRerankStrategy(_AnswerConfidenceRerankConfigMixin):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        validate_top_k(top_k)
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = ensure_capability(model_client, "answer", "answer")
        confidences: list[float] = []
        for document in documents:
            answer = parse_answer_response(model_client.answer(query, document.text))
            confidences.append(self._confidence(document=document, answer=answer))
        return self._results_from_confidence(documents, confidences, top_k)


@dataclass(frozen=True)
class AsyncAnswerConfidenceRerankStrategy(_AnswerConfidenceRerankConfigMixin):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        validate_top_k(top_k)
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = ensure_capability(model_client, "answer", "answer")
        # Answer generation is I/O-bound: run the calls concurrently. Scoring is
        # a synchronous CPU/encoder step, so it stays sequential afterwards.
        raw_answers = await asyncio.gather(
            *(model_client.answer(query, document.text) for document in documents)
        )
        confidences: list[float] = []
        for document, raw in zip(documents, raw_answers, strict=True):
            answer = parse_answer_response(raw)
            confidences.append(self._confidence(document=document, answer=answer))
        return self._results_from_confidence(documents, confidences, top_k)
