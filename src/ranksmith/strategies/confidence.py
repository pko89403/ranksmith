from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ranksmith.confidence.types import JudgmentConfidenceInput
from ranksmith.errors import RerankInputError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import JudgmentValue, parse_judgment_response
from ranksmith.types import Document, RerankResult

from .common import (
    ensure_capability,
    validate_documents_max_chars,
    validate_top_k,
)


class _ConfidenceResult(Protocol):
    score: float


class JudgmentConfidenceScorer(Protocol):
    task_type: str

    def score(self, item: JudgmentConfidenceInput) -> _ConfidenceResult: ...


@dataclass(frozen=True)
class _ConfidenceRerankConfigMixin:
    estimator: JudgmentConfidenceScorer
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.estimator.task_type != "judgment_confidence":
            raise RerankInputError(
                'estimator.task_type must be "judgment_confidence"'
            )
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )

    def _signed_confidence(
        self,
        *,
        query: str,
        document: Document,
        judgment: JudgmentValue,
    ) -> float:
        confidence = self.estimator.score(
            JudgmentConfidenceInput(
                query=query,
                document=document.text,
                judgment=judgment,
            )
        ).score
        return confidence if judgment == "relevant" else -confidence

    def _results_from_signed(
        self,
        documents: Sequence[Document],
        signed: Sequence[tuple[float, JudgmentValue]],
        top_k: int | None,
    ) -> list[RerankResult]:
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-signed[index][0], index),
        )
        if top_k is not None:
            ordered_indexes = ordered_indexes[:top_k]
        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "confidence",
                    "judgment": signed[original_index][1],
                    "signed_confidence": signed[original_index][0],
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]


@dataclass(frozen=True)
class ConfidenceRerankStrategy(_ConfidenceRerankConfigMixin):
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

        model_client = ensure_capability(model_client, "relevance", "judge")
        signed: list[tuple[float, JudgmentValue]] = []
        for document in documents:
            judgment = parse_judgment_response(model_client.judge(query, document))
            signed.append(
                (
                    self._signed_confidence(
                        query=query,
                        document=document,
                        judgment=judgment,
                    ),
                    judgment,
                )
            )
        return self._results_from_signed(documents, signed, top_k)


@dataclass(frozen=True)
class AsyncConfidenceRerankStrategy(_ConfidenceRerankConfigMixin):
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

        model_client = ensure_capability(model_client, "relevance", "judge")
        # Judge calls are I/O-bound: run them concurrently. Scoring is a
        # synchronous CPU/encoder step, so it stays sequential afterwards.
        raw_judgments = await asyncio.gather(
            *(model_client.judge(query, document) for document in documents)
        )
        signed: list[tuple[float, JudgmentValue]] = []
        for document, raw in zip(documents, raw_judgments, strict=True):
            judgment = parse_judgment_response(raw)
            signed.append(
                (
                    self._signed_confidence(
                        query=query,
                        document=document,
                        judgment=judgment,
                    ),
                    judgment,
                )
            )
        return self._results_from_signed(documents, signed, top_k)
