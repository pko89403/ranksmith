from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ranksmith._providers import LLMProvider
from ranksmith.errors import DocumentTooLongError, RerankInputError, RerankParseError
from ranksmith.types import Document, RerankResult

Algorithm = Literal["direct", "sliding_window"]


@dataclass(frozen=True)
class ListwiseStrategy:
    algorithm: Algorithm = "sliding_window"
    window_size: int = 20
    stride: int = 10
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.algorithm not in {"direct", "sliding_window"}:
            raise ValueError('algorithm must be "direct" or "sliding_window"')
        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0")
        if self.stride < 1:
            raise ValueError("stride must be greater than 0")
        if self.algorithm == "sliding_window" and self.stride > self.window_size:
            raise RerankInputError(
                "stride must be less than or equal to window_size for sliding_window."
            )
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        if self.algorithm == "direct" or len(documents) <= self.window_size:
            ordered_indexes = self._rank_window(query, documents, provider)
        else:
            ordered_indexes = self._rank_sliding_windows(query, documents, provider)

        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "listwise", "algorithm": self.algorithm},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        if top_k is None:
            return results
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        return results[:top_k]

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        for index, document in enumerate(documents):
            length = len(document.text)
            if length > self.max_document_chars:
                message = (
                    f"Document at index {index} has {length} characters, exceeding "
                    f"max_document_chars={self.max_document_chars}. Shorten the "
                    "document, chunk it before reranking, or increase "
                    "max_document_chars."
                )
                raise DocumentTooLongError(message)

    def _rank_window(
        self,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
    ) -> list[int]:
        raw_response = provider.rank(query, list(documents))
        ranking = _parse_ranking(raw_response, expected_count=len(documents))
        return [number - 1 for number in ranking]

    def _rank_sliding_windows(
        self,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
    ) -> list[int]:
        scores = [0.0 for _ in documents]
        last_evidence = [-1 for _ in documents]

        for window_number, start in enumerate(
            _window_starts(len(documents), self.window_size, self.stride)
        ):
            window_documents = list(documents[start : start + self.window_size])
            raw_response = provider.rank(query, window_documents)
            ranking = _parse_ranking(raw_response, expected_count=len(window_documents))
            for position, local_number in enumerate(ranking):
                original_index = start + local_number - 1
                scores[original_index] += len(window_documents) - position - 1
                last_evidence[original_index] = window_number

        if all(evidence == -1 for evidence in last_evidence):
            return self._rank_window(query, documents, provider)

        return sorted(
            range(len(documents)),
            key=lambda index: (scores[index], last_evidence[index], index),
            reverse=True,
        )


def _parse_ranking(raw_response: str, *, expected_count: int) -> list[int]:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RerankParseError("LLM response is not valid JSON.", raw_response) from exc

    ranking = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(ranking, list):
        raise RerankParseError(
            'LLM response must contain a "ranking" list.',
            raw_response,
        )
    if not all(isinstance(item, int) for item in ranking):
        raise RerankParseError("ranking must contain only integers.", raw_response)

    expected = set(range(1, expected_count + 1))
    actual = set(ranking)
    if len(ranking) != expected_count:
        raise RerankParseError(
            f"ranking must contain exactly {expected_count} items.",
            raw_response,
        )
    if actual != expected:
        raise RerankParseError(
            f"ranking must be a permutation of 1..{expected_count}.",
            raw_response,
        )
    return ranking


def _window_starts(document_count: int, window_size: int, stride: int) -> list[int]:
    last_start = document_count - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
