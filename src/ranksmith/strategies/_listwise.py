from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ranksmith.errors import RerankInputError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_ranking_response
from ranksmith.types import Document, RerankResult

from ._common import (
    ensure_async_listwise_model_client,
    ensure_listwise_model_client,
    validate_documents_max_chars,
    validate_top_k,
)

Algorithm = Literal["rankgpt_sliding_window"]


@dataclass(frozen=True)
class _ListwiseConfigMixin:
    algorithm: Algorithm = "rankgpt_sliding_window"
    window_size: int = 20
    stride: int = 10
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.algorithm != "rankgpt_sliding_window":
            raise ValueError('algorithm must be "rankgpt_sliding_window"')
        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0")
        if self.stride < 1:
            raise ValueError("stride must be greater than 0")
        if (
            self.algorithm == "rankgpt_sliding_window"
            and self.stride > self.window_size
        ):
            raise RerankInputError(
                "stride must be less than or equal to window_size "
                'for "rankgpt_sliding_window".'
            )
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )


@dataclass(frozen=True)
class ListwiseStrategy(_ListwiseConfigMixin):
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

        model_client = ensure_listwise_model_client(model_client)
        if len(documents) <= self.window_size:
            ordered_indexes = self._rank_window(query, documents, model_client)
        else:
            ordered_indexes = self._rank_rankgpt_sliding_windows(
                query, documents, model_client
            )

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
        return results[:top_k]

    def _rank_window(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
    ) -> list[int]:
        raw_response = model_client.rank(query, list(documents))
        ranking = parse_ranking_response(raw_response, expected_count=len(documents))
        return [number - 1 for number in ranking]

    def _rank_rankgpt_sliding_windows(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
    ) -> list[int]:
        document_count = len(documents)
        current_order = list(range(document_count))

        start_pos = document_count - self.window_size
        while True:
            start_pos = max(0, start_pos)

            window_indices = current_order[start_pos : start_pos + self.window_size]
            window_documents = [documents[i] for i in window_indices]

            raw_response = model_client.rank(query, window_documents)
            ranking = parse_ranking_response(
                raw_response,
                expected_count=len(window_documents),
            )

            new_window_indices = [window_indices[idx - 1] for idx in ranking]
            current_order[start_pos : start_pos + self.window_size] = new_window_indices

            if start_pos == 0:
                break

            start_pos -= self.stride

        return current_order


@dataclass(frozen=True)
class AsyncListwiseStrategy(_ListwiseConfigMixin):
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

        model_client = ensure_async_listwise_model_client(model_client)
        if len(documents) <= self.window_size:
            ordered_indexes = await self._rank_window(query, documents, model_client)
        else:
            ordered_indexes = await self._rank_rankgpt_sliding_windows(
                query, documents, model_client
            )

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
        return results[:top_k]

    async def _rank_window(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
    ) -> list[int]:
        raw_response = await model_client.rank(query, list(documents))
        ranking = parse_ranking_response(raw_response, expected_count=len(documents))
        return [number - 1 for number in ranking]

    async def _rank_rankgpt_sliding_windows(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
    ) -> list[int]:
        document_count = len(documents)
        current_order = list(range(document_count))

        start_pos = document_count - self.window_size
        while True:
            start_pos = max(0, start_pos)

            window_indices = current_order[start_pos : start_pos + self.window_size]
            window_documents = [documents[i] for i in window_indices]

            raw_response = await model_client.rank(query, window_documents)
            ranking = parse_ranking_response(
                raw_response,
                expected_count=len(window_documents),
            )

            new_window_indices = [window_indices[idx - 1] for idx in ranking]
            current_order[start_pos : start_pos + self.window_size] = new_window_indices

            if start_pos == 0:
                break

            start_pos -= self.stride

        return current_order
