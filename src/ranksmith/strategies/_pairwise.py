from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ranksmith.errors import RerankParseError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.types import Document, RerankResult

from ._common import (
    ensure_async_pairwise_model_client,
    ensure_pairwise_model_client,
    validate_documents_max_chars,
    validate_top_k,
)

PairwiseAlgorithm = Literal["prp_sliding_k"]


@dataclass(frozen=True)
class _PairwiseConfigMixin:
    algorithm: PairwiseAlgorithm = "prp_sliding_k"
    passes: int = 10
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.algorithm != "prp_sliding_k":
            raise ValueError('algorithm must be "prp_sliding_k"')
        if self.passes < 1:
            raise ValueError("passes must be greater than 0")
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )


@dataclass(frozen=True)
class PairwiseStrategy(_PairwiseConfigMixin):
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

        model_client = ensure_pairwise_model_client(model_client)
        ordered_indexes = self._rank_prp_sliding_k(query, documents, model_client)

        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "pairwise", "algorithm": self.algorithm},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        if top_k is None:
            return results
        return results[:top_k]

    def _rank_prp_sliding_k(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
    ) -> list[int]:
        current_order = list(range(len(documents)))

        for _ in range(self.passes):
            for right_pos in range(len(current_order) - 1, 0, -1):
                left_pos = right_pos - 1
                left_index = current_order[left_pos]
                right_index = current_order[right_pos]

                first = _parse_pairwise_winner_response(
                    model_client.compare(
                        query,
                        documents[left_index],
                        documents[right_index],
                    )
                )
                second = _parse_pairwise_winner_response(
                    model_client.compare(
                        query,
                        documents[right_index],
                        documents[left_index],
                    )
                )

                first_winner = left_index if first == "A" else right_index
                second_winner = right_index if second == "A" else left_index

                if first_winner == second_winner and first_winner == right_index:
                    current_order[left_pos], current_order[right_pos] = (
                        current_order[right_pos],
                        current_order[left_pos],
                    )

        return current_order


@dataclass(frozen=True)
class AsyncPairwiseStrategy(_PairwiseConfigMixin):
    pair_order_parallelism: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pair_order_parallelism not in {1, 2}:
            raise ValueError("pair_order_parallelism must be 1 or 2")

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

        model_client = ensure_async_pairwise_model_client(model_client)
        ordered_indexes = await self._rank_prp_sliding_k(query, documents, model_client)

        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "pairwise", "algorithm": self.algorithm},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        if top_k is None:
            return results
        return results[:top_k]

    async def _rank_prp_sliding_k(
        self,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
    ) -> list[int]:
        current_order = list(range(len(documents)))

        for _ in range(self.passes):
            for right_pos in range(len(current_order) - 1, 0, -1):
                left_pos = right_pos - 1
                left_index = current_order[left_pos]
                right_index = current_order[right_pos]

                first_raw, second_raw = await self._compare_pair_orders(
                    query=query,
                    documents=documents,
                    model_client=model_client,
                    left_index=left_index,
                    right_index=right_index,
                )
                first = _parse_pairwise_winner_response(first_raw)
                second = _parse_pairwise_winner_response(second_raw)

                first_winner = left_index if first == "A" else right_index
                second_winner = right_index if second == "A" else left_index

                if first_winner == second_winner and first_winner == right_index:
                    current_order[left_pos], current_order[right_pos] = (
                        current_order[right_pos],
                        current_order[left_pos],
                    )

        return current_order

    async def _compare_pair_orders(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        left_index: int,
        right_index: int,
    ) -> tuple[str, str]:
        if self.pair_order_parallelism == 1:
            first_raw = await model_client.compare(
                query,
                documents[left_index],
                documents[right_index],
            )
            second_raw = await model_client.compare(
                query,
                documents[right_index],
                documents[left_index],
            )
            return first_raw, second_raw

        return await asyncio.gather(
            model_client.compare(
                query,
                documents[left_index],
                documents[right_index],
            ),
            model_client.compare(
                query,
                documents[right_index],
                documents[left_index],
            ),
        )


def _parse_pairwise_winner_response(raw_response: str) -> Literal["A", "B"]:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RerankParseError("LLM response is not valid JSON.", raw_response) from exc

    winner = data.get("winner") if isinstance(data, dict) else None
    if winner not in {"A", "B"}:
        raise RerankParseError(
            'LLM response must contain a "winner" value of "A" or "B".',
            raw_response,
        )
    return cast(Literal["A", "B"], winner)
