from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast

from ranksmith._providers import (
    AsyncLLMProvider,
    AsyncPairwiseLLMProvider,
    LLMProvider,
    PairwiseLLMProvider,
)
from ranksmith.errors import DocumentTooLongError, RerankInputError, RerankParseError
from ranksmith.types import Document, RerankResult

Algorithm = Literal["rankgpt_sliding_window"]
PairwiseAlgorithm = Literal["prp_sliding_k"]
Provider: TypeAlias = LLMProvider | PairwiseLLMProvider
AsyncProvider: TypeAlias = AsyncLLMProvider | AsyncPairwiseLLMProvider


class RerankStrategy(Protocol):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: Provider,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...


class AsyncRerankStrategy(Protocol):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: AsyncProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...


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


@dataclass(frozen=True)
class ListwiseStrategy(_ListwiseConfigMixin):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: Provider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        provider = _ensure_listwise_provider(provider)
        if len(documents) <= self.window_size:
            ordered_indexes = self._rank_window(query, documents, provider)
        else:
            ordered_indexes = self._rank_rankgpt_sliding_windows(
                query, documents, provider
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        return results[:top_k]

    def _rank_window(
        self,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
    ) -> list[int]:
        raw_response = provider.rank(query, list(documents))
        ranking = _parse_ranking(raw_response, expected_count=len(documents))
        return [number - 1 for number in ranking]

    def _rank_rankgpt_sliding_windows(
        self,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
    ) -> list[int]:
        document_count = len(documents)
        current_order = list(range(document_count))

        start_pos = document_count - self.window_size
        while True:
            start_pos = max(0, start_pos)

            window_indices = current_order[start_pos : start_pos + self.window_size]
            window_documents = [documents[i] for i in window_indices]

            raw_response = provider.rank(query, window_documents)
            ranking = _parse_ranking(raw_response, expected_count=len(window_documents))

            new_window_indices = [window_indices[idx - 1] for idx in ranking]
            current_order[start_pos : start_pos + self.window_size] = new_window_indices

            if start_pos == 0:
                break

            start_pos -= self.stride

        return current_order


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


@dataclass(frozen=True)
class PairwiseStrategy(_PairwiseConfigMixin):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: Provider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        provider = _ensure_pairwise_provider(provider)
        ordered_indexes = self._rank_prp_sliding_k(query, documents, provider)

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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        return results[:top_k]

    def _rank_prp_sliding_k(
        self,
        query: str,
        documents: Sequence[Document],
        provider: PairwiseLLMProvider,
    ) -> list[int]:
        current_order = list(range(len(documents)))

        for _ in range(self.passes):
            for right_pos in range(len(current_order) - 1, 0, -1):
                left_pos = right_pos - 1
                left_index = current_order[left_pos]
                right_index = current_order[right_pos]

                first = _parse_pairwise_winner(
                    provider.compare(
                        query,
                        documents[left_index],
                        documents[right_index],
                    )
                )
                second = _parse_pairwise_winner(
                    provider.compare(
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
class AsyncListwiseStrategy(_ListwiseConfigMixin):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: AsyncProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        provider = _ensure_async_listwise_provider(provider)
        if len(documents) <= self.window_size:
            ordered_indexes = await self._rank_window(query, documents, provider)
        else:
            ordered_indexes = await self._rank_rankgpt_sliding_windows(
                query, documents, provider
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        return results[:top_k]

    async def _rank_window(
        self,
        query: str,
        documents: Sequence[Document],
        provider: AsyncLLMProvider,
    ) -> list[int]:
        raw_response = await provider.rank(query, list(documents))
        ranking = _parse_ranking(raw_response, expected_count=len(documents))
        return [number - 1 for number in ranking]

    async def _rank_rankgpt_sliding_windows(
        self,
        query: str,
        documents: Sequence[Document],
        provider: AsyncLLMProvider,
    ) -> list[int]:
        document_count = len(documents)
        current_order = list(range(document_count))

        start_pos = document_count - self.window_size
        while True:
            start_pos = max(0, start_pos)

            window_indices = current_order[start_pos : start_pos + self.window_size]
            window_documents = [documents[i] for i in window_indices]

            raw_response = await provider.rank(query, window_documents)
            ranking = _parse_ranking(raw_response, expected_count=len(window_documents))

            new_window_indices = [window_indices[idx - 1] for idx in ranking]
            current_order[start_pos : start_pos + self.window_size] = new_window_indices

            if start_pos == 0:
                break

            start_pos -= self.stride

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
        provider: AsyncProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        provider = _ensure_async_pairwise_provider(provider)
        ordered_indexes = await self._rank_prp_sliding_k(query, documents, provider)

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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        return results[:top_k]

    async def _rank_prp_sliding_k(
        self,
        query: str,
        documents: Sequence[Document],
        provider: AsyncPairwiseLLMProvider,
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
                    provider=provider,
                    left_index=left_index,
                    right_index=right_index,
                )
                first = _parse_pairwise_winner(first_raw)
                second = _parse_pairwise_winner(second_raw)

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
        provider: AsyncPairwiseLLMProvider,
        left_index: int,
        right_index: int,
    ) -> tuple[str, str]:
        if self.pair_order_parallelism == 1:
            first_raw = await provider.compare(
                query,
                documents[left_index],
                documents[right_index],
            )
            second_raw = await provider.compare(
                query,
                documents[right_index],
                documents[left_index],
            )
            return first_raw, second_raw

        return await asyncio.gather(
            provider.compare(
                query,
                documents[left_index],
                documents[right_index],
            ),
            provider.compare(
                query,
                documents[right_index],
                documents[left_index],
            ),
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


def _parse_pairwise_winner(raw_response: str) -> Literal["A", "B"]:
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


def _ensure_listwise_provider(provider: object) -> LLMProvider:
    rank = getattr(provider, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(LLMProvider, provider)


def _ensure_async_listwise_provider(provider: object) -> AsyncLLMProvider:
    rank = getattr(provider, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(AsyncLLMProvider, provider)


def _ensure_pairwise_provider(provider: object) -> PairwiseLLMProvider:
    compare = getattr(provider, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(PairwiseLLMProvider, provider)


def _ensure_async_pairwise_provider(
    provider: object,
) -> AsyncPairwiseLLMProvider:
    compare = getattr(provider, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(AsyncPairwiseLLMProvider, provider)
