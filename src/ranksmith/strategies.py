from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, cast

from ranksmith.errors import DocumentTooLongError, RerankInputError, RerankParseError
from ranksmith.parsing import parse_ranking_response, parse_selection_response
from ranksmith.protocols import (
    AsyncLLMProvider,
    AsyncPairwiseLLMProvider,
    AsyncProvider,
    AsyncSelectionLLMProvider,
    LLMProvider,
    PairwiseLLMProvider,
    Provider,
    SelectionLLMProvider,
)
from ranksmith.types import Document, RerankResult

Algorithm = Literal["rankgpt_sliding_window"]
PairwiseAlgorithm = Literal["prp_sliding_k"]
TourRankAlgorithm = Literal["tourrank_r"]


@dataclass(frozen=True)
class TourRankStageConfig:
    group_count: int
    group_size: int
    selected_count: int

    def __post_init__(self) -> None:
        if self.group_count < 1:
            raise ValueError("group_count must be greater than 0")
        if self.group_size < 1:
            raise ValueError("group_size must be greater than 0")
        if self.selected_count < 1:
            raise ValueError("selected_count must be greater than 0")
        if self.selected_count >= self.group_size:
            raise ValueError("selected_count must be less than group_size")


DEFAULT_TOURRANK_STAGE_CONFIGS: tuple[TourRankStageConfig, ...] = (
    TourRankStageConfig(group_count=5, group_size=20, selected_count=10),
    TourRankStageConfig(group_count=5, group_size=10, selected_count=4),
    TourRankStageConfig(group_count=1, group_size=20, selected_count=10),
    TourRankStageConfig(group_count=1, group_size=10, selected_count=5),
    TourRankStageConfig(group_count=1, group_size=5, selected_count=2),
)


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
        ranking = parse_ranking_response(raw_response, expected_count=len(documents))
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
class _TourRankConfigMixin:
    algorithm: TourRankAlgorithm = "tourrank_r"
    rounds: int = 2
    stage_configs: tuple[TourRankStageConfig, ...] = field(
        default_factory=lambda: DEFAULT_TOURRANK_STAGE_CONFIGS
    )
    shuffle_seed: int = 13
    group_parallelism: int | None = None
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.algorithm != "tourrank_r":
            raise ValueError('algorithm must be "tourrank_r"')
        if self.rounds < 1:
            raise ValueError("rounds must be greater than 0")
        if not self.stage_configs:
            raise ValueError("stage_configs must not be empty")
        if self.group_parallelism is not None and self.group_parallelism < 1:
            raise ValueError("group_parallelism must be greater than 0")
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

    def _validate_stage_pipeline(self, document_count: int) -> None:
        current_count = document_count
        for stage_index, stage in enumerate(self.stage_configs, start=1):
            required_count = stage.group_count * stage.group_size
            if current_count != required_count:
                raise RerankInputError(
                    "stage_configs do not match the candidate count: "
                    f"stage {stage_index} requires {required_count} documents, "
                    f"but received {current_count}."
                )
            current_count = stage.group_count * stage.selected_count

    def _build_stage_groups(
        self,
        current_order: Sequence[int],
        stage: TourRankStageConfig,
        *,
        round_index: int,
        stage_index: int,
    ) -> list[list[int]]:
        groups = [
            [
                current_order[group_index + offset * stage.group_count]
                for offset in range(stage.group_size)
            ]
            for group_index in range(stage.group_count)
        ]
        for group_index, group in enumerate(groups):
            seed = (
                self.shuffle_seed
                + round_index * 1_000_003
                + stage_index * 1_009
                + group_index
            )
            random.Random(seed).shuffle(group)
        return groups

    def _results_from_scores(
        self,
        documents: Sequence[Document],
        scores: Sequence[int],
        top_k: int | None,
    ) -> list[RerankResult]:
        if top_k is not None and top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-scores[index], index),
        )
        if top_k is not None:
            ordered_indexes = ordered_indexes[:top_k]
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "tourrank",
                    "algorithm": self.algorithm,
                    "rounds": self.rounds,
                    "score": scores[original_index],
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results


@dataclass(frozen=True)
class TourRankStrategy(_TourRankConfigMixin):
    group_parallelism: int = 1

    def __post_init__(self) -> None:
        if self.group_parallelism is None:
            raise ValueError("group_parallelism must be greater than 0")
        super().__post_init__()

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
        self._validate_stage_pipeline(len(documents))

        provider = _ensure_selection_provider(provider)
        scores = [0 for _ in documents]
        for round_index in range(self.rounds):
            current_order = list(range(len(documents)))
            for stage_index, stage in enumerate(self.stage_configs):
                current_order = self._run_selection_stage(
                    query=query,
                    documents=documents,
                    provider=provider,
                    current_order=current_order,
                    stage=stage,
                    round_index=round_index,
                    stage_index=stage_index,
                    scores=scores,
                )
        return self._results_from_scores(documents, scores, top_k)

    def _run_selection_stage(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: SelectionLLMProvider,
        current_order: Sequence[int],
        stage: TourRankStageConfig,
        round_index: int,
        stage_index: int,
        scores: list[int],
    ) -> list[int]:
        next_order: list[int] = []
        groups = self._build_stage_groups(
            current_order,
            stage,
            round_index=round_index,
            stage_index=stage_index,
        )
        if self.group_parallelism == 1 or len(groups) == 1:
            for group in groups:
                selected_indexes = self._select_group(
                    query=query,
                    documents=documents,
                    provider=provider,
                    group=group,
                    selected_count=stage.selected_count,
                )
                for original_index in selected_indexes:
                    scores[original_index] += 1
                next_order.extend(selected_indexes)
            return next_order

        with ThreadPoolExecutor(max_workers=self.group_parallelism) as executor:
            selected_groups = executor.map(
                lambda group: self._select_group(
                    query=query,
                    documents=documents,
                    provider=provider,
                    group=group,
                    selected_count=stage.selected_count,
                ),
                groups,
            )
            for selected_indexes in selected_groups:
                for original_index in selected_indexes:
                    scores[original_index] += 1
                next_order.extend(selected_indexes)
        return next_order

    def _select_group(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: SelectionLLMProvider,
        group: Sequence[int],
        selected_count: int,
    ) -> list[int]:
        group_documents = [documents[index] for index in group]
        raw_response = provider.select(query, group_documents, selected_count)
        selected = parse_selection_response(
            raw_response,
            expected_count=len(group_documents),
            selected_count=selected_count,
        )
        return [group[number - 1] for number in selected]


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
        ranking = parse_ranking_response(raw_response, expected_count=len(documents))
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


@dataclass(frozen=True)
class AsyncTourRankStrategy(_TourRankConfigMixin):
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
        self._validate_stage_pipeline(len(documents))

        provider = _ensure_async_selection_provider(provider)
        scores = [0 for _ in documents]
        for round_index in range(self.rounds):
            current_order = list(range(len(documents)))
            for stage_index, stage in enumerate(self.stage_configs):
                current_order = await self._run_selection_stage(
                    query=query,
                    documents=documents,
                    provider=provider,
                    current_order=current_order,
                    stage=stage,
                    round_index=round_index,
                    stage_index=stage_index,
                    scores=scores,
                )
        return self._results_from_scores(documents, scores, top_k)

    async def _run_selection_stage(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: AsyncSelectionLLMProvider,
        current_order: Sequence[int],
        stage: TourRankStageConfig,
        round_index: int,
        stage_index: int,
        scores: list[int],
    ) -> list[int]:
        groups = self._build_stage_groups(
            current_order,
            stage,
            round_index=round_index,
            stage_index=stage_index,
        )
        semaphore = (
            asyncio.Semaphore(self.group_parallelism)
            if self.group_parallelism is not None
            else None
        )
        raw_responses = await asyncio.gather(
            *(
                self._select_group(
                    query,
                    documents,
                    provider,
                    group,
                    stage.selected_count,
                    semaphore,
                )
                for group in groups
            )
        )
        next_order: list[int] = []
        for group, raw_response in zip(groups, raw_responses, strict=True):
            selected = parse_selection_response(
                raw_response,
                expected_count=len(group),
                selected_count=stage.selected_count,
            )
            selected_indexes = [group[number - 1] for number in selected]
            for original_index in selected_indexes:
                scores[original_index] += 1
            next_order.extend(selected_indexes)
        return next_order

    async def _select_group(
        self,
        query: str,
        documents: Sequence[Document],
        provider: AsyncSelectionLLMProvider,
        group: Sequence[int],
        selected_count: int,
        semaphore: asyncio.Semaphore | None,
    ) -> str:
        group_documents = [documents[index] for index in group]
        if semaphore is None:
            return await provider.select(query, group_documents, selected_count)
        async with semaphore:
            return await provider.select(query, group_documents, selected_count)


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


def _ensure_selection_provider(provider: object) -> SelectionLLMProvider:
    select = getattr(provider, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(SelectionLLMProvider, provider)


def _ensure_async_pairwise_provider(
    provider: object,
) -> AsyncPairwiseLLMProvider:
    compare = getattr(provider, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(AsyncPairwiseLLMProvider, provider)


def _ensure_async_selection_provider(
    provider: object,
) -> AsyncSelectionLLMProvider:
    select = getattr(provider, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(AsyncSelectionLLMProvider, provider)
