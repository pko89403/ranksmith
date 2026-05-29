from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, TypeVar, cast

import trueskill  # type: ignore[import-untyped]

from ranksmith.errors import DocumentTooLongError, RerankInputError, RerankParseError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_ranking_response, parse_selection_response
from ranksmith.types import Document, RerankResult

Algorithm = Literal["rankgpt_sliding_window"]
PairwiseAlgorithm = Literal["prp_sliding_k"]
TourRankAlgorithm = Literal["tourrank_r"]
AcuRankAlgorithm = Literal["acurank"]
_T = TypeVar("_T")


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
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = _ensure_listwise_provider(model_client)
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
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
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = _ensure_pairwise_provider(model_client)
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
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

                first = _parse_pairwise_winner(
                    model_client.compare(
                        query,
                        documents[left_index],
                        documents[right_index],
                    )
                )
                second = _parse_pairwise_winner(
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
class _AcuRankConfigMixin:
    algorithm: AcuRankAlgorithm = "acurank"
    target_rank: int = 10
    window_size: int = 20
    tolerance: float = 0.01
    uncertain_threshold: int = 10
    max_adaptive_reranker_calls: int | None = None
    batch_parallelism: int = 1
    initial_pass: bool = True
    score_metadata_key: str = "score"
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.algorithm != "acurank":
            raise ValueError('algorithm must be "acurank"')
        if self.target_rank < 1:
            raise ValueError("target_rank must be greater than 0")
        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0")
        if self.tolerance <= 0 or self.tolerance >= 0.5:
            raise ValueError("tolerance must be greater than 0 and less than 0.5")
        if self.uncertain_threshold < 1:
            raise ValueError("uncertain_threshold must be greater than 0")
        if (
            self.max_adaptive_reranker_calls is not None
            and self.max_adaptive_reranker_calls < 0
        ):
            raise ValueError(
                "max_adaptive_reranker_calls must be greater than or equal to 0"
            )
        if self.batch_parallelism < 1:
            raise ValueError("batch_parallelism must be greater than 0")
        if self.score_metadata_key == "":
            raise ValueError("score_metadata_key must not be empty")
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

    def _initialize_ratings(
        self,
        documents: Sequence[Document],
    ) -> list[trueskill.Rating]:
        has_scores = [
            self.score_metadata_key in document.metadata for document in documents
        ]
        if any(has_scores) and not all(has_scores):
            raise RerankInputError(
                f"score metadata key {self.score_metadata_key!r} must be present "
                "for every document or omitted for every document."
            )
        if not any(has_scores):
            return [trueskill.Rating() for _ in documents]

        ratings: list[trueskill.Rating] = []
        for index, document in enumerate(documents):
            score = document.metadata[self.score_metadata_key]
            if isinstance(score, bool) or not isinstance(score, Real):
                raise RerankInputError(
                    f"score metadata at index {index} must be numeric."
                )
            score_value = float(score)
            if score_value <= 0:
                raise RerankInputError(
                    f"score metadata at index {index} must be greater than 0."
                )
            ratings.append(trueskill.Rating(mu=score_value, sigma=score_value / 3))
        return ratings

    def _results_from_ratings(
        self,
        documents: Sequence[Document],
        ratings: Sequence[trueskill.Rating],
        probabilities: Sequence[float],
        reranker_calls: int,
        top_k: int | None,
    ) -> list[RerankResult]:
        if top_k is not None and top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-ratings[index].mu, index),
        )
        if top_k is not None:
            ordered_indexes = ordered_indexes[:top_k]
        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "acurank",
                    "algorithm": self.algorithm,
                    "mu": ratings[original_index].mu,
                    "sigma": ratings[original_index].sigma,
                    "top_k_probability": probabilities[original_index],
                    "reranker_calls": reranker_calls,
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]


@dataclass(frozen=True)
class AcuRankStrategy(_AcuRankConfigMixin):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []
        target_rank = min(self.target_rank, len(documents))

        model_client = _ensure_listwise_provider(model_client)
        ratings = self._initialize_ratings(documents)
        reranker_calls = 0
        adaptive_reranker_calls = 0

        if self.initial_pass:
            reranker_calls += _rank_and_apply_acurank_batches(
                query=query,
                documents=documents,
                model_client=model_client,
                batches=_chunks(range(len(documents)), self.window_size),
                ratings=ratings,
                batch_parallelism=self.batch_parallelism,
            )

        probabilities = _acurank_topk_probabilities(ratings, target_rank)
        while _has_call_budget(
            adaptive_reranker_calls, self.max_adaptive_reranker_calls
        ):
            probability_order = _acurank_probability_order(probabilities)
            uncertain = [
                index
                for index in probability_order
                if self.tolerance < probabilities[index] < 1 - self.tolerance
            ]
            final_iteration = False
            if len(uncertain) < self.uncertain_threshold:
                uncertain = [
                    index
                    for index in probability_order
                    if probabilities[index] > self.tolerance
                ]
                final_iteration = True
            updated = False
            batches = _chunks(uncertain, self.window_size)
            if self.max_adaptive_reranker_calls is not None:
                remaining_calls = (
                    self.max_adaptive_reranker_calls - adaptive_reranker_calls
                )
                batches = batches[:remaining_calls]
            calls = _rank_and_apply_acurank_batches(
                query=query,
                documents=documents,
                model_client=model_client,
                batches=batches,
                ratings=ratings,
                batch_parallelism=self.batch_parallelism,
            )
            reranker_calls += calls
            adaptive_reranker_calls += calls
            updated = calls > 0
            probabilities = _acurank_topk_probabilities(ratings, target_rank)
            if final_iteration or not updated:
                break

        return self._results_from_ratings(
            documents,
            ratings,
            probabilities,
            reranker_calls,
            top_k,
        )


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
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []
        self._validate_stage_pipeline(len(documents))

        model_client = _ensure_selection_provider(model_client)
        scores = [0 for _ in documents]
        for round_index in range(self.rounds):
            current_order = list(range(len(documents)))
            for stage_index, stage in enumerate(self.stage_configs):
                current_order = self._run_selection_stage(
                    query=query,
                    documents=documents,
                    model_client=model_client,
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
        model_client: ModelClient,
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
                    model_client=model_client,
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
                    model_client=model_client,
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
        model_client: ModelClient,
        group: Sequence[int],
        selected_count: int,
    ) -> list[int]:
        group_documents = [documents[index] for index in group]
        raw_response = model_client.select(query, group_documents, selected_count)
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
        model_client: AsyncModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = _ensure_async_listwise_provider(model_client)
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
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
        self._validate_documents(documents)
        if not documents:
            return []

        model_client = _ensure_async_pairwise_provider(model_client)
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
        if top_k < 0:
            raise RerankInputError("top_k must be greater than or equal to 0")
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


@dataclass(frozen=True)
class AsyncTourRankStrategy(_TourRankConfigMixin):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []
        self._validate_stage_pipeline(len(documents))

        model_client = _ensure_async_selection_provider(model_client)
        scores = [0 for _ in documents]
        for round_index in range(self.rounds):
            current_order = list(range(len(documents)))
            for stage_index, stage in enumerate(self.stage_configs):
                current_order = await self._run_selection_stage(
                    query=query,
                    documents=documents,
                    model_client=model_client,
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
        model_client: AsyncModelClient,
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
                    model_client,
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
        model_client: AsyncModelClient,
        group: Sequence[int],
        selected_count: int,
        semaphore: asyncio.Semaphore | None,
    ) -> str:
        group_documents = [documents[index] for index in group]
        if semaphore is None:
            return await model_client.select(query, group_documents, selected_count)
        async with semaphore:
            return await model_client.select(query, group_documents, selected_count)


@dataclass(frozen=True)
class AsyncAcuRankStrategy(_AcuRankConfigMixin):
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self._validate_documents(documents)
        if not documents:
            return []
        target_rank = min(self.target_rank, len(documents))

        model_client = _ensure_async_listwise_provider(model_client)
        ratings = self._initialize_ratings(documents)
        reranker_calls = 0
        adaptive_reranker_calls = 0

        if self.initial_pass:
            reranker_calls += await _rank_and_apply_acurank_batches_async(
                query=query,
                documents=documents,
                model_client=model_client,
                batches=_chunks(range(len(documents)), self.window_size),
                ratings=ratings,
                batch_parallelism=self.batch_parallelism,
            )

        probabilities = _acurank_topk_probabilities(ratings, target_rank)
        while _has_call_budget(
            adaptive_reranker_calls, self.max_adaptive_reranker_calls
        ):
            probability_order = _acurank_probability_order(probabilities)
            uncertain = [
                index
                for index in probability_order
                if self.tolerance < probabilities[index] < 1 - self.tolerance
            ]
            final_iteration = False
            if len(uncertain) < self.uncertain_threshold:
                uncertain = [
                    index
                    for index in probability_order
                    if probabilities[index] > self.tolerance
                ]
                final_iteration = True
            updated = False
            batches = _chunks(uncertain, self.window_size)
            if self.max_adaptive_reranker_calls is not None:
                remaining_calls = (
                    self.max_adaptive_reranker_calls - adaptive_reranker_calls
                )
                batches = batches[:remaining_calls]
            calls = await _rank_and_apply_acurank_batches_async(
                query=query,
                documents=documents,
                model_client=model_client,
                batches=batches,
                ratings=ratings,
                batch_parallelism=self.batch_parallelism,
            )
            reranker_calls += calls
            adaptive_reranker_calls += calls
            updated = calls > 0
            probabilities = _acurank_topk_probabilities(ratings, target_rank)
            if final_iteration or not updated:
                break

        return self._results_from_ratings(
            documents,
            ratings,
            probabilities,
            reranker_calls,
            top_k,
        )


@dataclass(frozen=True)
class _AcuRankBatchRanking:
    batch: tuple[int, ...]
    ranking: tuple[int, ...]


def _chunks(indexes: Sequence[_T], size: int) -> list[list[_T]]:
    return [
        list(indexes[start : start + size]) for start in range(0, len(indexes), size)
    ]


def _has_call_budget(current_calls: int, max_calls: int | None) -> bool:
    return max_calls is None or current_calls < max_calls


def _acurank_probability_order(probabilities: Sequence[float]) -> list[int]:
    return sorted(
        range(len(probabilities)),
        key=lambda index: (-probabilities[index], index),
    )


def _rank_and_apply_acurank_batches(
    *,
    query: str,
    documents: Sequence[Document],
    model_client: ModelClient,
    batches: Sequence[Sequence[int]],
    ratings: list[trueskill.Rating],
    batch_parallelism: int,
) -> int:
    if batch_parallelism == 1 or len(batches) <= 1:
        ranked_batches = [
            _rank_acurank_batch(
                query=query,
                documents=documents,
                model_client=model_client,
                batch=batch,
            )
            for batch in batches
        ]
    else:
        with ThreadPoolExecutor(max_workers=batch_parallelism) as executor:
            ranked_batches = list(
                executor.map(
                    lambda batch: _rank_acurank_batch(
                        query=query,
                        documents=documents,
                        model_client=model_client,
                        batch=batch,
                    ),
                    batches,
                )
            )
    calls = 0
    for ranked_batch in ranked_batches:
        if ranked_batch is None:
            continue
        _apply_acurank_batch_ranking(ranked_batch, ratings)
        calls += 1
    return calls


async def _rank_and_apply_acurank_batches_async(
    *,
    query: str,
    documents: Sequence[Document],
    model_client: AsyncModelClient,
    batches: Sequence[Sequence[int]],
    ratings: list[trueskill.Rating],
    batch_parallelism: int,
) -> int:
    calls = 0
    for batch_group in _chunks(batches, batch_parallelism):
        ranked_batches = await asyncio.gather(
            *[
                _rank_acurank_batch_async(
                    query=query,
                    documents=documents,
                    model_client=model_client,
                    batch=batch,
                )
                for batch in batch_group
            ]
        )
        for ranked_batch in ranked_batches:
            if ranked_batch is None:
                continue
            _apply_acurank_batch_ranking(ranked_batch, ratings)
            calls += 1
    return calls


def _rank_acurank_batch(
    *,
    query: str,
    documents: Sequence[Document],
    model_client: ModelClient,
    batch: Sequence[int],
) -> _AcuRankBatchRanking | None:
    if len(batch) <= 1:
        return None
    batch_documents = [documents[index] for index in batch]
    raw_response = model_client.rank(query, batch_documents)
    ranking = parse_ranking_response(raw_response, expected_count=len(batch_documents))
    return _AcuRankBatchRanking(tuple(batch), tuple(ranking))


async def _rank_acurank_batch_async(
    *,
    query: str,
    documents: Sequence[Document],
    model_client: AsyncModelClient,
    batch: Sequence[int],
) -> _AcuRankBatchRanking | None:
    if len(batch) <= 1:
        return None
    batch_documents = [documents[index] for index in batch]
    raw_response = await model_client.rank(query, batch_documents)
    ranking = parse_ranking_response(raw_response, expected_count=len(batch_documents))
    return _AcuRankBatchRanking(tuple(batch), tuple(ranking))


def _apply_acurank_batch_ranking(
    ranked_batch: _AcuRankBatchRanking,
    ratings: list[trueskill.Rating],
) -> None:
    _update_acurank_ratings(ranked_batch.batch, ranked_batch.ranking, ratings)


def _update_acurank_ratings(
    batch: Sequence[int],
    ranking: Sequence[int],
    ratings: list[trueskill.Rating],
) -> None:
    ordered_indexes = [batch[number - 1] for number in ranking]
    updated = trueskill.rate(
        [[ratings[index]] for index in ordered_indexes],
        ranks=range(len(ordered_indexes)),
    )
    for original_index, updated_group in zip(ordered_indexes, updated, strict=True):
        ratings[original_index] = updated_group[0]


def _acurank_topk_probabilities(
    ratings: Sequence[trueskill.Rating],
    target_rank: int,
) -> list[float]:
    threshold = _find_acurank_threshold(ratings, target_rank)
    return [
        _normal_right_tail(threshold, rating.mu, rating.sigma) for rating in ratings
    ]


def _find_acurank_threshold(
    ratings: Sequence[trueskill.Rating],
    target_rank: int,
) -> float:
    max_sigma = max(float(rating.sigma) for rating in ratings)
    left = min(float(rating.mu) for rating in ratings) - 8 * max_sigma
    right = max(float(rating.mu) for rating in ratings) + 8 * max_sigma
    for _ in range(100):
        mid = (left + right) / 2
        expected_above = sum(
            _normal_right_tail(mid, float(rating.mu), float(rating.sigma))
            for rating in ratings
        )
        if expected_above > target_rank:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def _normal_right_tail(threshold: float, mu: float, sigma: float) -> float:
    z = (threshold - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


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


def _ensure_listwise_provider(provider: object) -> ModelClient:
    rank = getattr(provider, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(ModelClient, provider)


def _ensure_async_listwise_provider(provider: object) -> AsyncModelClient:
    rank = getattr(provider, "rank", None)
    if not callable(rank):
        raise RerankInputError("provider must support listwise rank()")
    return cast(AsyncModelClient, provider)


def _ensure_pairwise_provider(provider: object) -> ModelClient:
    compare = getattr(provider, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(ModelClient, provider)


def _ensure_selection_provider(provider: object) -> ModelClient:
    select = getattr(provider, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(ModelClient, provider)


def _ensure_async_pairwise_provider(
    provider: object,
) -> AsyncModelClient:
    compare = getattr(provider, "compare", None)
    if not callable(compare):
        raise RerankInputError("provider must support pairwise compare()")
    return cast(AsyncModelClient, provider)


def _ensure_async_selection_provider(
    provider: object,
) -> AsyncModelClient:
    select = getattr(provider, "select", None)
    if not callable(select):
        raise RerankInputError("provider must support selection select()")
    return cast(AsyncModelClient, provider)
