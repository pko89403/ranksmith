from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeVar

import trueskill  # type: ignore[import-untyped]

from ranksmith.errors import DocumentTooLongError, RerankInputError
from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_ranking_response
from ranksmith.types import Document, RerankResult

from ._common import (
    ensure_async_listwise_model_client,
    ensure_listwise_model_client,
    validate_top_k,
)

AcuRankAlgorithm = Literal["acurank"]
_T = TypeVar("_T")


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
        validate_top_k(top_k)
        self._validate_documents(documents)
        if not documents:
            return []
        target_rank = min(self.target_rank, len(documents))

        model_client = ensure_listwise_model_client(model_client)
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
class AsyncAcuRankStrategy(_AcuRankConfigMixin):
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
        target_rank = min(self.target_rank, len(documents))

        model_client = ensure_async_listwise_model_client(model_client)
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
