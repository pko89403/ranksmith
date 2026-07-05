from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_selection_response
from ranksmith.types import Document, RerankResult

from ._common import (
    ensure_async_selection_model_client,
    ensure_selection_model_client,
    validate_documents_max_chars,
    validate_top_k,
)


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
class _TourRankConfigMixin:
    rounds: int = 2
    stage_configs: tuple[TourRankStageConfig, ...] = field(
        default_factory=lambda: DEFAULT_TOURRANK_STAGE_CONFIGS
    )
    shuffle_seed: int = 13
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be greater than 0")
        if not self.stage_configs:
            raise ValueError("stage_configs must not be empty")
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )

    def _validate_stage_pipeline(self, document_count: int) -> None:
        current_count = document_count
        for stage_index, stage in enumerate(self.stage_configs, start=1):
            required_count = stage.group_count * stage.group_size
            if current_count != required_count:
                from ranksmith.errors import RerankInputError

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
                    "algorithm": "tourrank_r",
                    "rounds": self.rounds,
                    "score": scores[original_index],
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results


@dataclass(frozen=True)
class TourRankStrategy(_TourRankConfigMixin):
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
        self._validate_stage_pipeline(len(documents))

        model_client = ensure_selection_model_client(model_client)
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
class AsyncTourRankStrategy(_TourRankConfigMixin):
    group_parallelism: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_parallelism is not None and self.group_parallelism < 1:
            raise ValueError("group_parallelism must be greater than 0")

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
        self._validate_stage_pipeline(len(documents))

        model_client = ensure_async_selection_model_client(model_client)
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
