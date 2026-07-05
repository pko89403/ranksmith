from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ranksmith.model import AsyncModelClient, ModelClient
from ranksmith.parsing import parse_selection_response
from ranksmith.types import Document, RerankResult

from .common import (
    ensure_capability,
    validate_documents_max_chars,
    validate_top_k,
)


@dataclass(frozen=True)
class _SetwiseConfigMixin:
    set_size: int = 3
    max_document_chars: int = 4000

    def __post_init__(self) -> None:
        if self.set_size < 3:
            raise ValueError("set_size must be greater than or equal to 3")
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")

    @property
    def _child_arity(self) -> int:
        return self.set_size - 1

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )

    def _results_from_indexes(
        self,
        documents: Sequence[Document],
        ordered_indexes: Sequence[int],
    ) -> list[RerankResult]:
        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "setwise",
                    "algorithm": "setwise_heapsort",
                    "set_size": self.set_size,
                },
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]


@dataclass(frozen=True)
class SetwiseStrategy(_SetwiseConfigMixin):
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
        if not documents or top_k == 0:
            return []

        model_client = ensure_capability(model_client, "selection", "select")
        ordered_indexes = self._rank_setwise_heapsort(
            query=query,
            documents=documents,
            model_client=model_client,
            top_k=top_k,
        )
        return self._results_from_indexes(documents, ordered_indexes)

    def _rank_setwise_heapsort(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        top_k: int | None,
    ) -> list[int]:
        heap = list(range(len(documents)))
        heap_size = len(heap)
        child_arity = self._child_arity
        _build_heap(
            heap=heap,
            heap_size=heap_size,
            child_arity=child_arity,
            heapify=lambda root: self._heapify(
                query=query,
                documents=documents,
                model_client=model_client,
                heap=heap,
                root=root,
                heap_size=heap_size,
            ),
        )

        limit = _extraction_limit(heap_size, top_k)
        ordered_indexes: list[int] = []
        for extract_index in range(limit):
            ordered_indexes.append(heap[0])
            heap[0], heap[heap_size - 1] = heap[heap_size - 1], heap[0]
            heap_size -= 1
            if extract_index < limit - 1 and heap_size > 1:
                self._heapify(
                    query=query,
                    documents=documents,
                    model_client=model_client,
                    heap=heap,
                    root=0,
                    heap_size=heap_size,
                )
        return ordered_indexes

    def _heapify(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        heap: list[int],
        root: int,
        heap_size: int,
    ) -> None:
        child_arity = self._child_arity
        current = root
        while True:
            positions = _candidate_positions(
                root=current,
                heap_size=heap_size,
                child_arity=child_arity,
            )
            if len(positions) == 1:
                return

            candidate_documents = [documents[heap[position]] for position in positions]
            raw_response = model_client.select(query, candidate_documents, 1)
            selected = parse_selection_response(
                raw_response,
                expected_count=len(candidate_documents),
                selected_count=1,
            )[0]
            selected_position = positions[selected - 1]
            if selected_position == current:
                return
            heap[current], heap[selected_position] = (
                heap[selected_position],
                heap[current],
            )
            current = selected_position


@dataclass(frozen=True)
class AsyncSetwiseStrategy(_SetwiseConfigMixin):
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
        if not documents or top_k == 0:
            return []

        model_client = ensure_capability(model_client, "selection", "select")
        ordered_indexes = await self._rank_setwise_heapsort(
            query=query,
            documents=documents,
            model_client=model_client,
            top_k=top_k,
        )
        return self._results_from_indexes(documents, ordered_indexes)

    async def _rank_setwise_heapsort(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        top_k: int | None,
    ) -> list[int]:
        heap = list(range(len(documents)))
        heap_size = len(heap)
        child_arity = self._child_arity
        # Heap mutations are step-dependent; async supports non-blocking I/O,
        # not intra-heap parallelism.
        await _build_heap_async(
            heap=heap,
            heap_size=heap_size,
            child_arity=child_arity,
            heapify=lambda root: self._heapify(
                query=query,
                documents=documents,
                model_client=model_client,
                heap=heap,
                root=root,
                heap_size=heap_size,
            ),
        )

        limit = _extraction_limit(heap_size, top_k)
        ordered_indexes: list[int] = []
        for extract_index in range(limit):
            ordered_indexes.append(heap[0])
            heap[0], heap[heap_size - 1] = heap[heap_size - 1], heap[0]
            heap_size -= 1
            if extract_index < limit - 1 and heap_size > 1:
                await self._heapify(
                    query=query,
                    documents=documents,
                    model_client=model_client,
                    heap=heap,
                    root=0,
                    heap_size=heap_size,
                )
        return ordered_indexes

    async def _heapify(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: AsyncModelClient,
        heap: list[int],
        root: int,
        heap_size: int,
    ) -> None:
        child_arity = self._child_arity
        current = root
        while True:
            positions = _candidate_positions(
                root=current,
                heap_size=heap_size,
                child_arity=child_arity,
            )
            if len(positions) == 1:
                return

            candidate_documents = [documents[heap[position]] for position in positions]
            raw_response = await model_client.select(query, candidate_documents, 1)
            selected = parse_selection_response(
                raw_response,
                expected_count=len(candidate_documents),
                selected_count=1,
            )[0]
            selected_position = positions[selected - 1]
            if selected_position == current:
                return
            heap[current], heap[selected_position] = (
                heap[selected_position],
                heap[current],
            )
            current = selected_position


def _extraction_limit(heap_size: int, top_k: int | None) -> int:
    return heap_size if top_k is None else min(top_k, heap_size)


def _build_heap(
    *,
    heap: list[int],
    heap_size: int,
    child_arity: int,
    heapify: Callable[[int], None],
) -> None:
    if heap_size < 2:
        return
    last_parent = (heap_size - 2) // child_arity
    for root in range(last_parent, -1, -1):
        heapify(root)


async def _build_heap_async(
    *,
    heap: list[int],
    heap_size: int,
    child_arity: int,
    heapify: Callable[[int], Awaitable[None]],
) -> None:
    if heap_size < 2:
        return
    last_parent = (heap_size - 2) // child_arity
    for root in range(last_parent, -1, -1):
        await heapify(root)


def _candidate_positions(
    *,
    root: int,
    heap_size: int,
    child_arity: int,
) -> list[int]:
    positions = [root]
    first_child = root * child_arity + 1
    for offset in range(child_arity):
        child = first_child + offset
        if child < heap_size:
            positions.append(child)
    return positions
