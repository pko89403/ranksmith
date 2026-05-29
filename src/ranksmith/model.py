from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ranksmith.errors import RerankProviderError
from ranksmith.types import Document, RerankUsage

__all__ = [
    "AsyncModelClient",
    "AsyncModelProvider",
    "AsyncUsageCallback",
    "ModelClient",
    "ModelMessage",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "UsageCallback",
]

UsageCallback = Callable[[RerankUsage], None]
AsyncUsageCallback = Callable[[RerankUsage], Awaitable[None] | None]


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[ModelMessage]
    response_format: Literal["json_object"] = "json_object"
    temperature: float = 0


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: RerankUsage | None = None


class ModelProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a vendor-backed model response for a JSON completion request."""


class AsyncModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a vendor-backed model response asynchronously."""


class ModelClient:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._provider = provider
        self._on_usage = on_usage

    def rank(self, query: str, documents: list[Document]) -> str:
        candidate_count = len(documents)
        return self._complete(
            system=(
                "You are a reranking engine. Return only JSON with "
                'a "ranking" array. The ranking must be a permutation '
                "of the candidate numbers. "
                f"The ranking array must contain exactly {candidate_count} integers: "
                f"each integer from 1 to {candidate_count} exactly once."
            ),
            user=_build_prompt(query, documents),
        )

    def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        return self._complete(
            system=(
                "You are a pairwise reranking engine. Return only JSON "
                'with a "winner" value of "A" or "B".'
            ),
            user=_build_pairwise_prompt(query, document_a, document_b),
        )

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        candidate_count = len(documents)
        return self._complete(
            system=(
                "You are a tournament reranking engine. Return only JSON "
                'with a "selected" array of candidate numbers. '
                f"The selected array must contain exactly {top_m} integers from "
                f"1 to {candidate_count}, without duplicates."
            ),
            user=_build_selection_prompt(query, documents, top_m),
        )

    def _complete(self, *, system: str, user: str) -> str:
        try:
            response = self._provider.complete(
                ModelRequest(
                    messages=[
                        ModelMessage(role="system", content=system),
                        ModelMessage(role="user", content=user),
                    ]
                )
            )
        except RerankProviderError:
            raise
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        _emit_usage(response.usage, self._on_usage)
        if response.content == "":
            raise RerankProviderError("Model provider returned an empty response.")
        return response.content


class AsyncModelClient:
    def __init__(
        self,
        *,
        provider: AsyncModelProvider,
        on_usage: AsyncUsageCallback | None = None,
    ) -> None:
        self._provider = provider
        self._on_usage = on_usage

    async def rank(self, query: str, documents: list[Document]) -> str:
        candidate_count = len(documents)
        return await self._complete(
            system=(
                "You are a reranking engine. Return only JSON with "
                'a "ranking" array. The ranking must be a permutation '
                "of the candidate numbers. "
                f"The ranking array must contain exactly {candidate_count} integers: "
                f"each integer from 1 to {candidate_count} exactly once."
            ),
            user=_build_prompt(query, documents),
        )

    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        return await self._complete(
            system=(
                "You are a pairwise reranking engine. Return only JSON "
                'with a "winner" value of "A" or "B".'
            ),
            user=_build_pairwise_prompt(query, document_a, document_b),
        )

    async def select(self, query: str, documents: list[Document], top_m: int) -> str:
        candidate_count = len(documents)
        return await self._complete(
            system=(
                "You are a tournament reranking engine. Return only JSON "
                'with a "selected" array of candidate numbers. '
                f"The selected array must contain exactly {top_m} integers from "
                f"1 to {candidate_count}, without duplicates."
            ),
            user=_build_selection_prompt(query, documents, top_m),
        )

    async def _complete(self, *, system: str, user: str) -> str:
        try:
            response = await self._provider.complete(
                ModelRequest(
                    messages=[
                        ModelMessage(role="system", content=system),
                        ModelMessage(role="user", content=user),
                    ]
                )
            )
        except RerankProviderError:
            raise
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        await _emit_usage_async(response.usage, self._on_usage)
        if response.content == "":
            raise RerankProviderError("Model provider returned an empty response.")
        return response.content


def _emit_usage(usage: RerankUsage | None, callback: UsageCallback | None) -> None:
    if usage is not None and callback is not None:
        callback(usage)


async def _emit_usage_async(
    usage: RerankUsage | None,
    callback: AsyncUsageCallback | None,
) -> None:
    if usage is None or callback is None:
        return
    result = callback(usage)
    if isinstance(result, Awaitable):
        await result


def _build_prompt(query: str, documents: list[Document]) -> str:
    candidate_count = len(documents)
    ranking_example = ", ".join(str(index) for index in range(1, candidate_count + 1))
    candidates = "\n\n".join(
        [
            f"[{index}]\n{document.text}"
            for index, document in enumerate(documents, start=1)
        ]
    )
    return (
        "Rank the candidate documents by relevance to the query.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidate documents:\n{candidates}\n\n"
        "Return JSON exactly like this shape:\n"
        f'{{"ranking": [{ranking_example}]}}\n'
        f"Return exactly {candidate_count} candidate numbers. "
        f"Use every integer from 1 to {candidate_count} exactly once. "
        "Do not omit, duplicate, or invent candidate numbers. "
        f"Before returning, verify that sorting your ranking gives "
        f"[{ranking_example}]. "
        "Do not include any explanation."
    )


def _build_pairwise_prompt(
    query: str,
    document_a: Document,
    document_b: Document,
) -> str:
    return (
        "Given a query, choose which passage is more relevant to the query.\n\n"
        f"Query:\n{query}\n\n"
        f"Passage A:\n{document_a.text}\n\n"
        f"Passage B:\n{document_b.text}\n\n"
        "Return JSON exactly like this shape:\n"
        '{"winner": "A"}\n\n'
        'Use "A" if Passage A is more relevant. '
        'Use "B" if Passage B is more relevant.'
    )


def _build_selection_prompt(
    query: str,
    documents: list[Document],
    top_m: int,
) -> str:
    example = ", ".join(str(index) for index in range(1, top_m + 1))
    candidates = "\n\n".join(
        [
            f"[{index}]\n{document.text}"
            for index, document in enumerate(documents, start=1)
        ]
    )
    return (
        "Given a query and candidate documents, select the most relevant "
        f"{top_m} candidate documents.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidate documents:\n{candidates}\n\n"
        "Return JSON exactly like this shape:\n"
        f'{{"selected": [{example}]}}\n'
        f"Return exactly {top_m} candidate numbers, without duplicates. "
        "Do not include any explanation."
    )
