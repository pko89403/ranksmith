from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from openai import AsyncAzureOpenAI, AzureOpenAI

from ranksmith.errors import RerankProviderError
from ranksmith.types import Document, RerankUsage

UsageCallback = Callable[[RerankUsage], None]
AsyncUsageCallback = Callable[[RerankUsage], Awaitable[None] | None]


class LLMProvider(Protocol):
    def rank(self, query: str, documents: list[Document]) -> str:
        """Return a JSON string containing a 1-based ranking permutation."""


class PairwiseLLMProvider(Protocol):
    def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        """Return a JSON string containing a pairwise winner, "A" or "B"."""


class AsyncLLMProvider(Protocol):
    async def rank(self, query: str, documents: list[Document]) -> str:
        """Return a JSON string containing a 1-based ranking asynchronously."""


class AsyncPairwiseLLMProvider(Protocol):
    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        """Return a JSON string containing a pairwise winner asynchronously."""


class AzureOpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        timeout: float | None = None,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._azure_deployment = azure_deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )
        self._on_usage = on_usage

    def rank(self, query: str, documents: list[Document]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reranking engine. Return only JSON with "
                            'a "ranking" array. The ranking must be a permutation '
                            "of the candidate numbers."
                        ),
                    },
                    {"role": "user", "content": _build_prompt(query, documents)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        _emit_usage(response, self._on_usage)
        content = response.choices[0].message.content
        if content is None:
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return content

    def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a pairwise reranking engine. Return only JSON "
                            'with a "winner" value of "A" or "B".'
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_pairwise_prompt(
                            query, document_a, document_b
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        _emit_usage(response, self._on_usage)
        content = response.choices[0].message.content
        if content is None:
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return content


class AsyncAzureOpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        timeout: float | None = None,
        on_usage: AsyncUsageCallback | None = None,
    ) -> None:
        self._azure_deployment = azure_deployment
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )
        self._on_usage = on_usage

    async def rank(self, query: str, documents: list[Document]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reranking engine. Return only JSON with "
                            'a "ranking" array. The ranking must be a permutation '
                            "of the candidate numbers."
                        ),
                    },
                    {"role": "user", "content": _build_prompt(query, documents)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        await _emit_usage_async(response, self._on_usage)
        content = response.choices[0].message.content
        if content is None:
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return content

    async def compare(
        self,
        query: str,
        document_a: Document,
        document_b: Document,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a pairwise reranking engine. Return only JSON "
                            'with a "winner" value of "A" or "B".'
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_pairwise_prompt(
                            query, document_a, document_b
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        await _emit_usage_async(response, self._on_usage)
        content = response.choices[0].message.content
        if content is None:
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return content


def _extract_usage(response: object) -> RerankUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return RerankUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _emit_usage(response: object, callback: UsageCallback | None) -> None:
    if callback is None:
        return
    usage = _extract_usage(response)
    if usage is not None:
        callback(usage)


async def _emit_usage_async(
    response: object, callback: AsyncUsageCallback | None
) -> None:
    if callback is None:
        return
    usage = _extract_usage(response)
    if usage is None:
        return
    result = callback(usage)
    if isinstance(result, Awaitable):
        await result


def _build_prompt(query: str, documents: list[Document]) -> str:
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
        '{"ranking": [1, 2, 3]}\n'
        "Use each candidate number exactly once."
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
