from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from ranksmith import Document, RerankUsage
from ranksmith._providers import (
    AsyncAzureOpenAIProvider,
    AzureOpenAIProvider,
    _build_pairwise_prompt,
    _emit_usage,
    _emit_usage_async,
    _extract_usage,
)


@dataclass
class _StubUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _StubResponse:
    usage: _StubUsage | None


@dataclass
class _StubMessage:
    content: str | None


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubCompletionResponse:
    usage: _StubUsage | None
    choices: list[_StubChoice]


class _StubCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _StubCompletionResponse:
        self.kwargs = kwargs
        return _StubCompletionResponse(
            usage=_StubUsage(11, 2, 13),
            choices=[_StubChoice(message=_StubMessage(content='{"winner": "A"}'))],
        )


class _StubAsyncCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _StubCompletionResponse:
        self.kwargs = kwargs
        return _StubCompletionResponse(
            usage=_StubUsage(17, 3, 20),
            choices=[_StubChoice(message=_StubMessage(content='{"winner": "B"}'))],
        )


@dataclass
class _StubChat:
    completions: _StubCompletions | _StubAsyncCompletions


@dataclass
class _StubClient:
    chat: _StubChat


def test_extract_usage_returns_none_when_missing() -> None:
    assert _extract_usage(_StubResponse(usage=None)) is None


def test_extract_usage_reads_token_fields() -> None:
    response = _StubResponse(usage=_StubUsage(10, 5, 15))
    assert _extract_usage(response) == RerankUsage(10, 5, 15)


def test_emit_usage_invokes_callback() -> None:
    captured: list[RerankUsage] = []
    _emit_usage(_StubResponse(usage=_StubUsage(3, 2, 5)), captured.append)
    assert captured == [RerankUsage(3, 2, 5)]


def test_emit_usage_skips_when_callback_is_none() -> None:
    _emit_usage(_StubResponse(usage=_StubUsage(3, 2, 5)), None)


def test_emit_usage_skips_when_usage_is_none() -> None:
    captured: list[RerankUsage] = []
    _emit_usage(_StubResponse(usage=None), captured.append)
    assert captured == []


def test_emit_usage_async_supports_sync_callback() -> None:
    captured: list[RerankUsage] = []

    async def run() -> None:
        await _emit_usage_async(
            _StubResponse(usage=_StubUsage(1, 1, 2)), captured.append
        )

    asyncio.run(run())
    assert captured == [RerankUsage(1, 1, 2)]


def test_emit_usage_async_awaits_coroutine_callback() -> None:
    captured: list[RerankUsage] = []

    async def callback(usage: RerankUsage) -> None:
        captured.append(usage)

    async def run() -> None:
        await _emit_usage_async(_StubResponse(usage=_StubUsage(7, 3, 10)), callback)

    asyncio.run(run())
    assert captured == [RerankUsage(7, 3, 10)]


def test_build_pairwise_prompt_requests_json_winner() -> None:
    prompt = _build_pairwise_prompt(
        "query",
        Document(text="alpha"),
        Document(text="beta"),
    )

    assert "Passage A:\nalpha" in prompt
    assert "Passage B:\nbeta" in prompt
    assert '{"winner": "A"}' in prompt


def test_azure_compare_uses_pairwise_prompt_and_emits_usage() -> None:
    completions = _StubCompletions()
    captured: list[RerankUsage] = []
    provider = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
    stubbed_provider = cast(Any, provider)
    stubbed_provider._azure_deployment = "deployment"
    stubbed_provider._client = _StubClient(chat=_StubChat(completions=completions))
    stubbed_provider._on_usage = captured.append

    result = provider.compare("query", Document(text="alpha"), Document(text="beta"))

    assert result == '{"winner": "A"}'
    assert captured == [RerankUsage(11, 2, 13)]
    assert completions.kwargs is not None
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["temperature"] == 0
    user_message = completions.kwargs["messages"][1]["content"]
    assert "Passage A:\nalpha" in user_message
    assert "Passage B:\nbeta" in user_message


def test_async_azure_compare_uses_pairwise_prompt_and_emits_usage() -> None:
    completions = _StubAsyncCompletions()
    captured: list[RerankUsage] = []
    provider = AsyncAzureOpenAIProvider.__new__(AsyncAzureOpenAIProvider)
    stubbed_provider = cast(Any, provider)
    stubbed_provider._azure_deployment = "deployment"
    stubbed_provider._client = _StubClient(chat=_StubChat(completions=completions))
    stubbed_provider._on_usage = captured.append

    async def run() -> str:
        return await provider.compare(
            "query",
            Document(text="alpha"),
            Document(text="beta"),
        )

    result = asyncio.run(run())

    assert result == '{"winner": "B"}'
    assert captured == [RerankUsage(17, 3, 20)]
    assert completions.kwargs is not None
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    user_message = completions.kwargs["messages"][1]["content"]
    assert "Passage A:\nalpha" in user_message
    assert "Passage B:\nbeta" in user_message
