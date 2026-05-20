from __future__ import annotations

import asyncio

from ranksmith import (
    AsyncModelClient,
    Document,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RerankUsage,
)


class UsageProvider:
    def __init__(self, usage: RerankUsage | None) -> None:
        self.usage = usage

    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(content='{"ranking": [1]}', usage=self.usage)


class AsyncUsageProvider:
    def __init__(self, usage: RerankUsage | None) -> None:
        self.usage = usage

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(content='{"ranking": [1]}', usage=self.usage)


def test_model_client_emits_usage() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    client = ModelClient(provider=UsageProvider(usage), on_usage=captured.append)

    client.rank("query", [Document(text="alpha")])

    assert captured == [usage]


def test_model_client_skips_missing_usage() -> None:
    captured: list[RerankUsage] = []
    client = ModelClient(provider=UsageProvider(None), on_usage=captured.append)

    client.rank("query", [Document(text="alpha")])

    assert captured == []


def test_async_model_client_supports_sync_usage_callback() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client = AsyncModelClient(
        provider=AsyncUsageProvider(usage),
        on_usage=captured.append,
    )

    async def run() -> None:
        await client.rank("query", [Document(text="alpha")])

    asyncio.run(run())
    assert captured == [usage]


def test_async_model_client_awaits_usage_callback() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)

    async def callback(value: RerankUsage) -> None:
        captured.append(value)

    client = AsyncModelClient(
        provider=AsyncUsageProvider(usage),
        on_usage=callback,
    )

    async def run() -> None:
        await client.rank("query", [Document(text="alpha")])

    asyncio.run(run())
    assert captured == [usage]
