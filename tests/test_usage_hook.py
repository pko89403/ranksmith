from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ranksmith import RerankUsage
from ranksmith._providers import _emit_usage, _emit_usage_async, _extract_usage


@dataclass
class _StubUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _StubResponse:
    usage: _StubUsage | None


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
