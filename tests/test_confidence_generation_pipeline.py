from __future__ import annotations

import pytest

from ranksmith.confidence_generation._pipeline import _call_provider
from ranksmith.confidence_generation._prompts import (
    build_answer_prompt,
    build_relevance_prompt,
)
from ranksmith.confidence_generation._types import (
    AnswerGenerationSample,
    RelevanceGenerationSample,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelMessage, ModelRequest, ModelResponse
from ranksmith.types import RerankUsage


class RecordingProvider:
    def __init__(self, outputs: list[ModelResponse]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outputs.pop(0)


class ProviderErrorProvider:
    def __init__(self, error: RerankProviderError) -> None:
        self.error = error

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise self.error


class BrokenProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("boom")


def test_answer_prompt_includes_configured_no_answer_value() -> None:
    prompt = build_answer_prompt(
        AnswerGenerationSample(
            id="a1",
            query="Who played Karen?",
            context="Nancy Travis played Karen.",
            gold_answer="Nancy Travis",
        ),
        no_answer_value="NO_CONTEXT_ANSWER",
    )

    assert "Question:\nWho played Karen?" in prompt
    assert "Context:\nNancy Travis played Karen." in prompt
    assert '{"answer":"..."}' in prompt
    assert '{"answer":"NO_CONTEXT_ANSWER"}' in prompt


def test_relevance_prompt_includes_relevant_not_relevant_json_contract() -> None:
    prompt = build_relevance_prompt(
        RelevanceGenerationSample(
            id="j1",
            query="Who played Karen?",
            document="Nancy Travis played Karen.",
            relevance_label=1,
        )
    )

    assert "Query:\nWho played Karen?" in prompt
    assert "Document:\nNancy Travis played Karen." in prompt
    assert '{"judgment":"relevant"}' in prompt
    assert '{"judgment":"not_relevant"}' in prompt


def test_call_provider_uses_json_request_and_emits_usage() -> None:
    usage = RerankUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    provider = RecordingProvider(
        [ModelResponse(content='{"answer":"ok"}', usage=usage)]
    )
    seen: list[RerankUsage] = []

    content = _call_provider(
        provider,
        system="system",
        user="user",
        on_usage=seen.append,
    )

    assert content == '{"answer":"ok"}'
    assert provider.requests[0] == ModelRequest(
        messages=[
            ModelMessage(role="system", content="system"),
            ModelMessage(role="user", content="user"),
        ],
        response_format="json_object",
        temperature=0,
    )
    assert seen == [usage]


def test_call_provider_preserves_provider_errors() -> None:
    error = RerankProviderError("provider failed")

    with pytest.raises(RerankProviderError) as exc_info:
        _call_provider(
            ProviderErrorProvider(error),
            system="system",
            user="user",
            on_usage=None,
        )

    assert exc_info.value is error


def test_call_provider_wraps_unexpected_errors() -> None:
    with pytest.raises(RerankProviderError, match="boom") as exc_info:
        _call_provider(
            BrokenProvider(),
            system="system",
            user="user",
            on_usage=None,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_call_provider_rejects_empty_content() -> None:
    with pytest.raises(RerankProviderError, match="empty response"):
        _call_provider(
            RecordingProvider([ModelResponse(content="")]),
            system="system",
            user="user",
            on_usage=None,
        )
