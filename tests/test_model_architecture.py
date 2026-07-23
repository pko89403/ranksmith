from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, cast

import pytest

from ranksmith import (
    AsyncAzureAOAIProvider,
    AsyncModelClient,
    AsyncModelProvider,
    AzureAOAIProvider,
    Document,
    ModelClient,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RerankProviderError,
    RerankUsage,
)


class FakeModelProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class FailingModelProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RuntimeError("provider timeout")


class AsyncFakeModelProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeModelClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rank(self, query: str, documents: list[Document]) -> str:
        self.calls.append((query, [document.text for document in documents]))
        return '{"ranking": [1]}'


class AsyncFakeModelClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rank(self, query: str, documents: list[Document]) -> str:
        self.calls.append((query, [document.text for document in documents]))
        return '{"ranking": [1]}'


def test_model_api_is_publicly_importable() -> None:
    assert ModelMessage is not None
    assert ModelRequest is not None
    assert ModelResponse is not None
    assert ModelProvider is not None
    assert AsyncModelProvider is not None
    assert ModelClient is not None
    assert AsyncModelClient is not None
    assert AzureAOAIProvider is not None
    assert AsyncAzureAOAIProvider is not None


def test_removed_llm_provider_names_are_not_public() -> None:
    ranksmith = importlib.import_module("ranksmith")
    protocols = importlib.import_module("ranksmith.protocols")

    removed_names = [
        "LLMProvider",
        "AsyncLLMProvider",
        "PairwiseLLMProvider",
        "AsyncPairwiseLLMProvider",
        "SelectionLLMProvider",
        "AsyncSelectionLLMProvider",
        "Provider",
        "AsyncProvider",
    ]
    for name in removed_names:
        assert not hasattr(ranksmith, name)
        assert not hasattr(protocols, name)


def test_azure_reranker_does_not_construct_provider_when_model_client_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("Azure provider should not be constructed")

    monkeypatch.setattr("ranksmith.azure.AzureAOAIProvider", fail_if_constructed)
    model_client = FakeModelClient()
    reranker = __import__("ranksmith").AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=model_client,
    )

    results = reranker.rerank("query", ["alpha"])

    assert [result.document.text for result in results] == ["alpha"]
    assert model_client.calls == [("query", ["alpha"])]


def test_azure_reranker_accepts_model_client_without_azure_credentials() -> None:
    model_client = FakeModelClient()
    reranker = __import__("ranksmith").AzureOpenAIReranker(model_client=model_client)

    results = reranker.rerank("query", ["alpha"])

    assert [result.document.text for result in results] == ["alpha"]
    assert model_client.calls == [("query", ["alpha"])]


@pytest.mark.asyncio
async def test_async_azure_reranker_skips_provider_when_model_client_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("Azure provider should not be constructed")

    monkeypatch.setattr("ranksmith.azure.AsyncAzureAOAIProvider", fail_if_constructed)
    model_client = AsyncFakeModelClient()
    reranker = __import__("ranksmith").AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=model_client,
    )

    results = await reranker.rerank("query", ["alpha"])

    assert [result.document.text for result in results] == ["alpha"]
    assert model_client.calls == [("query", ["alpha"])]


@pytest.mark.asyncio
async def test_async_azure_reranker_accepts_model_client_without_credentials() -> None:
    model_client = AsyncFakeModelClient()
    reranker = __import__("ranksmith").AsyncAzureOpenAIReranker(
        model_client=model_client
    )

    results = await reranker.rerank("query", ["alpha"])

    assert [result.document.text for result in results] == ["alpha"]
    assert model_client.calls == [("query", ["alpha"])]


def test_model_client_rank_builds_domain_prompt_and_emits_usage() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    provider = FakeModelProvider(
        [ModelResponse(content='{"ranking": [2, 1]}', usage=usage)]
    )
    client = ModelClient(provider=provider, on_usage=captured.append)

    result = client.rank("query", [Document(text="first"), Document(text="second")])

    assert result == '{"ranking": [2, 1]}'
    assert captured == [usage]
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert "ranking" in request.messages[0].content
    assert "exactly 2 integers" in request.messages[0].content
    assert "[1]\nfirst" in request.messages[1].content
    assert "[2]\nsecond" in request.messages[1].content
    assert '{"ranking": [1, 2]}' in request.messages[1].content
    assert "Return exactly 2 candidate numbers" in request.messages[1].content
    assert (
        "verify that sorting your ranking gives [1, 2]" in request.messages[1].content
    )


def test_model_client_compare_and_select_keep_existing_json_contracts() -> None:
    provider = FakeModelProvider(
        [
            ModelResponse(content='{"winner": "B"}'),
            ModelResponse(content='{"selected": [3, 1]}'),
        ]
    )
    client = ModelClient(provider=provider)

    winner = client.compare(
        "query",
        Document(text="alpha"),
        Document(text="beta"),
    )
    selected = client.select(
        "query",
        [Document(text="alpha"), Document(text="beta"), Document(text="gamma")],
        2,
    )

    assert winner == '{"winner": "B"}'
    assert selected == '{"selected": [3, 1]}'
    assert "winner" in provider.requests[0].messages[0].content
    assert "selected" in provider.requests[1].messages[0].content
    assert "exactly 2 integers from 1 to 3" in provider.requests[1].messages[0].content


def test_model_client_wraps_provider_errors_and_rejects_empty_content() -> None:
    client = ModelClient(provider=FailingModelProvider())
    with pytest.raises(RerankProviderError, match="provider timeout"):
        client.rank("query", [Document(text="alpha")])

    empty_client = ModelClient(provider=FakeModelProvider([ModelResponse(content="")]))
    with pytest.raises(RerankProviderError, match="empty response"):
        empty_client.rank("query", [Document(text="alpha")])


@pytest.mark.asyncio
async def test_async_model_client_rank_builds_domain_prompt_and_emits_usage() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9)
    provider = AsyncFakeModelProvider(
        [ModelResponse(content='{"ranking": [1]}', usage=usage)]
    )
    client = AsyncModelClient(provider=provider, on_usage=captured.append)

    result = await client.rank("query", [Document(text="alpha")])

    assert result == '{"ranking": [1]}'
    assert captured == [usage]
    assert "[1]\nalpha" in provider.requests[0].messages[1].content


@dataclass
class _StubUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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
    def __init__(self, content: str | None = '{"ranking": [1]}') -> None:
        self.content = content
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _StubCompletionResponse:
        self.kwargs = kwargs
        return _StubCompletionResponse(
            usage=_StubUsage(11, 2, 13),
            choices=[_StubChoice(message=_StubMessage(content=self.content))],
        )


class _FailingCompletions:
    def create(self, **kwargs: Any) -> _StubCompletionResponse:
        del kwargs
        raise RuntimeError("azure timeout")


class _EmptyChoicesCompletions:
    def create(self, **kwargs: Any) -> _StubCompletionResponse:
        del kwargs
        return _StubCompletionResponse(
            usage=_StubUsage(11, 2, 13),
            choices=[],
        )


class _AsyncEmptyChoicesCompletions:
    async def create(self, **kwargs: Any) -> _StubCompletionResponse:
        del kwargs
        return _StubCompletionResponse(
            usage=_StubUsage(11, 2, 13),
            choices=[],
        )


@dataclass
class _StubChat:
    completions: (
        _StubCompletions
        | _FailingCompletions
        | _EmptyChoicesCompletions
        | _AsyncEmptyChoicesCompletions
    )


@dataclass
class _StubSDKClient:
    chat: _StubChat


def test_azure_aoai_provider_completes_model_request() -> None:
    completions = _StubCompletions()
    provider = AzureAOAIProvider.__new__(AzureAOAIProvider)
    stubbed_provider = cast(Any, provider)
    stubbed_provider._azure_deployment = "deployment"
    stubbed_provider._client = _StubSDKClient(chat=_StubChat(completions=completions))

    response = provider.complete(
        ModelRequest(
            messages=[
                ModelMessage(role="system", content="system"),
                ModelMessage(role="user", content="user"),
            ]
        )
    )

    assert response == ModelResponse(
        content='{"ranking": [1]}',
        usage=RerankUsage(prompt_tokens=11, completion_tokens=2, total_tokens=13),
    )
    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "deployment"
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["temperature"] == 0


def test_azure_aoai_provider_fast_fails_empty_and_sdk_errors() -> None:
    empty_provider = AzureAOAIProvider.__new__(AzureAOAIProvider)
    stubbed_empty_provider = cast(Any, empty_provider)
    stubbed_empty_provider._azure_deployment = "deployment"
    stubbed_empty_provider._client = _StubSDKClient(
        chat=_StubChat(completions=_StubCompletions(content=None))
    )
    with pytest.raises(RerankProviderError, match="empty response"):
        empty_provider.complete(ModelRequest(messages=[ModelMessage("user", "hello")]))

    blank_provider = AzureAOAIProvider.__new__(AzureAOAIProvider)
    stubbed_blank_provider = cast(Any, blank_provider)
    stubbed_blank_provider._azure_deployment = "deployment"
    stubbed_blank_provider._client = _StubSDKClient(
        chat=_StubChat(completions=_StubCompletions(content=""))
    )
    with pytest.raises(RerankProviderError, match="empty response"):
        blank_provider.complete(ModelRequest(messages=[ModelMessage("user", "hello")]))

    failing_provider = AzureAOAIProvider.__new__(AzureAOAIProvider)
    stubbed_failing_provider = cast(Any, failing_provider)
    stubbed_failing_provider._azure_deployment = "deployment"
    stubbed_failing_provider._client = _StubSDKClient(
        chat=_StubChat(completions=_FailingCompletions())
    )
    with pytest.raises(RerankProviderError, match="azure timeout"):
        failing_provider.complete(
            ModelRequest(messages=[ModelMessage("user", "hello")])
        )


def test_azure_aoai_provider_fast_fails_empty_choices() -> None:
    provider = AzureAOAIProvider.__new__(AzureAOAIProvider)
    stubbed_provider = cast(Any, provider)
    stubbed_provider._azure_deployment = "deployment"
    stubbed_provider._client = _StubSDKClient(
        chat=_StubChat(completions=_EmptyChoicesCompletions())
    )

    with pytest.raises(RerankProviderError, match="invalid response"):
        provider.complete(ModelRequest(messages=[ModelMessage("user", "hello")]))


def test_async_azure_aoai_provider_fast_fails_empty_choices() -> None:
    async def run() -> None:
        provider = AsyncAzureAOAIProvider.__new__(AsyncAzureAOAIProvider)
        stubbed_provider = cast(Any, provider)
        stubbed_provider._azure_deployment = "deployment"
        stubbed_provider._client = _StubSDKClient(
            chat=_StubChat(completions=_AsyncEmptyChoicesCompletions())
        )

        with pytest.raises(RerankProviderError, match="invalid response"):
            await provider.complete(
                ModelRequest(messages=[ModelMessage("user", "hello")])
            )

    asyncio.run(run())


def test_model_client_answer_builds_domain_prompt_and_emits_usage() -> None:
    captured: list[RerankUsage] = []
    usage = RerankUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    provider = FakeModelProvider(
        [ModelResponse(content='{"answer": "scurvy"}', usage=usage)]
    )
    client = ModelClient(provider=provider, on_usage=captured.append)

    result = client.answer("What causes scurvy?", "Vitamin C deficiency.")

    assert result == '{"answer": "scurvy"}'
    assert captured == [usage]
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert 'Return only JSON with an "answer" string' in request.messages[0].content
    assert "Question:\nWhat causes scurvy?" in request.messages[1].content
    assert "Context:\nVitamin C deficiency." in request.messages[1].content
    assert '{"answer":"..."}' in request.messages[1].content
    assert '{"answer":"__NO_ANSWER__"}' in request.messages[1].content


def test_model_client_answer_wraps_errors_and_rejects_empty_content() -> None:
    client = ModelClient(provider=FailingModelProvider())
    with pytest.raises(RerankProviderError, match="provider timeout"):
        client.answer("query", "context")

    empty_client = ModelClient(provider=FakeModelProvider([ModelResponse(content="")]))
    with pytest.raises(RerankProviderError, match="empty response"):
        empty_client.answer("query", "context")


@pytest.mark.asyncio
async def test_async_model_client_answer_matches_sync_prompt() -> None:
    sync_provider = FakeModelProvider([ModelResponse(content='{"answer": "x"}')])
    ModelClient(provider=sync_provider).answer("q", "ctx")
    async_provider = AsyncFakeModelProvider([ModelResponse(content='{"answer": "x"}')])

    result = await AsyncModelClient(provider=async_provider).answer("q", "ctx")

    assert result == '{"answer": "x"}'
    assert async_provider.requests[0].messages == sync_provider.requests[0].messages
