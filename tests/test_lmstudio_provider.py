from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast

import pytest

from ranksmith.errors import RerankInputError, RerankProviderError
from ranksmith.model import ModelMessage, ModelRequest
from ranksmith.types import RerankUsage


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage | None = None


class FakeCompletions:
    def __init__(self, response: object | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = type(
            "FakeChat",
            (),
            {"completions": completions},
        )()


class RecordingOpenAI(FakeClient):
    instances: list[RecordingOpenAI] = []

    def __init__(self, *, api_key: str, base_url: str, timeout: float | None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.completions = FakeCompletions(
            FakeResponse(choices=[FakeChoice(FakeMessage('{"ranking": [1]}'))])
        )
        super().__init__(self.completions)
        self.instances.append(self)


def _request(response_format: str = "json_object") -> ModelRequest:
    return ModelRequest(
        messages=[
            ModelMessage(role="system", content="system"),
            ModelMessage(role="user", content="user"),
        ],
        response_format=cast(Any, response_format),
        temperature=0.2,
    )


def test_lmstudio_provider_export_is_submodule_only() -> None:
    integrations = importlib.import_module("ranksmith.integrations")
    root = importlib.import_module("ranksmith")

    assert integrations.LMStudioModelProvider is not None
    assert not hasattr(root, "LMStudioModelProvider")


def test_lmstudio_provider_converts_json_object_to_json_schema() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    completions = FakeCompletions(
        FakeResponse(
            choices=[FakeChoice(FakeMessage('{"ranking": [1]}'))],
            usage=FakeUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )
    )
    provider = LMStudioModelProvider(
        model="local-model",
        client=FakeClient(completions),
    )

    response = provider.complete(_request())

    assert response.content == '{"ranking": [1]}'
    assert response.usage == RerankUsage(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
    )
    assert completions.calls == [
        {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ranksmith_json_response",
                    "schema": {"type": "object"},
                },
            },
            "temperature": 0.2,
            "max_tokens": 128,
            "reasoning_effort": "none",
        }
    ]


def test_lmstudio_provider_passes_custom_max_tokens() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    completions = FakeCompletions(
        FakeResponse(choices=[FakeChoice(FakeMessage('{"ranking": [1]}'))])
    )
    provider = LMStudioModelProvider(
        model="local-model",
        max_tokens=64,
        client=FakeClient(completions),
    )

    provider.complete(_request())

    assert completions.calls[0]["max_tokens"] == 64


def test_lmstudio_provider_uses_lmstudio_model_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    monkeypatch.setenv("LMSTUDIO_MODEL", "env-model")
    completions = FakeCompletions(
        FakeResponse(choices=[FakeChoice(FakeMessage('{"ranking": [1]}'))])
    )
    provider = LMStudioModelProvider(client=FakeClient(completions))

    provider.complete(_request())

    assert completions.calls[0]["model"] == "env-model"


def test_lmstudio_provider_constructs_client_from_env_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ranksmith.integrations._lmstudio_provider as lmstudio_provider
    from ranksmith.integrations import LMStudioModelProvider

    RecordingOpenAI.instances = []
    monkeypatch.setattr(lmstudio_provider, "OpenAI", RecordingOpenAI)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:4321/v1")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "env-key")
    monkeypatch.setenv("LMSTUDIO_MODEL", "env-model")

    provider = LMStudioModelProvider(timeout=2.5)
    provider.complete(_request())

    assert len(RecordingOpenAI.instances) == 1
    constructed = RecordingOpenAI.instances[0]
    assert constructed.base_url == "http://localhost:4321/v1"
    assert constructed.api_key == "env-key"
    assert constructed.timeout == 2.5
    assert constructed.completions.calls[0]["model"] == "env-model"


def test_lmstudio_provider_constructs_client_from_default_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ranksmith.integrations._lmstudio_provider as lmstudio_provider
    from ranksmith.integrations import LMStudioModelProvider

    RecordingOpenAI.instances = []
    monkeypatch.setattr(lmstudio_provider, "OpenAI", RecordingOpenAI)
    monkeypatch.setenv("LMSTUDIO_MODEL", "env-model")
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)

    provider = LMStudioModelProvider()

    assert len(RecordingOpenAI.instances) == 1
    constructed = RecordingOpenAI.instances[0]
    assert constructed.base_url == "http://localhost:1234/v1"
    assert constructed.api_key == "lm-studio"
    assert provider.model == "env-model"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key_configured is True
    assert provider.max_tokens == 128


def test_lmstudio_provider_missing_model_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)

    with pytest.raises(RerankInputError, match="LMSTUDIO_MODEL is required"):
        LMStudioModelProvider(model=" ")


def test_lmstudio_provider_rejects_blank_explicit_base_url() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    with pytest.raises(RerankInputError, match="base_url"):
        LMStudioModelProvider(model="local-model", base_url=" ")


def test_lmstudio_provider_rejects_blank_explicit_api_key() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    with pytest.raises(RerankInputError, match="api_key"):
        LMStudioModelProvider(model="local-model", api_key=" ")


@pytest.mark.parametrize(
    ("env_name", "match"),
    [
        ("LMSTUDIO_BASE_URL", "LMSTUDIO_BASE_URL"),
        ("LMSTUDIO_API_KEY", "LMSTUDIO_API_KEY"),
    ],
)
def test_lmstudio_provider_rejects_blank_env_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    match: str,
) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    monkeypatch.setenv(env_name, " ")

    with pytest.raises(RerankInputError, match=match):
        LMStudioModelProvider(model="local-model")


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_lmstudio_provider_rejects_non_positive_max_tokens(max_tokens: int) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    with pytest.raises(RerankInputError, match="max_tokens"):
        LMStudioModelProvider(
            model="local-model",
            max_tokens=max_tokens,
            client=FakeClient(FakeCompletions()),
        )


def test_lmstudio_provider_wraps_client_errors() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    provider = LMStudioModelProvider(
        model="local-model",
        client=FakeClient(FakeCompletions(error=RuntimeError("boom"))),
    )

    with pytest.raises(RerankProviderError, match="boom"):
        provider.complete(_request())


def test_lmstudio_provider_wraps_empty_client_error_with_context() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    provider = LMStudioModelProvider(
        model="local-model",
        client=FakeClient(FakeCompletions(error=RuntimeError())),
    )

    with pytest.raises(RerankProviderError, match="LM Studio request failed"):
        provider.complete(_request())


def test_lmstudio_provider_rejects_unexpected_response_format() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    provider = LMStudioModelProvider(
        model="local-model",
        client=FakeClient(FakeCompletions()),
    )

    with pytest.raises(RerankProviderError, match="response_format"):
        provider.complete(_request("text"))


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(choices=[]),
        FakeResponse(choices=[FakeChoice(FakeMessage(None))]),
        FakeResponse(choices=[FakeChoice(FakeMessage(""))]),
    ],
)
def test_lmstudio_provider_rejects_invalid_or_empty_content(
    response: FakeResponse,
) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    provider = LMStudioModelProvider(
        model="local-model",
        client=FakeClient(FakeCompletions(response)),
    )

    with pytest.raises(RerankProviderError):
        provider.complete(_request())
