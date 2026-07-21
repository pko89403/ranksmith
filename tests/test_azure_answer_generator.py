from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from ranksmith.errors import RerankInputError, RerankParseError, RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse


@dataclass
class FakeProvider:
    responses: list[str]
    requests: list[ModelRequest]

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self.responses.pop(0))


def test_integrations_exports_are_submodule_only() -> None:
    integrations = importlib.import_module("ranksmith.integrations")
    root = importlib.import_module("ranksmith")

    assert integrations.AzureAnswerGenerator is not None
    assert not hasattr(root, "AzureAnswerGenerator")


def test_azure_answer_generator_parses_query_answer_json() -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "Nancy Travis"}'], requests=[])
    generator = AzureAnswerGenerator(provider=provider)

    assert generator.answer_query("who played karen?") == "Nancy Travis"
    assert provider.requests[0].response_format == "json_object"
    assert provider.requests[0].temperature == 0
    assert provider.requests[0].messages[0].role == "system"
    assert provider.requests[0].messages[1].role == "user"
    assert "who played karen?" in provider.requests[0].messages[1].content
    assert "__NO_ANSWER__" in provider.requests[0].messages[1].content
    assert "best concise answer" not in provider.requests[0].messages[0].content


def test_azure_answer_generator_parses_context_answer_json() -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "Nancy Travis"}'], requests=[])
    generator = AzureAnswerGenerator(provider=provider)

    assert (
        generator.answer_with_context(
            "who played karen?",
            "Nancy Travis played Karen.",
        )
        == "Nancy Travis"
    )
    assert "Nancy Travis played Karen." in provider.requests[0].messages[1].content
    assert "__NO_ANSWER__" in provider.requests[0].messages[1].content


def test_azure_answer_generator_uses_configured_no_answer_value() -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "UNKNOWN"}'], requests=[])
    generator = AzureAnswerGenerator(provider=provider, no_answer_value="UNKNOWN")

    assert generator.answer_query("query") == "UNKNOWN"
    assert '{"answer":"UNKNOWN"}' in provider.requests[0].messages[1].content


def test_azure_answer_generator_rejects_empty_no_answer_value() -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    with pytest.raises(ValueError, match="no_answer_value"):
        AzureAnswerGenerator(
            provider=FakeProvider(responses=[], requests=[]),
            no_answer_value=" ",
        )


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        '{"answer": ""}',
        '{"answer": "   "}',
        '{"answer": 123}',
    ],
)
def test_azure_answer_generator_rejects_invalid_answer_json(content: str) -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    generator = AzureAnswerGenerator(
        provider=FakeProvider(responses=[content], requests=[]),
    )

    with pytest.raises(RerankParseError):
        generator.answer_query("query")


def test_azure_answer_generator_preserves_provider_error() -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    class FailingProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RerankProviderError("provider failed")

    generator = AzureAnswerGenerator(provider=FailingProvider())

    with pytest.raises(RerankProviderError, match="provider failed"):
        generator.answer_query("query")


def test_azure_answer_generator_requires_azure_config_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ranksmith.integrations import AzureAnswerGenerator

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_LLM_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RerankInputError, match="AZURE_OPENAI_API_KEY"):
        AzureAnswerGenerator.from_env()
