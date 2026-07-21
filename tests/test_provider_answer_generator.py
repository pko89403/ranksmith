from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from ranksmith.errors import RerankParseError, RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse


@dataclass
class FakeProvider:
    responses: list[str]
    requests: list[ModelRequest]

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self.responses.pop(0))


def test_provider_answer_generator_export_is_submodule_only() -> None:
    integrations = importlib.import_module("ranksmith.integrations")
    root = importlib.import_module("ranksmith")

    assert integrations.ProviderAnswerGenerator is not None
    assert not hasattr(root, "ProviderAnswerGenerator")


def test_provider_answer_generator_parses_query_answer_json() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "Paris"}'], requests=[])
    generator = ProviderAnswerGenerator(provider=provider)

    assert generator.answer_query("capital of france?") == "Paris"
    assert provider.requests[0].response_format == "json_object"
    assert provider.requests[0].temperature == 0
    assert provider.requests[0].messages[0].role == "system"
    assert provider.requests[0].messages[1].role == "user"
    assert "capital of france?" in provider.requests[0].messages[1].content
    assert "Answer from your parametric knowledge" in (
        provider.requests[0].messages[1].content
    )
    assert "__NO_ANSWER__" in provider.requests[0].messages[1].content
    assert "best concise answer" not in provider.requests[0].messages[0].content


def test_provider_answer_generator_parses_context_answer_json() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "Nancy Travis"}'], requests=[])
    generator = ProviderAnswerGenerator(provider=provider)

    assert (
        generator.answer_with_context(
            "who played karen?",
            "Nancy Travis played Karen.",
        )
        == "Nancy Travis"
    )
    assert "Nancy Travis played Karen." in provider.requests[0].messages[1].content
    assert "Use only the context" in provider.requests[0].messages[1].content
    assert "__NO_ANSWER__" in provider.requests[0].messages[1].content


def test_provider_answer_generator_uses_configured_no_answer_value() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    provider = FakeProvider(responses=['{"answer": "UNKNOWN"}'], requests=[])
    generator = ProviderAnswerGenerator(provider=provider, no_answer_value="UNKNOWN")

    assert generator.answer_query("query") == "UNKNOWN"
    assert '{"answer":"UNKNOWN"}' in provider.requests[0].messages[1].content


def test_provider_answer_generator_rejects_empty_no_answer_value() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    with pytest.raises(ValueError, match="no_answer_value"):
        ProviderAnswerGenerator(
            provider=FakeProvider(responses=[], requests=[]),
            no_answer_value=" ",
        )


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        "{}",
        '{"answer": ""}',
        '{"answer": "   "}',
        '{"answer": 123}',
    ],
)
def test_provider_answer_generator_rejects_invalid_answer_json(content: str) -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    generator = ProviderAnswerGenerator(
        provider=FakeProvider(responses=[content], requests=[]),
    )

    with pytest.raises(RerankParseError):
        generator.answer_query("query")


def test_provider_answer_generator_preserves_provider_error() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    class FailingProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RerankProviderError("provider failed")

    generator = ProviderAnswerGenerator(provider=FailingProvider())

    with pytest.raises(RerankProviderError, match="provider failed"):
        generator.answer_query("query")


def test_provider_answer_generator_wraps_unexpected_provider_error() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    class FailingProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RuntimeError("boom")

    generator = ProviderAnswerGenerator(provider=FailingProvider())

    with pytest.raises(RerankProviderError, match="boom"):
        generator.answer_query("query")
