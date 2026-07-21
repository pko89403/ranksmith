from __future__ import annotations

import os
from dataclasses import dataclass, field

from ranksmith.errors import RerankInputError
from ranksmith.integrations._answer_generator import ProviderAnswerGenerator
from ranksmith.integrations._validation import validate_no_answer_value
from ranksmith.model import ModelProvider
from ranksmith.providers import AzureAOAIProvider


@dataclass(frozen=True)
class AzureAnswerGenerator:
    provider: ModelProvider = field(init=False)
    no_answer_value: str = field(init=False, default="__NO_ANSWER__")
    _generator: ProviderAnswerGenerator = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str = "2024-08-01-preview",
        timeout: float | None = None,
        provider: ModelProvider | None = None,
        no_answer_value: str = "__NO_ANSWER__",
    ) -> None:
        validate_no_answer_value(no_answer_value)
        if provider is not None:
            resolved_provider = provider
        else:
            if api_key is None:
                raise RerankInputError("AZURE_OPENAI_API_KEY is required")
            if azure_endpoint is None:
                raise RerankInputError("AZURE_OPENAI_ENDPOINT is required")
            if azure_deployment is None:
                raise RerankInputError(
                    "AZURE_OPENAI_LLM_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT is required"
                )
            resolved_provider = AzureAOAIProvider(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                timeout=timeout,
            )
        resolved_generator = ProviderAnswerGenerator(
            provider=resolved_provider,
            no_answer_value=no_answer_value,
        )
        object.__setattr__(self, "provider", resolved_provider)
        object.__setattr__(self, "no_answer_value", no_answer_value)
        object.__setattr__(self, "_generator", resolved_generator)

    @classmethod
    def from_env(
        cls,
        *,
        timeout: float | None = None,
        no_answer_value: str = "__NO_ANSWER__",
    ) -> AzureAnswerGenerator:
        return cls(
            api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=_required_env(
                "AZURE_OPENAI_LLM_DEPLOYMENT",
                fallback="AZURE_OPENAI_DEPLOYMENT",
            ),
            api_version=_env_value(
                "AZURE_OPENAI_LLM_API_VERSION",
                fallback="AZURE_OPENAI_API_VERSION",
                default="2024-08-01-preview",
            )
            or "2024-08-01-preview",
            timeout=timeout,
            no_answer_value=no_answer_value,
        )

    def answer_query(self, query: str) -> str:
        return self._generator.answer_query(query)

    def answer_with_context(self, query: str, context: str) -> str:
        return self._generator.answer_with_context(query, context)


def _required_env(name: str, *, fallback: str | None = None) -> str:
    value = _env_value(name, fallback=fallback)
    if value is None or value == "":
        names = name if fallback is None else f"{name} or {fallback}"
        raise RerankInputError(f"{names} is required")
    return value


def _env_value(
    name: str,
    *,
    fallback: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    if fallback is not None:
        fallback_value = os.environ.get(fallback)
        if fallback_value is not None and fallback_value != "":
            return fallback_value
    return default
