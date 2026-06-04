from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ranksmith.errors import RerankInputError, RerankParseError, RerankProviderError
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest
from ranksmith.providers import AzureAOAIProvider


@dataclass(frozen=True)
class AzureAnswerGenerator:
    provider: ModelProvider
    no_answer_value: str = "__NO_ANSWER__"

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
        _validate_no_answer_value(no_answer_value)
        object.__setattr__(self, "no_answer_value", no_answer_value)
        if provider is not None:
            object.__setattr__(self, "provider", provider)
            return
        if api_key is None:
            raise RerankInputError("AZURE_OPENAI_API_KEY is required")
        if azure_endpoint is None:
            raise RerankInputError("AZURE_OPENAI_ENDPOINT is required")
        if azure_deployment is None:
            raise RerankInputError(
                "AZURE_OPENAI_LLM_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT is required"
            )
        object.__setattr__(
            self,
            "provider",
            AzureAOAIProvider(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                timeout=timeout,
            ),
        )

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
        return self._complete(
            system=(
                "You answer questions for confidence estimation. Return only JSON "
                'with an "answer" string.'
            ),
            user=(
                f"Question:\n{query}\n\n"
                "Return JSON exactly like this shape:\n"
                '{"answer":"..."}\n\n'
                "Answer from your parametric knowledge. If you do not know the "
                f"answer, return {_answer_contract(self.no_answer_value)}."
            ),
        )

    def answer_with_context(self, query: str, context: str) -> str:
        return self._complete(
            system=(
                "You answer questions using the provided context for confidence "
                'estimation. Return only JSON with an "answer" string.'
            ),
            user=(
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Return JSON exactly like this shape:\n"
                '{"answer":"..."}\n\n'
                "Use only the context. If the context does not contain the answer, "
                f"return {_answer_contract(self.no_answer_value)}."
            ),
        )

    def _complete(self, *, system: str, user: str) -> str:
        try:
            response = self.provider.complete(
                ModelRequest(
                    messages=[
                        ModelMessage(role="system", content=system),
                        ModelMessage(role="user", content=user),
                    ],
                    response_format="json_object",
                    temperature=0,
                )
            )
        except RerankProviderError:
            raise
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc
        return _parse_answer(response.content)


def _parse_answer(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RerankParseError("answer response must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RerankParseError("answer response must be a JSON object")
    answer = parsed.get("answer")
    if not isinstance(answer, str) or answer.strip() == "":
        raise RerankParseError('answer response must contain a non-empty "answer"')
    return answer


def _answer_contract(no_answer_value: str) -> str:
    return json.dumps(
        {"answer": no_answer_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_no_answer_value(value: object) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("no_answer_value must be a non-empty string")


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
