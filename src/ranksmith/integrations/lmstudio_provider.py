from __future__ import annotations

import os
from typing import Any, Protocol, cast

from openai import OpenAI

from ranksmith.errors import RerankInputError, RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse
from ranksmith.types import RerankUsage


class _CompletionsClient(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
    ) -> object: ...


class _ChatClient(Protocol):
    completions: _CompletionsClient


class _OpenAICompatibleClient(Protocol):
    chat: _ChatClient


class LMStudioModelProvider:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int = 128,
        client: _OpenAICompatibleClient | None = None,
    ) -> None:
        resolved_model = (
            model if model is not None else os.environ.get("LMSTUDIO_MODEL")
        )
        if resolved_model is None or resolved_model.strip() == "":
            raise RerankInputError("LMSTUDIO_MODEL is required")
        if max_tokens <= 0:
            raise RerankInputError("max_tokens must be greater than 0")

        resolved_base_url = _resolve_optional_setting(
            explicit=base_url,
            env_name="LMSTUDIO_BASE_URL",
            default="http://localhost:1234/v1",
            setting_name="base_url",
        )
        resolved_api_key = _resolve_optional_setting(
            explicit=api_key,
            env_name="LMSTUDIO_API_KEY",
            default="lm-studio",
            setting_name="api_key",
        )

        self._model = resolved_model
        self._base_url = resolved_base_url
        self._api_key_configured = resolved_api_key != ""
        self._max_tokens = max_tokens
        self._client: _OpenAICompatibleClient = (
            client
            if client is not None
            else cast(
                _OpenAICompatibleClient,
                OpenAI(
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    timeout=timeout,
                ),
            )
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key_configured(self) -> bool:
        return self._api_key_configured

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def complete(self, request: ModelRequest) -> ModelResponse:
        response_format = _to_lmstudio_response_format(request)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(request),
                response_format=response_format,
                temperature=request.temperature,
                max_tokens=self._max_tokens,
                reasoning_effort="none",
            )
        except Exception as exc:
            message = str(exc)
            if message:
                raise RerankProviderError(
                    f"LM Studio request failed: {message}"
                ) from exc
            raise RerankProviderError("LM Studio request failed.") from exc

        content = _extract_content(response)
        if content is None or content == "":
            raise RerankProviderError("LM Studio returned an empty response.")
        return ModelResponse(content=content, usage=_extract_usage(response))


def _to_lmstudio_response_format(request: ModelRequest) -> dict[str, Any]:
    if request.response_format != "json_object":
        raise RerankProviderError("LM Studio received unsupported response_format.")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ranksmith_json_response",
            "schema": {"type": "object"},
        },
    }


def _resolve_optional_setting(
    *,
    explicit: str | None,
    env_name: str,
    default: str,
    setting_name: str,
) -> str:
    if explicit is not None:
        if explicit.strip() == "":
            raise RerankInputError(f"{setting_name} must not be blank")
        return explicit
    value = os.environ.get(env_name)
    if value is not None:
        if value.strip() == "":
            raise RerankInputError(f"{env_name} must not be blank")
        return value
    return default


def _extract_usage(response: object) -> RerankUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return RerankUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _extract_content(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RerankProviderError("LM Studio returned an invalid response.")
    try:
        content = choices[0].message.content
    except AttributeError as exc:
        raise RerankProviderError("LM Studio returned an invalid response.") from exc
    if content is not None and not isinstance(content, str):
        raise RerankProviderError("LM Studio returned an invalid response.")
    return content


def _to_openai_messages(request: ModelRequest) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
