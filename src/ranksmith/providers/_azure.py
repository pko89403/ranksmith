from __future__ import annotations

from typing import Any, cast

from openai import AsyncAzureOpenAI, AzureOpenAI

from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse
from ranksmith.types import RerankUsage


class AzureAOAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        timeout: float | None = None,
    ) -> None:
        self._azure_deployment = azure_deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=_to_openai_messages(request),
                response_format=cast(Any, {"type": "json_object"}),
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = _extract_content(response)
        if content is None or content == "":
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return ModelResponse(content=content, usage=_extract_usage(response))


class AsyncAzureAOAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        timeout: float | None = None,
    ) -> None:
        self._azure_deployment = azure_deployment
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=_to_openai_messages(request),
                response_format=cast(Any, {"type": "json_object"}),
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = _extract_content(response)
        if content is None or content == "":
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return ModelResponse(content=content, usage=_extract_usage(response))


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
        raise RerankProviderError("Azure OpenAI returned an invalid response.")
    try:
        return cast(str | None, choices[0].message.content)
    except AttributeError as exc:
        raise RerankProviderError("Azure OpenAI returned an invalid response.") from exc


def _to_openai_messages(request: ModelRequest) -> Any:
    return [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
