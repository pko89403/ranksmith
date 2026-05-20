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
                response_format=cast(Any, {"type": request.response_format}),
                temperature=request.temperature,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = response.choices[0].message.content
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
                response_format=cast(Any, {"type": request.response_format}),
                temperature=request.temperature,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = response.choices[0].message.content
        if content is None or content == "":
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return ModelResponse(content=content, usage=_extract_usage(response))


class OpenAIProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("OpenAI provider is not implemented yet.")


class AsyncOpenAIProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("OpenAI provider is not implemented yet.")


class AnthropicProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("Anthropic provider is not implemented yet.")


class AsyncAnthropicProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("Anthropic provider is not implemented yet.")


class GeminiProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("Gemini provider is not implemented yet.")


class AsyncGeminiProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise RerankProviderError("Gemini provider is not implemented yet.")


def _extract_usage(response: object) -> RerankUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return RerankUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _to_openai_messages(request: ModelRequest) -> Any:
    return [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
