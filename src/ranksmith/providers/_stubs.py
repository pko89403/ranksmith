from __future__ import annotations

from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse


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
