from __future__ import annotations

from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
    UsageCallback,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest
from ranksmith.types import RerankUsage


def generate_answer_confidence_dataset(
    config: AnswerGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError


def generate_judgment_confidence_dataset(
    config: RelevanceGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError


def _call_provider(
    provider: ModelProvider,
    *,
    system: str,
    user: str,
    on_usage: UsageCallback | None,
) -> str:
    try:
        response = provider.complete(
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

    _emit_usage(response.usage, on_usage)
    if response.content == "":
        raise RerankProviderError("Model provider returned an empty response.")
    return response.content


def _emit_usage(usage: RerankUsage | None, callback: UsageCallback | None) -> None:
    if usage is not None and callback is not None:
        callback(usage)
