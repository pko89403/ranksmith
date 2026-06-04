from __future__ import annotations

from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
)


def generate_answer_confidence_dataset(
    config: AnswerGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError


def generate_judgment_confidence_dataset(
    config: RelevanceGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError
