from __future__ import annotations

from typing import Protocol

from ranksmith.confidence import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
)
from ranksmith.confidence._errors import ConfidenceError
from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    extract_structural_features,
)
from ranksmith.confidence._types import StructuralConfidenceInput
from ranksmith.confidence_training._errors import ConfidenceTrainingError
from ranksmith.confidence_training._types import (
    CanonicalConfidenceSample,
    ConfidenceFeatureRow,
)


class EncoderLike(Protocol):
    @property
    def max_length(self) -> int: ...

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]: ...


def extract_feature_rows(
    samples: list[CanonicalConfidenceSample] | tuple[CanonicalConfidenceSample, ...],
    *,
    encoder: EncoderLike,
) -> list[ConfidenceFeatureRow]:
    rows: list[ConfidenceFeatureRow] = []
    for sample in samples:
        rows.append(_extract_feature_row(sample, encoder=encoder))
    return rows


def _extract_feature_row(
    sample: CanonicalConfidenceSample,
    *,
    encoder: EncoderLike,
) -> ConfidenceFeatureRow:
    try:
        hidden_states, attention_mask = encoder.encode(_sample_text(sample))
        features = extract_structural_features(
            hidden_states,
            attention_mask,
            max_length=encoder.max_length,
        )
    except (ConfidenceError, TypeError, ValueError) as exc:
        raise ConfidenceTrainingError(
            f"feature extraction failed for sample {sample.id}"
        ) from exc
    if len(features) != FEATURE_DIM:
        raise ConfidenceTrainingError("feature vector length mismatch")
    return ConfidenceFeatureRow(
        id=sample.id,
        task_type=sample.task_type,
        label=sample.label,
        features=tuple(features),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        metadata=sample.metadata,
    )


def _sample_text(sample: CanonicalConfidenceSample) -> str:
    from ranksmith.confidence._templates import format_confidence_input

    return format_confidence_input(sample.task_type, _sample_input(sample))


def _sample_input(sample: CanonicalConfidenceSample) -> StructuralConfidenceInput:
    if sample.task_type == "answer_confidence":
        if sample.context is None or sample.answer is None:
            raise ConfidenceTrainingError("answer sample is missing text fields")
        return AnswerConfidenceInput(context=sample.context, answer=sample.answer)
    if sample.task_type == "judgment_confidence":
        if sample.query is None or sample.document is None or sample.judgment is None:
            raise ConfidenceTrainingError("judgment sample is missing text fields")
        return JudgmentConfidenceInput(
            query=sample.query,
            document=sample.document,
            judgment=sample.judgment,
        )
    if sample.task_type == "query_answerability_confidence":
        if sample.query is None or sample.answer is None:
            raise ConfidenceTrainingError(
                "query answerability sample is missing text fields"
            )
        return QueryAnswerabilityConfidenceInput(
            query=sample.query,
            answer=sample.answer,
        )
    if sample.task_type == "query_context_answerability_confidence":
        if sample.query is None or sample.context is None or sample.answer is None:
            raise ConfidenceTrainingError(
                "query context answerability sample is missing text fields"
            )
        return QueryContextAnswerabilityConfidenceInput(
            query=sample.query,
            context=sample.context,
            answer=sample.answer,
        )
    raise ConfidenceTrainingError(f"unsupported task_type: {sample.task_type}")
