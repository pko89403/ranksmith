from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceArtifactError,
    ConfidenceDependencyError,
    ConfidenceError,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceResult,
)


def test_answer_confidence_input_is_frozen() -> None:
    item = AnswerConfidenceInput(context="context", answer="answer")

    with pytest.raises(FrozenInstanceError):
        item.answer = "changed"  # type: ignore[misc]


def test_judgment_confidence_input_is_frozen() -> None:
    item = JudgmentConfidenceInput(
        query="query",
        document="document",
        judgment="direct evidence",
    )

    with pytest.raises(FrozenInstanceError):
        item.judgment = "changed"  # type: ignore[misc]


def test_structural_confidence_result_copies_metadata() -> None:
    metadata = {"encoder_name": "bert-base-uncased"}

    result = StructuralConfidenceResult(
        score=0.7,
        task_type="answer_confidence",
        feature_schema_version="structural-v1",
        metadata=metadata,
    )
    metadata["encoder_name"] = "changed"

    assert result.metadata == {"encoder_name": "bert-base-uncased"}


def test_scorer_metadata_preserves_extra_fields() -> None:
    metadata = ScorerMetadata(
        artifact_schema_version="structural-artifact-v1",
        scorer_type="lightgbm",
        task_type="answer_confidence",
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        input_template_version="structural-template-v1",
        feature_schema_version="structural-v1",
        feature_dim=70,
        feature_dtype="float64",
        max_length=256,
        granularity="two_scale",
        local_window_size=5,
        local_stride=2,
        score_output="probability",
        positive_class_index=1,
        extra={"trained_on": "fixture"},
    )

    assert metadata.extra == {"trained_on": "fixture"}


def test_confidence_errors_share_base_class() -> None:
    assert issubclass(ConfidenceDependencyError, ConfidenceError)
    assert issubclass(ConfidenceInputError, ConfidenceError)
    assert issubclass(ConfidenceArtifactError, ConfidenceError)
