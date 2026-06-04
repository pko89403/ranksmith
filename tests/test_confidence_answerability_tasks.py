from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from ranksmith.confidence import (
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceEstimator,
    TaskType,
)


@dataclass
class FakeEncoder:
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_revision: str | None = None
    max_length: int = 64
    last_text: str | None = None

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        self.last_text = text
        hidden = [[float(row + col) for col in range(4)] for row in range(8)]
        mask = [1] * 8
        return hidden, mask


class FakeScorer:
    def __init__(
        self,
        *,
        task_type: TaskType,
        score: float = 0.75,
    ) -> None:
        self.metadata = ScorerMetadata(
            artifact_schema_version="structural-artifact-v1",
            scorer_type="fake",
            task_type=task_type,
            encoder_name="bert-base-uncased",
            encoder_revision=None,
            tokenizer_name="bert-base-uncased",
            tokenizer_revision=None,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
            feature_dtype="float64",
            max_length=64,
            granularity="two_scale",
            local_window_size=5,
            local_stride=2,
            score_output="probability",
            positive_class_index=1,
        )
        self.score = score

    def predict_confidence(self, features: Sequence[float]) -> float:
        assert len(features) == 70
        return self.score


@pytest.fixture(autouse=True)
def fake_structural_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ranksmith.confidence._structural.extract_structural_features",
        lambda hidden_states, mask, *, max_length: [0.0] * 70,
    )


def test_confidence_submodule_exports_answerability_inputs() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert confidence.QueryAnswerabilityConfidenceInput is (
        QueryAnswerabilityConfidenceInput
    )
    assert confidence.QueryContextAnswerabilityConfidenceInput is (
        QueryContextAnswerabilityConfidenceInput
    )


def test_estimator_scores_query_answerability_input_with_exact_template() -> None:
    encoder = FakeEncoder()
    estimator = StructuralConfidenceEstimator(
        encoder=encoder,
        scorer=FakeScorer(task_type="query_answerability_confidence"),
        task_type="query_answerability_confidence",
    )

    result = estimator.score(
        QueryAnswerabilityConfidenceInput(query="Who?", answer="Nancy Travis")
    )

    assert result.score == 0.75
    assert result.task_type == "query_answerability_confidence"
    assert encoder.last_text == "Query:\nWho?\n\nAnswer:\nNancy Travis"


def test_estimator_scores_query_context_answerability_input_with_exact_template() -> (
    None
):
    encoder = FakeEncoder()
    estimator = StructuralConfidenceEstimator(
        encoder=encoder,
        scorer=FakeScorer(task_type="query_context_answerability_confidence"),
        task_type="query_context_answerability_confidence",
    )

    result = estimator.score(
        QueryContextAnswerabilityConfidenceInput(
            query="Who?",
            context="Karen was played by Nancy Travis.",
            answer="Nancy Travis",
        )
    )

    assert result.score == 0.75
    assert result.task_type == "query_context_answerability_confidence"
    assert encoder.last_text == (
        "Query:\nWho?\n\nContext:\nKaren was played by Nancy Travis."
        "\n\nAnswer:\nNancy Travis"
    )
