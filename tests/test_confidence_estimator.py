from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceArtifactError,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceEstimator,
    TaskType,
    _encoder,
)


@dataclass(frozen=True)
class FakeEncoder:
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_revision: str | None = None
    max_length: int = 64
    cache_dir: str | None = "/tmp/private-cache"

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        assert text
        hidden = [[float(row + col) for col in range(4)] for row in range(8)]
        mask = [1] * 8
        return hidden, mask


class FakeScorer:
    def __init__(
        self,
        *,
        score: object = 0.75,
        task_type: TaskType = "answer_confidence",
        max_length: int = 64,
        scorer_type: str = "fake",
        encoder_name: str = "bert-base-uncased",
        tokenizer_name: str = "bert-base-uncased",
    ) -> None:
        self.metadata = ScorerMetadata(
            artifact_schema_version="structural-artifact-v1",
            scorer_type=scorer_type,
            task_type=task_type,
            encoder_name=encoder_name,
            encoder_revision=None,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=None,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
            feature_dtype="float64",
            max_length=max_length,
            granularity="two_scale",
            local_window_size=5,
            local_stride=2,
            score_output="probability",
            positive_class_index=1,
        )
        self.score = score
        self.last_features: list[float] | None = None

    def predict_confidence(self, features: Sequence[float]) -> float:
        assert len(features) == 70
        self.last_features = list(features)
        return self.score  # type: ignore[return-value]


def test_estimator_is_frozen() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(FrozenInstanceError):
        estimator.task_type = "judgment_confidence"  # type: ignore[misc]


def test_estimator_scores_answer_input() -> None:
    scorer = FakeScorer(score=0.75)
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=scorer,
        task_type="answer_confidence",
    )

    result = estimator.score(AnswerConfidenceInput(context="context", answer="answer"))

    assert result.score == 0.75
    assert result.task_type == "answer_confidence"
    assert result.feature_schema_version == "structural-v1"
    assert scorer.last_features is not None
    assert result.metadata["encoder_name"] == "bert-base-uncased"
    assert result.metadata["tokenizer_name"] == "bert-base-uncased"
    assert result.metadata["max_length"] == 64
    assert result.metadata["feature_dim"] == 70
    assert result.metadata["feature_dtype"] == "float64"
    assert result.metadata["granularity"] == "two_scale"
    assert result.metadata["local_window_size"] == 5
    assert result.metadata["local_stride"] == 2
    assert result.metadata["input_template_version"] == "structural-template-v1"
    assert result.metadata["scorer_type"] == "fake"
    assert result.metadata["artifact_schema_version"] == "structural-artifact-v1"


def test_estimator_scores_judgment_input_with_matching_metadata() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(score=0.6, task_type="judgment_confidence"),
        task_type="judgment_confidence",
    )

    result = estimator.score(
        JudgmentConfidenceInput(
            query="query",
            document="document",
            judgment="direct evidence",
        )
    )

    assert result.score == 0.6
    assert result.task_type == "judgment_confidence"


def test_direct_constructor_rejects_task_type_metadata_mismatch() -> None:
    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator(
            encoder=FakeEncoder(),
            scorer=FakeScorer(task_type="judgment_confidence"),
            task_type="answer_confidence",
        )


def test_direct_constructor_rejects_max_length_metadata_mismatch() -> None:
    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator(
            encoder=FakeEncoder(max_length=64),
            scorer=FakeScorer(max_length=128),
            task_type="answer_confidence",
        )


def test_estimator_rejects_wrong_input_type() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score(
            JudgmentConfidenceInput(
                query="query",
                document="document",
                judgment="direct evidence",
            )
        )


def test_from_pretrained_rejects_scorer_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        captured.update(kwargs)
        return FakeEncoder()

    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator.from_pretrained(
            encoder_name="bert-base-uncased",
            scorer=FakeScorer(max_length=128),
            task_type="answer_confidence",
            max_length=64,
            hf_token="secret-token",
            cache_dir="/tmp/private-cache",
        )

    assert captured["hf_token"] == "secret-token"
    assert captured["cache_dir"] == "/tmp/private-cache"


def test_from_pretrained_accepts_matching_scorer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        assert kwargs["tokenizer_name"] is None
        return FakeEncoder()

    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    estimator = StructuralConfidenceEstimator.from_pretrained(
        encoder_name="bert-base-uncased",
        scorer=FakeScorer(),
        task_type="answer_confidence",
        max_length=64,
    )

    assert estimator.encoder.encoder_name == "bert-base-uncased"


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, "0.5", None])
def test_estimator_rejects_invalid_scores(score: object) -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(score=score),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceArtifactError):
        estimator.score(AnswerConfidenceInput(context="context", answer="answer"))


def test_result_metadata_excludes_sensitive_and_heavy_values() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    result = estimator.score(AnswerConfidenceInput(context="context", answer="answer"))

    forbidden = {
        "hf_token",
        "token",
        "cache_dir",
        "model",
        "tokenizer",
        "features",
        "feature_vector",
        "local_path",
    }
    assert forbidden.isdisjoint(result.metadata)
