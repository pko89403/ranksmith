from __future__ import annotations

import math

import pytest

from ranksmith.confidence_training import ConfidenceTrainingError
from ranksmith.confidence_training._features import extract_feature_rows
from ranksmith.confidence_training._types import CanonicalConfidenceSample


class FakeEncoder:
    max_length = 34

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        assert "Context:" in text or "Query:" in text
        hidden = [[float(row), float(row + 1)] for row in range(6)]
        mask = [1, 1, 1, 1, 1, 1]
        return hidden, mask


class NaNEncoder:
    max_length = 34

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        return [[1.0, math.nan], [2.0, 3.0]], [1, 1]


def test_extract_feature_rows_for_answer_samples() -> None:
    samples = [
        CanonicalConfidenceSample(
            id="a1",
            task_type="answer_confidence",
            label=1,
            context="Paris is in France.",
            answer="France",
            metadata={"source": "fixture"},
        )
    ]

    rows = extract_feature_rows(samples, encoder=FakeEncoder())

    assert len(rows) == 1
    assert rows[0].id == "a1"
    assert rows[0].task_type == "answer_confidence"
    assert rows[0].label == 1
    assert rows[0].feature_schema_version == "structural-v1"
    assert len(rows[0].features) == 70
    assert rows[0].metadata["source"] == "fixture"


def test_extract_feature_rows_for_judgment_samples() -> None:
    samples = [
        CanonicalConfidenceSample(
            id="j1",
            task_type="judgment_confidence",
            label=0,
            query="capital of France",
            document="Paris is the capital of France.",
            judgment="direct evidence",
        )
    ]

    rows = extract_feature_rows(samples, encoder=FakeEncoder())

    assert len(rows) == 1
    assert rows[0].id == "j1"
    assert rows[0].task_type == "judgment_confidence"
    assert rows[0].label == 0
    assert len(rows[0].features) == 70


def test_extract_feature_rows_fail_on_nan_features() -> None:
    samples = [
        CanonicalConfidenceSample(
            id="a1",
            task_type="answer_confidence",
            label=1,
            context="context",
            answer="answer",
        )
    ]

    with pytest.raises(ConfidenceTrainingError, match="feature extraction failed"):
        extract_feature_rows(samples, encoder=NaNEncoder())
