from __future__ import annotations

import pytest

from ranksmith.confidence_training import ConfidenceTrainingError
from ranksmith.confidence_training._report import generate_training_report
from ranksmith.confidence_training._train import (
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training._types import ConfidenceFeatureRow


def _feature_rows(count: int = 40) -> list[ConfidenceFeatureRow]:
    rows: list[ConfidenceFeatureRow] = []
    for i in range(count):
        label = i % 2
        features = [0.0] * 70
        features[0] = float(label)
        features[1] = float(i) / float(count)
        rows.append(
            ConfidenceFeatureRow(
                id=f"f{i}",
                task_type="answer_confidence",
                label=label,
                features=tuple(features),
                feature_schema_version="structural-v1",
            )
        )
    return rows


def test_train_and_calibrate_classifier_returns_probability() -> None:
    rows = _feature_rows()
    train_rows = rows[:30]
    valid_rows = rows[30:36]

    model = train_lightgbm_classifier(train_rows, seed=7)
    scorer = calibrate_classifier(model, valid_rows)

    score = scorer.predict_confidence(rows[37].features)

    assert 0.0 <= score <= 1.0


def test_generate_training_report_uses_valid_and_test_splits() -> None:
    rows = _feature_rows()
    train_rows = rows[:30]
    valid_rows = rows[30:36]
    test_rows = rows[36:]

    model = train_lightgbm_classifier(train_rows, seed=7)
    scorer = calibrate_classifier(model, valid_rows)
    report = generate_training_report(
        model,
        scorer,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )

    assert report["valid"]["sample_count"] == 6
    assert report["test"]["sample_count"] == 4
    assert "brier_score" in report["test"]
    assert "calibrated" in report["valid"]


def test_train_fails_when_train_labels_have_one_class() -> None:
    rows = [
        ConfidenceFeatureRow(
            id=f"f{i}",
            task_type="answer_confidence",
            label=0,
            features=tuple([0.0] * 70),
            feature_schema_version="structural-v1",
        )
        for i in range(30)
    ]

    with pytest.raises(ConfidenceTrainingError, match="positive and negative"):
        train_lightgbm_classifier(rows, seed=7)


def test_calibration_fails_when_valid_labels_have_one_class() -> None:
    rows = _feature_rows()
    model = train_lightgbm_classifier(rows[:30], seed=7)
    valid_rows = [
        ConfidenceFeatureRow(
            id=f"v{i}",
            task_type="answer_confidence",
            label=1,
            features=tuple([1.0] * 70),
            feature_schema_version="structural-v1",
        )
        for i in range(4)
    ]

    with pytest.raises(ConfidenceTrainingError, match="positive and negative"):
        calibrate_classifier(model, valid_rows)


def test_training_fails_on_feature_length_mismatch() -> None:
    row = ConfidenceFeatureRow(
        id="bad",
        task_type="answer_confidence",
        label=1,
        features=(1.0,),
        feature_schema_version="structural-v1",
    )

    with pytest.raises(ConfidenceTrainingError, match="feature vector length"):
        train_lightgbm_classifier([row], seed=7)
