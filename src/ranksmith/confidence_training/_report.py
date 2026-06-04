from __future__ import annotations

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence_training._calibration import (
    CalibratedConfidenceScorer,
    PredictiveModel,
    predict_model_probability,
)
from ranksmith.confidence_training._errors import ConfidenceTrainingError
from ranksmith.confidence_training._train import feature_matrix_and_labels
from ranksmith.confidence_training._types import (
    ConfidenceFeatureRow,
    ConfidenceMetricReport,
    ConfidenceTrainingReport,
    ConfidenceValidationReport,
)


def generate_training_report(
    model: PredictiveModel,
    scorer: CalibratedConfidenceScorer,
    *,
    valid_rows: list[ConfidenceFeatureRow] | tuple[ConfidenceFeatureRow, ...],
    test_rows: list[ConfidenceFeatureRow] | tuple[ConfidenceFeatureRow, ...],
) -> ConfidenceTrainingReport:
    valid_matrix, valid_labels = feature_matrix_and_labels(valid_rows)
    test_matrix, test_labels = feature_matrix_and_labels(test_rows)
    valid_uncalibrated = _metrics(
        valid_labels,
        predict_model_probability(model, valid_matrix),
    )
    valid_calibrated = _metrics(
        valid_labels,
        [scorer.predict_confidence(tuple(row)) for row in valid_matrix],
    )
    test_calibrated = _metrics(
        test_labels,
        [scorer.predict_confidence(tuple(row)) for row in test_matrix],
    )
    return ConfidenceTrainingReport(
        valid=ConfidenceValidationReport(
            uncalibrated=valid_uncalibrated,
            calibrated=valid_calibrated,
        ),
        test=test_calibrated,
    )


def _metrics(labels: list[int], scores: list[float]) -> ConfidenceMetricReport:
    if set(labels) != {0, 1}:
        raise ConfidenceTrainingError(
            "metric split must contain positive and negative labels"
        )
    sklearn_metrics = import_optional_dependency(
        "sklearn.metrics",
        extra="confidence-train",
    )
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    return ConfidenceMetricReport(
        sample_count=len(labels),
        positive_count=sum(labels),
        negative_count=len(labels) - sum(labels),
        accuracy=float(sklearn_metrics.accuracy_score(labels, predictions)),
        roc_auc=float(sklearn_metrics.roc_auc_score(labels, scores)),
        average_precision=float(
            sklearn_metrics.average_precision_score(labels, scores)
        ),
        brier_score=float(sklearn_metrics.brier_score_loss(labels, scores)),
        log_loss=float(sklearn_metrics.log_loss(labels, scores, labels=[0, 1])),
        calibration_error=_expected_calibration_error(labels, scores),
    )


def _expected_calibration_error(labels: list[int], scores: list[float]) -> float:
    bin_count = 10
    total = len(labels)
    error = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        pairs = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if lower <= score < upper or (index == bin_count - 1 and score == 1.0)
        ]
        if not pairs:
            continue
        accuracy = sum(label for label, _ in pairs) / len(pairs)
        confidence = sum(score for _, score in pairs) / len(pairs)
        error += (len(pairs) / total) * abs(accuracy - confidence)
    return float(error)
