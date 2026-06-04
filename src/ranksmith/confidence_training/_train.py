from __future__ import annotations

from typing import Any, cast

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._features import FEATURE_DIM, FEATURE_SCHEMA_VERSION
from ranksmith.confidence_training._calibration import (
    CalibratedConfidenceScorer,
    PredictiveModel,
    PredictProbaModel,
)
from ranksmith.confidence_training._errors import ConfidenceTrainingError
from ranksmith.confidence_training._types import ConfidenceFeatureRow


def train_lightgbm_classifier(
    rows: list[ConfidenceFeatureRow] | tuple[ConfidenceFeatureRow, ...],
    *,
    seed: int,
) -> PredictiveModel:
    matrix, labels = feature_matrix_and_labels(rows)
    _require_two_classes(labels, split_name="train")
    lightgbm = import_optional_dependency("lightgbm", extra="confidence-train")
    np: Any = import_optional_dependency("numpy", extra="confidence-train")
    model = lightgbm.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=1,
        min_data_in_bin=1,
        verbosity=-1,
    )
    model.fit(np.asarray(matrix, dtype=float), labels)
    return cast(PredictiveModel, model)


def calibrate_classifier(
    model: PredictiveModel,
    valid_rows: list[ConfidenceFeatureRow] | tuple[ConfidenceFeatureRow, ...],
) -> CalibratedConfidenceScorer:
    matrix, labels = feature_matrix_and_labels(valid_rows)
    _require_two_classes(labels, split_name="valid")
    sklearn_linear = import_optional_dependency(
        "sklearn.linear_model",
        extra="confidence-train",
    )
    from ranksmith.confidence_training._calibration import predict_model_probability

    base_scores = predict_model_probability(model, matrix)
    calibrator = sklearn_linear.LogisticRegression(solver="liblinear")
    calibrator.fit([[score] for score in base_scores], labels)
    return CalibratedConfidenceScorer(
        model=model,
        calibrator=_require_predict_proba(calibrator),
        feature_dim=FEATURE_DIM,
    )


def feature_matrix_and_labels(
    rows: list[ConfidenceFeatureRow] | tuple[ConfidenceFeatureRow, ...],
) -> tuple[list[list[float]], list[int]]:
    if not rows:
        raise ConfidenceTrainingError("feature rows must not be empty")
    matrix: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        _validate_feature_row(row)
        matrix.append(list(row.features))
        labels.append(row.label)
    return matrix, labels


def _validate_feature_row(row: ConfidenceFeatureRow) -> None:
    if row.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ConfidenceTrainingError("unsupported feature schema")
    if len(row.features) != FEATURE_DIM:
        raise ConfidenceTrainingError("feature vector length mismatch")
    np: Any = import_optional_dependency("numpy", extra="confidence-train")
    if not bool(np.all(np.isfinite(np.asarray(row.features, dtype=float)))):
        raise ConfidenceTrainingError("feature vector contains NaN or Inf")
    if row.label not in {0, 1}:
        raise ConfidenceTrainingError("label must be 0 or 1")


def _require_two_classes(labels: list[int], *, split_name: str) -> None:
    if set(labels) != {0, 1}:
        raise ConfidenceTrainingError(
            f"{split_name} split must contain positive and negative labels"
        )


def _require_predict_proba(value: object) -> PredictProbaModel:
    if not callable(getattr(value, "predict_proba", None)):
        raise ConfidenceTrainingError("calibrator must provide predict_proba()")
    return cast(PredictProbaModel, value)
