from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence_training._errors import ConfidenceTrainingError


@dataclass
class CalibratedConfidenceScorer:
    model: object
    calibrator: object
    feature_dim: int
    positive_class_index: int = 1

    def predict_confidence(self, features: tuple[float, ...] | list[float]) -> float:
        if len(features) != self.feature_dim:
            raise ConfidenceTrainingError("feature vector length mismatch")
        base_score = predict_model_probability(
            self.model,
            [list(features)],
            positive_class_index=self.positive_class_index,
        )[0]
        raw = self.calibrator.predict_proba([[base_score]])
        return _extract_probability(raw, positive_class_index=1)


def predict_model_probability(
    model: object,
    rows: list[list[float]],
    *,
    positive_class_index: int = 1,
) -> list[float]:
    np = import_optional_dependency("numpy", extra="confidence-train")
    matrix = np.asarray(rows, dtype=float)
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw = predict_proba(matrix)
        return [
            _extract_probability([row], positive_class_index=positive_class_index)
            for row in raw
        ]
    predict = getattr(model, "predict", None)
    if callable(predict):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw = predict(matrix)
        return [float(value) for value in raw]
    raise ConfidenceTrainingError("model must provide predict_proba() or predict()")


def _extract_probability(raw: Any, *, positive_class_index: int) -> float:
    row = raw[0]
    value = row[positive_class_index]
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ConfidenceTrainingError("confidence score must be in [0, 1]")
    return score
