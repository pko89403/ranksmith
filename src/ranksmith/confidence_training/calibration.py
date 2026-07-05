from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard, cast

from ranksmith.confidence.dependencies import import_optional_dependency
from ranksmith.confidence_training.errors import ConfidenceTrainingError


class PredictProbaModel(Protocol):
    def predict_proba(self, rows: object) -> object: ...


class PredictScoreModel(Protocol):
    def predict(self, rows: object) -> object: ...


PredictiveModel = PredictProbaModel | PredictScoreModel


@dataclass
class CalibratedConfidenceScorer:
    model: PredictiveModel
    calibrator: PredictProbaModel
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
    model: PredictiveModel,
    rows: list[list[float]],
    *,
    positive_class_index: int = 1,
) -> list[float]:
    np = import_optional_dependency("numpy", extra="confidence-train")
    matrix = np.asarray(rows, dtype=float)
    if _has_predict_proba(model):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw = cast(Any, model.predict_proba(matrix))
        return [
            _extract_probability([row], positive_class_index=positive_class_index)
            for row in raw
        ]
    if _has_predict(model):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw = cast(Any, model.predict(matrix))
        return [float(value) for value in raw]
    raise ConfidenceTrainingError("model must provide predict_proba() or predict()")


def _extract_probability(raw: Any, *, positive_class_index: int) -> float:
    row = raw[0]
    value = row[positive_class_index]
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ConfidenceTrainingError("confidence score must be in [0, 1]")
    return score


def _has_predict_proba(model: object) -> TypeGuard[PredictProbaModel]:
    return callable(getattr(model, "predict_proba", None))


def _has_predict(model: object) -> TypeGuard[PredictScoreModel]:
    return callable(getattr(model, "predict", None))
