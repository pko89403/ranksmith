from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ranksmith.confidence import TaskType
from ranksmith.confidence.features import MIN_MAX_LENGTH
from ranksmith.confidence_training.errors import ConfidenceTrainingConfigError

CalibrationMethod = Literal["sigmoid"]


@dataclass(frozen=True)
class ConfidenceTrainingConfig:
    task_type: TaskType
    dataset_path: str | Path
    output_dir: str | Path
    export_path: str | Path
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    max_length: int = 256
    allow_truncation: bool = False
    seed: int = 42
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    calibration_method: CalibrationMethod = "sigmoid"

    def __post_init__(self) -> None:
        if self.task_type not in {
            "answer_confidence",
            "judgment_confidence",
            "query_answerability_confidence",
            "query_context_answerability_confidence",
        }:
            raise ConfidenceTrainingConfigError("unsupported task_type")
        if self.calibration_method != "sigmoid":
            raise ConfidenceTrainingConfigError("unsupported calibration_method")
        if self.max_length < MIN_MAX_LENGTH:
            raise ConfidenceTrainingConfigError(
                f"max_length must be >= {MIN_MAX_LENGTH}"
            )
        if self.train_ratio <= 0 or self.valid_ratio <= 0 or self.test_ratio <= 0:
            raise ConfidenceTrainingConfigError("split ratios must be positive")
        ratio_sum = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ConfidenceTrainingConfigError("split ratios must sum to 1.0")


@dataclass(frozen=True)
class ConfidenceTrainingResult:
    output_dir: Path
    export_path: Path
    report_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CanonicalConfidenceSample:
    id: str
    task_type: TaskType
    label: int
    context: str | None = None
    answer: str | None = None
    query: str | None = None
    document: str | None = None
    judgment: str | None = None
    gold_answer: str | list[str] | None = None
    relevance_label: int | float | bool | None = None
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ConfidenceDatasetSplit:
    train: tuple[CanonicalConfidenceSample, ...]
    valid: tuple[CanonicalConfidenceSample, ...]
    test: tuple[CanonicalConfidenceSample, ...]


@dataclass(frozen=True)
class ConfidenceFeatureRow:
    id: str
    task_type: TaskType
    label: int
    features: tuple[float, ...]
    feature_schema_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(float(v) for v in self.features))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ConfidenceMetricReport:
    sample_count: int
    positive_count: int
    negative_count: int
    accuracy: float
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    calibration_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
        }


@dataclass(frozen=True)
class ConfidenceValidationReport:
    uncalibrated: ConfidenceMetricReport
    calibrated: ConfidenceMetricReport

    def to_dict(self) -> dict[str, object]:
        return {
            **self.calibrated.to_dict(),
            "uncalibrated": self.uncalibrated.to_dict(),
            "calibrated": self.calibrated.to_dict(),
        }


@dataclass(frozen=True)
class ConfidenceTrainingReport:
    valid: ConfidenceValidationReport
    test: ConfidenceMetricReport

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid.to_dict(),
            "test": self.test.to_dict(),
        }
