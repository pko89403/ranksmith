from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ranksmith.confidence import TaskType

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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ConfidenceDatasetSplit:
    train: tuple[CanonicalConfidenceSample, ...]
    valid: tuple[CanonicalConfidenceSample, ...]
    test: tuple[CanonicalConfidenceSample, ...]
