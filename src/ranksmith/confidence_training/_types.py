from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
