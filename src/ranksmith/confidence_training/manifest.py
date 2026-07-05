from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ranksmith.confidence_training.types import (
    ConfidenceDatasetSplit,
    ConfidenceTrainingConfig,
)


def build_dataset_manifest(
    dataset_path: Path,
    *,
    task_type: str,
    sample_count: int,
) -> dict[str, object]:
    return {
        "dataset_path": str(dataset_path),
        "dataset_hash": file_sha256(dataset_path),
        "task_type": task_type,
        "sample_count": sample_count,
    }


def build_split_manifest(
    split: ConfidenceDatasetSplit,
    *,
    seed: int,
    task_type: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "train_count": len(split.train),
        "valid_count": len(split.valid),
        "test_count": len(split.test),
        "task_type": task_type,
    }


def training_config_hash(config: ConfidenceTrainingConfig) -> str:
    data = asdict(config)
    data.pop("dataset_path", None)
    data.pop("output_dir", None)
    data.pop("export_path", None)
    return json_hash(data)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(data: object) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
