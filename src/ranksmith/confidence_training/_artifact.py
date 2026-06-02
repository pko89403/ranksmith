from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    GRANULARITY,
    LOCAL_STRIDE,
    LOCAL_WINDOW_SIZE,
)
from ranksmith.confidence._scorer import (
    ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_SCORE_OUTPUT,
    metadata_from_dict,
)
from ranksmith.confidence._templates import INPUT_TEMPLATE_VERSION
from ranksmith.confidence_training._errors import ConfidenceTrainingError
from ranksmith.confidence_training._types import ConfidenceTrainingConfig

LABEL_SCHEMA_VERSION = "supervised-binary-v1"


def build_scorer_metadata(
    config: ConfidenceTrainingConfig,
    *,
    train_count: int,
    valid_count: int,
    test_count: int,
    dataset_manifest_hash: str,
    training_config_hash: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scorer_type": "lightgbm-calibrated-sigmoid",
        "task_type": config.task_type,
        "encoder_name": config.encoder_name,
        "encoder_revision": config.encoder_revision,
        "tokenizer_name": config.tokenizer_name or config.encoder_name,
        "tokenizer_revision": config.tokenizer_revision,
        "input_template_version": INPUT_TEMPLATE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": FEATURE_DTYPE,
        "max_length": config.max_length,
        "granularity": GRANULARITY,
        "local_window_size": LOCAL_WINDOW_SIZE,
        "local_stride": LOCAL_STRIDE,
        "score_output": SUPPORTED_SCORE_OUTPUT,
        "positive_class_index": 1,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "dataset_manifest_hash": dataset_manifest_hash,
        "split_seed": config.seed,
        "train_count": train_count,
        "valid_count": valid_count,
        "test_count": test_count,
        "calibration_method": config.calibration_method,
        "training_config_hash": training_config_hash,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _validate_phase_1_metadata(metadata)
    return metadata


def export_scorer_artifact(
    scorer: object,
    *,
    config: ConfidenceTrainingConfig,
    train_count: int,
    valid_count: int,
    test_count: int,
    dataset_manifest_hash: str,
    training_config_hash: str,
) -> dict[str, Any]:
    metadata = build_scorer_metadata(
        config,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        dataset_manifest_hash=dataset_manifest_hash,
        training_config_hash=training_config_hash,
    )
    joblib = __import__("joblib")
    export_path = Path(config.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "scorer": scorer}, export_path)
    return metadata


def write_metadata_json(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_phase_1_metadata(metadata: dict[str, Any]) -> None:
    try:
        metadata_from_dict(metadata)
    except Exception as exc:
        raise ConfidenceTrainingError(
            "metadata incompatible with Phase 1 loader"
        ) from exc
