from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ranksmith.confidence_training import ConfidenceTrainingConfigError


def test_confidence_training_public_submodule_exports_are_available() -> None:
    confidence_training = importlib.import_module("ranksmith.confidence_training")

    assert hasattr(confidence_training, "ConfidenceTrainingConfig")
    assert hasattr(confidence_training, "ConfidenceTrainingResult")
    assert hasattr(confidence_training, "ConfidenceTrainingError")
    assert hasattr(confidence_training, "ConfidenceDatasetError")
    assert hasattr(confidence_training, "ConfidenceLabelError")
    assert hasattr(confidence_training, "ConfidenceTrainingConfigError")
    assert hasattr(confidence_training, "train_confidence_scorer")


def test_confidence_training_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    assert not hasattr(ranksmith, "ConfidenceTrainingConfig")
    assert not hasattr(ranksmith, "ConfidenceTrainingResult")
    assert not hasattr(ranksmith, "ConfidenceTrainingError")
    assert not hasattr(ranksmith, "train_confidence_scorer")


def test_confidence_training_config_defaults_are_phase_2a_scope(
    tmp_path: Path,
) -> None:
    confidence_training = importlib.import_module("ranksmith.confidence_training")

    config = confidence_training.ConfidenceTrainingConfig(
        task_type="answer_confidence",
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "run",
        export_path=tmp_path / "artifact.joblib",
    )

    assert config.encoder_name == "bert-base-uncased"
    assert config.tokenizer_name is None
    assert config.max_length == 256
    assert config.allow_truncation is False
    assert config.local_files_only is False
    assert config.train_ratio == 0.8
    assert config.valid_ratio == 0.1
    assert config.test_ratio == 0.1
    assert config.calibration_method == "sigmoid"


def test_confidence_training_config_rejects_unsupported_calibration_method(
    tmp_path: Path,
) -> None:
    confidence_training = importlib.import_module("ranksmith.confidence_training")

    with pytest.raises(
        ConfidenceTrainingConfigError,
        match="unsupported calibration_method",
    ):
        confidence_training.ConfidenceTrainingConfig(
            task_type="answer_confidence",
            dataset_path=tmp_path / "dataset.jsonl",
            output_dir=tmp_path / "run",
            export_path=tmp_path / "artifact.joblib",
            calibration_method="isotonic",
        )
