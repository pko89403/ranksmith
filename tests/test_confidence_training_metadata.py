from __future__ import annotations

from pathlib import Path

from ranksmith.confidence.scorer import metadata_from_dict
from ranksmith.confidence_training import ConfidenceTrainingConfig
from ranksmith.confidence_training.artifact import build_scorer_metadata


def test_build_scorer_metadata_contains_phase_1_required_fields(tmp_path: Path) -> None:
    config = ConfidenceTrainingConfig(
        task_type="judgment_confidence",
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "run",
        export_path=tmp_path / "artifact.joblib",
        max_length=34,
    )

    metadata = build_scorer_metadata(
        config,
        train_count=30,
        valid_count=6,
        test_count=4,
        dataset_manifest_hash="dataset-hash",
        training_config_hash="config-hash",
    )

    parsed = metadata_from_dict(metadata)

    assert parsed.task_type == "judgment_confidence"
    assert parsed.input_template_version == "structural-template-v1"
    assert parsed.feature_schema_version == "structural-v1"
    assert parsed.feature_dim == 70
    assert parsed.feature_dtype == "float64"
    assert parsed.score_output == "probability"
    assert parsed.extra["train_count"] == 30
