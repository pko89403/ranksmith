from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence import load_lightgbm_scorer
from ranksmith.confidence_training import (
    ConfidenceTrainingConfig,
    ConfidenceTrainingResult,
    train_confidence_scorer,
)
from ranksmith.confidence_training._artifact import export_scorer_artifact
from ranksmith.confidence_training._train import (
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training._types import ConfidenceFeatureRow


class FakeEncoder:
    max_length = 34

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        signal = 1.0 if "positive" in text else 0.0
        hidden = [[signal + float(row), float(row + 1)] for row in range(6)]
        return hidden, [1, 1, 1, 1, 1, 1]


def _fake_from_pretrained(**kwargs: Any) -> FakeEncoder:
    return FakeEncoder()


def _feature_rows(count: int = 40) -> list[ConfidenceFeatureRow]:
    rows: list[ConfidenceFeatureRow] = []
    for i in range(count):
        label = i % 2
        features = [0.0] * 70
        features[0] = float(label)
        features[1] = float(i) / float(count)
        rows.append(
            ConfidenceFeatureRow(
                id=f"f{i}",
                task_type="answer_confidence",
                label=label,
                features=tuple(features),
                feature_schema_version="structural-v1",
            )
        )
    return rows


def _write_answer_dataset(path: Path, count: int = 40) -> None:
    rows = []
    for i in range(count):
        label = i % 2
        rows.append(
            {
                "id": f"a{i}",
                "context": f"{'positive' if label else 'negative'} context {i}",
                "answer": f"answer {i}",
                "label": label,
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exported_artifact_loads_with_phase_1_loader(tmp_path: Path) -> None:
    rows = _feature_rows()
    model = train_lightgbm_classifier(rows[:30], seed=7)
    scorer = calibrate_classifier(model, rows[30:36])
    export_path = tmp_path / "answer_confidence.joblib"
    config = ConfidenceTrainingConfig(
        task_type="answer_confidence",
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "run",
        export_path=export_path,
        max_length=34,
    )

    export_scorer_artifact(
        scorer,
        config=config,
        train_count=30,
        valid_count=6,
        test_count=4,
        dataset_manifest_hash="dataset-hash",
        training_config_hash="config-hash",
    )

    loaded = load_lightgbm_scorer(export_path)

    assert loaded.metadata.task_type == "answer_confidence"
    assert loaded.metadata.feature_dim == 70
    assert 0.0 <= loaded.predict_confidence(rows[37].features) <= 1.0


def test_train_confidence_scorer_pipeline_exports_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    export_path = tmp_path / "artifact.joblib"
    output_dir = tmp_path / "run"
    _write_answer_dataset(dataset_path)

    monkeypatch.setattr(
        "ranksmith.confidence_training._pipeline.FrozenAutoEncoder.from_pretrained",
        _fake_from_pretrained,
    )

    result = train_confidence_scorer(
        ConfidenceTrainingConfig(
            task_type="answer_confidence",
            dataset_path=dataset_path,
            output_dir=output_dir,
            export_path=export_path,
            max_length=34,
            seed=7,
        )
    )

    assert isinstance(result, ConfidenceTrainingResult)
    assert result.export_path == export_path
    assert export_path.exists()
    assert (output_dir / "report.json").exists()
    assert load_lightgbm_scorer(export_path).metadata.task_type == "answer_confidence"
