from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksmith.confidence_training import (
    ConfidenceDatasetError,
    ConfidenceTrainingConfig,
)
from ranksmith.confidence_training.dataset import load_canonical_dataset
from ranksmith.confidence_training.features import extract_feature_rows
from ranksmith.confidence_training.types import CanonicalConfidenceSample


class RecordingEncoder:
    max_length = 34

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        self.texts.append(text)
        return [[0.0, 1.0], [1.0, 2.0]], [1, 1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_config_accepts_query_answerability_confidence(tmp_path: Path) -> None:
    config = ConfidenceTrainingConfig(
        task_type="query_answerability_confidence",
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "run",
        export_path=tmp_path / "artifact.joblib",
    )

    assert config.task_type == "query_answerability_confidence"


def test_config_accepts_query_context_answerability_confidence(
    tmp_path: Path,
) -> None:
    config = ConfidenceTrainingConfig(
        task_type="query_context_answerability_confidence",
        dataset_path=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "run",
        export_path=tmp_path / "artifact.joblib",
    )

    assert config.task_type == "query_context_answerability_confidence"


def test_load_query_answerability_confidence_canonical_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query-answerability.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "qa1",
                "query": "Who played Karen?",
                "answer": "Nancy Travis",
                "label": 1,
                "gold_answer": ["Nancy Travis"],
                "metadata": {"split": "fixture"},
            }
        ],
    )

    samples = load_canonical_dataset(
        path,
        task_type="query_answerability_confidence",
    )

    assert len(samples) == 1
    assert samples[0].task_type == "query_answerability_confidence"
    assert samples[0].query == "Who played Karen?"
    assert samples[0].answer == "Nancy Travis"
    assert samples[0].gold_answer == ["Nancy Travis"]
    assert samples[0].metadata["split"] == "fixture"


def test_load_query_context_answerability_confidence_canonical_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query-context-answerability.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "qca1",
                "query": "Who played Karen?",
                "context": "Karen was played by Nancy Travis.",
                "answer": "Nancy Travis",
                "label": 1,
                "gold_answer": "Nancy Travis",
                "group_id": "question-1",
            }
        ],
    )

    samples = load_canonical_dataset(
        path,
        task_type="query_context_answerability_confidence",
    )

    assert len(samples) == 1
    assert samples[0].task_type == "query_context_answerability_confidence"
    assert samples[0].query == "Who played Karen?"
    assert samples[0].context == "Karen was played by Nancy Travis."
    assert samples[0].answer == "Nancy Travis"
    assert samples[0].gold_answer == "Nancy Travis"
    assert samples[0].group_id == "question-1"


def test_query_context_answerability_missing_context_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-context.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "qca1",
                "query": "Who played Karen?",
                "answer": "Nancy Travis",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ConfidenceDatasetError, match="missing required field: context"):
        load_canonical_dataset(
            path,
            task_type="query_context_answerability_confidence",
        )


def test_extract_feature_rows_formats_answerability_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ranksmith.confidence_training.features.extract_structural_features",
        lambda hidden_states, attention_mask, *, max_length: [0.0] * 70,
    )
    encoder = RecordingEncoder()
    samples = [
        CanonicalConfidenceSample(
            id="qa1",
            task_type="query_answerability_confidence",
            label=1,
            query="Who played Karen?",
            answer="Nancy Travis",
        ),
        CanonicalConfidenceSample(
            id="qca1",
            task_type="query_context_answerability_confidence",
            label=0,
            query="Who played Karen?",
            context="Karen was played by Nancy Travis.",
            answer="Nancy Travis",
        ),
    ]

    rows = extract_feature_rows(samples, encoder=encoder)

    assert [row.task_type for row in rows] == [
        "query_answerability_confidence",
        "query_context_answerability_confidence",
    ]
    assert encoder.texts == [
        "Query:\nWho played Karen?\n\nAnswer:\nNancy Travis",
        (
            "Query:\nWho played Karen?\n\nContext:\n"
            "Karen was played by Nancy Travis.\n\nAnswer:\nNancy Travis"
        ),
    ]
