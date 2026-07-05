from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksmith.confidence_training import ConfidenceDatasetError, ConfidenceLabelError
from ranksmith.confidence_training.dataset import load_canonical_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_answer_confidence_canonical_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a1",
                "context": "Paris is in France.",
                "answer": "France",
                "label": 1,
                "gold_answer": "France",
                "source": "fixture",
                "group_id": "q1",
                "metadata": {"kind": "answer"},
            }
        ],
    )

    samples = load_canonical_dataset(path, task_type="answer_confidence")

    assert len(samples) == 1
    assert samples[0].id == "a1"
    assert samples[0].label == 1
    assert samples[0].task_type == "answer_confidence"
    assert samples[0].context == "Paris is in France."
    assert samples[0].answer == "France"
    assert samples[0].group_id == "q1"
    assert samples[0].metadata["kind"] == "answer"


def test_load_judgment_confidence_canonical_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "judgment.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "j1",
                "query": "capital of France",
                "document": "Paris is the capital of France.",
                "judgment": "direct evidence",
                "label": 1,
                "relevance_label": 1,
                "source": "fixture",
                "metadata": {"kind": "judgment"},
            }
        ],
    )

    samples = load_canonical_dataset(path, task_type="judgment_confidence")

    assert len(samples) == 1
    assert samples[0].id == "j1"
    assert samples[0].label == 1
    assert samples[0].task_type == "judgment_confidence"
    assert samples[0].query == "capital of France"
    assert samples[0].document == "Paris is the capital of France."
    assert samples[0].judgment == "direct evidence"
    assert samples[0].metadata["kind"] == "judgment"


def test_missing_required_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "context", "label": 1}])

    with pytest.raises(ConfidenceDatasetError, match="missing required field: answer"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_empty_text_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": " ", "answer": "x", "label": 1}])

    with pytest.raises(ConfidenceDatasetError, match="context must not be empty"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_invalid_label_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-label.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 2}])

    with pytest.raises(ConfidenceLabelError, match="label must be 0 or 1"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_bool_label_fails(tmp_path: Path) -> None:
    path = tmp_path / "bool-label.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": True}])

    with pytest.raises(ConfidenceLabelError, match="label must be 0 or 1"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_float_label_fails(tmp_path: Path) -> None:
    path = tmp_path / "float-label.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 1.0}])

    with pytest.raises(ConfidenceLabelError, match="label must be 0 or 1"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_unhashable_label_fails_with_label_error(tmp_path: Path) -> None:
    path = tmp_path / "list-label.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": []}])

    with pytest.raises(ConfidenceLabelError, match="label must be 0 or 1"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_duplicate_id_fails(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "a1", "context": "c1", "answer": "a1", "label": 1},
            {"id": "a1", "context": "c2", "answer": "a2", "label": 0},
        ],
    )

    with pytest.raises(ConfidenceDatasetError, match="duplicate id: a1"):
        load_canonical_dataset(path, task_type="answer_confidence")


def test_unsupported_task_type_fails(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 1}])

    with pytest.raises(ConfidenceDatasetError, match="unsupported task_type"):
        load_canonical_dataset(path, task_type="other")  # type: ignore[arg-type]


def test_task_specific_foreign_fields_fail(tmp_path: Path) -> None:
    path = tmp_path / "foreign.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a1",
                "context": "c",
                "answer": "a",
                "query": "q",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ConfidenceDatasetError, match="unexpected field for task"):
        load_canonical_dataset(path, task_type="answer_confidence")
