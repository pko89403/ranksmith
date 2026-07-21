from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence_training import ConfidenceDatasetError
from ranksmith.confidence_training.dataset_report import build_dataset_report

REPORT_SCRIPT_PATH = Path("scripts/report_confidence_dataset.py")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_report_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "report_confidence_dataset",
        REPORT_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_confidence_dataset_help_exits_successfully() -> None:
    result = subprocess.run(
        ["uv", "run", "python", str(REPORT_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--task" in result.stdout


def test_build_dataset_report_counts_sources_groups_and_missing_values(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "q1",
                "task_type": "query_answerability_confidence",
                "query": "q1",
                "answer": "a",
                "label": 1,
                "source": "alpha",
                "group_id": "g1",
            },
            {
                "id": "q2",
                "task_type": "query_answerability_confidence",
                "query": "q2",
                "answer": "a",
                "label": 0,
                "source": "alpha",
            },
            {
                "id": "q3",
                "task_type": "query_answerability_confidence",
                "query": "q3",
                "answer": "a",
                "label": 1,
                "source": "beta",
                "group_id": "g2",
            },
            {
                "id": "q4",
                "task_type": "query_answerability_confidence",
                "query": "q4",
                "answer": "a",
                "label": 0,
            },
        ],
    )

    report = build_dataset_report(dataset, "query_answerability_confidence")

    assert report["task_type"] == "query_answerability_confidence"
    assert report["sample_count"] == 4
    assert report["positive_count"] == 2
    assert report["negative_count"] == 2
    assert report["positive_rate"] == 0.5
    assert report["source_count"] == 2
    assert report["group_count"] == 2
    assert report["missing_source_count"] == 1
    assert report["missing_source_rate"] == 0.25
    assert report["missing_group_id_count"] == 2
    assert report["missing_group_id_rate"] == 0.5
    assert report["sources"] == {
        "__MISSING__": {
            "sample_count": 1,
            "positive_count": 0,
            "negative_count": 1,
            "positive_rate": 0.0,
        },
        "alpha": {
            "sample_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "positive_rate": 0.5,
        },
        "beta": {
            "sample_count": 1,
            "positive_count": 1,
            "negative_count": 0,
            "positive_rate": 1.0,
        },
    }


def test_build_dataset_report_inherits_duplicate_id_error(tmp_path: Path) -> None:
    dataset = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "q1", "query": "q", "answer": "a", "label": 1},
            {"id": "q1", "query": "q", "answer": "a", "label": 0},
        ],
    )

    with pytest.raises(ConfidenceDatasetError, match="duplicate id"):
        build_dataset_report(dataset, "query_answerability_confidence")


def test_report_cli_prints_pretty_json(tmp_path: Path, capsys: Any) -> None:
    script = _load_report_script()
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [{"id": "q1", "query": "q", "answer": "a", "label": 1, "source": "fixture"}],
    )

    status = script.main(
        [
            "--task",
            "query_answerability_confidence",
            "--dataset",
            str(dataset),
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert '\n  "sample_count": 1' in output
    assert json.loads(output)["sources"]["fixture"]["positive_rate"] == 1.0
