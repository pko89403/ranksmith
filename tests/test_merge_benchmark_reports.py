from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_benchmark_reports",
    ROOT / "scripts" / "merge_benchmark_reports.py",
)
assert SPEC is not None
merger = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(merger)


def _report(
    algorithms: list[str],
    *,
    query_ids: list[str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    query_ids = query_ids if query_ids is not None else ["q1", "q2"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_type": "reranking_with_first_stage_candidates",
        "dataset": "benchmark-cache",
        "dataset_name": "askubuntu-bm25",
        "candidate_strategy": "candidate_file",
        "candidates": "benchmark-results/pyserini/askubuntu-bm25-top20.trec",
        "candidate_count": 20,
        "case_count": len(query_ids),
        "top_k": 5,
        "seed": 13,
        "timeout": None,
        "checkpoint_output": None,
        "algorithm": algorithms,
        "aggregate": [
            {
                "algorithm": algorithm,
                "case_count": len(query_ids),
                "metrics": {"ndcg@5": 0.4},
            }
            for algorithm in algorithms
        ],
        "per_query": [
            {"algorithm": algorithm, "query_id": query_id, "valid": True}
            for algorithm in algorithms
            for query_id in query_ids
        ],
        "call_estimates": {algorithm: 1 for algorithm in algorithms},
        "method_settings": {algorithm: {} for algorithm in algorithms},
    }
    report.update(overrides)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run_merge(tmp_path: Path, *reports: dict[str, Any]) -> dict[str, Any]:
    paths = [
        _write_report(tmp_path / f"report{index}.json", report)
        for index, report in enumerate(reports)
    ]
    output = tmp_path / "merged.json"
    original = sys.argv
    sys.argv = [
        "merge_benchmark_reports.py",
        *[str(path) for path in paths],
        "--output",
        str(output),
    ]
    try:
        merger.main()
    finally:
        sys.argv = original
    return json.loads(output.read_text(encoding="utf-8"))


def test_merge_combines_disjoint_algorithm_runs(tmp_path: Path) -> None:
    merged = _run_merge(
        tmp_path,
        _report(["original_bm25", "single_call_listwise@20"]),
        _report(["answer_confidence"], timeout=120.0),
    )

    assert merged["algorithm"] == [
        "original_bm25",
        "single_call_listwise@20",
        "answer_confidence",
    ]
    assert [row["algorithm"] for row in merged["aggregate"]] == merged["algorithm"]
    assert len(merged["per_query"]) == 6
    assert merged["call_estimates"] == {
        "original_bm25": 1,
        "single_call_listwise@20": 1,
        "answer_confidence": 1,
    }
    assert merged["merged_from"][0]["divergent_fields"] == {}
    assert merged["merged_from"][1]["divergent_fields"] == {"timeout": 120.0}


def test_merge_rejects_duplicate_algorithms(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="more than one input"):
        _run_merge(
            tmp_path,
            _report(["original_bm25"]),
            _report(["original_bm25"]),
        )


def test_merge_rejects_mismatched_benchmark_identity(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="disagree on 'candidate_count'"):
        _run_merge(
            tmp_path,
            _report(["original_bm25"]),
            _report(["answer_confidence"], candidate_count=10),
        )


def test_merge_rejects_different_candidate_files(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="different candidate files"):
        _run_merge(
            tmp_path,
            _report(["original_bm25"]),
            _report(["answer_confidence"], candidates="runs/other-top20.trec"),
        )


def test_merge_accepts_same_candidate_file_under_different_paths(
    tmp_path: Path,
) -> None:
    merged = _run_merge(
        tmp_path,
        _report(["original_bm25"]),
        _report(
            ["answer_confidence"],
            candidates="/home/elsewhere/askubuntu-bm25-top20.trec",
        ),
    )

    assert merged["algorithm"] == ["original_bm25", "answer_confidence"]


def test_merge_rejects_runs_on_different_query_ids(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="different query ids"):
        _run_merge(
            tmp_path,
            _report(["original_bm25"]),
            _report(
                ["answer_confidence"],
                query_ids=["q1", "q3"],
                case_count=2,
            ),
        )


def test_merge_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "merged.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _run_merge(tmp_path, _report(["original_bm25"]), _report(["tourrank_r2"]))
