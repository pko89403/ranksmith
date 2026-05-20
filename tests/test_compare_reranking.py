from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from ranksmith._benchmark import BenchmarkCase, BenchmarkDocument
from ranksmith._mteb_eval import tourrank_stage_configs_for_candidate_count

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_reranking",
    ROOT / "scripts" / "compare_reranking.py",
)
assert SPEC is not None
compare_reranking = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_reranking)


def test_compare_all_includes_tourrank_for_non_100_candidate_cases() -> None:
    args = argparse.Namespace(algorithm="all")
    cases = [
        BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=tuple(
                BenchmarkDocument(id=str(index), title="", text="")
                for index in range(5)
            ),
            qrels={},
        )
    ]

    algorithms = compare_reranking._selected_algorithms(args, cases)

    assert algorithms == (
        "rankgpt_sliding_window",
        "prp_sliding_k",
        "tourrank_r",
    )


def test_compare_all_includes_tourrank_for_100_candidate_cases() -> None:
    args = argparse.Namespace(algorithm="all")
    cases = [
        BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=tuple(
                BenchmarkDocument(id=str(index), title="", text="")
                for index in range(100)
            ),
            qrels={},
        )
    ]

    assert compare_reranking._selected_algorithms(args, cases) == (
        "rankgpt_sliding_window",
        "prp_sliding_k",
        "tourrank_r",
    )


def test_compare_explicit_tourrank_is_preserved_for_non_100_candidate_cases() -> None:
    args = argparse.Namespace(algorithm="tourrank_r")
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("tourrank_r",)


def test_compare_builds_tourrank_stages_for_non_100_candidate_cases() -> None:
    stage_configs = tourrank_stage_configs_for_candidate_count(5)

    assert [(s.group_count, s.group_size, s.selected_count) for s in stage_configs] == [
        (1, 5, 2),
        (1, 2, 1),
    ]


def test_compare_estimates_tourrank_calls_from_generated_stages() -> None:
    assert (
        compare_reranking._estimate_provider_calls(
            5,
            "tourrank_r",
            window_size=3,
            stride=2,
            tourrank_rounds=2,
        )
        == 4
    )
