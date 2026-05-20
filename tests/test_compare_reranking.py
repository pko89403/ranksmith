from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from ranksmith._benchmark import BenchmarkCase, BenchmarkDocument

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_reranking",
    ROOT / "scripts" / "compare_reranking.py",
)
assert SPEC is not None
compare_reranking = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_reranking)


def test_compare_all_skips_tourrank_when_cases_are_not_100_candidates(
    capsys,
) -> None:
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

    assert algorithms == ("rankgpt_sliding_window", "prp_sliding_k")
    assert "Skipping tourrank_r" in capsys.readouterr().err


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
