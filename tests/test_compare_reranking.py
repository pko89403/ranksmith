from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from ranksmith import RerankParseError
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


def test_compare_all_uses_top20_bm25_default_methods() -> None:
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
        "original_bm25",
        "single_call_listwise@20",
        "rankgpt_sw_w5",
        "acurank_k5_b1",
        "tourrank_r2",
        "prp_sliding_p1",
    )


def test_compare_all_is_stable_for_100_candidate_cases() -> None:
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
        "original_bm25",
        "single_call_listwise@20",
        "rankgpt_sw_w5",
        "acurank_k5_b1",
        "tourrank_r2",
        "prp_sliding_p1",
    )


def test_compare_optional_prp_sliding_p3_is_preserved() -> None:
    args = argparse.Namespace(algorithm="prp_sliding_p3")
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("prp_sliding_p3",)


def test_compare_optional_prp_sliding_p1_is_preserved() -> None:
    args = argparse.Namespace(algorithm="prp_sliding_p1")
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("prp_sliding_p1",)


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


def test_compare_explicit_acurank_is_preserved() -> None:
    args = argparse.Namespace(algorithm="acurank")
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("acurank",)


def test_compare_estimates_acurank_calls_from_candidate_count() -> None:
    assert (
        compare_reranking._estimate_provider_calls(
            5,
            "acurank_b1",
            window_size=3,
            stride=2,
        )
        == 3
    )


def test_compare_top20_default_alias_call_estimates() -> None:
    assert (
        compare_reranking._estimate_provider_calls(
            20, "single_call_listwise@20", window_size=20, stride=10
        )
        == 1
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "rankgpt_sw_w5", window_size=20, stride=10
        )
        == 9
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "acurank_k5_b1", window_size=20, stride=10
        )
        == 2
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "tourrank_r2", window_size=20, stride=10
        )
        == 8
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "prp_sliding_p1", window_size=20, stride=10
        )
        == 38
    )


def test_compare_optional_alias_call_estimates() -> None:
    assert (
        compare_reranking._estimate_provider_calls(
            20, "acurank_k5_b4", window_size=20, stride=10
        )
        == 5
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "tourrank_r10", window_size=20, stride=10
        )
        == 40
    )
    assert (
        compare_reranking._estimate_provider_calls(
            20, "prp_sliding_p3", window_size=20, stride=10
        )
        == 114
    )


def test_compare_top100_optional_call_estimates_remain_available() -> None:
    assert (
        compare_reranking._estimate_provider_calls(
            100, "tourrank_r10", window_size=20, stride=10
        )
        == 130
    )


def test_compare_acurank_live_strategy_does_not_force_estimated_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, **kwargs: object) -> None:
            captured["strategy"] = kwargs["strategy"]

        def rerank(
            self,
            query: str,
            documents: list[object],
        ) -> list[object]:
            del query
            return [
                type("Result", (), {"document": document})() for document in documents
            ]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setattr("ranksmith.AzureOpenAIReranker", FakeReranker)

    compare_reranking._rank_case(
        case=BenchmarkCase(
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
        ),
        algorithm="acurank_b4",
        window_size=3,
        stride=2,
        passes=10,
        tourrank_rounds=2,
    )

    assert captured["strategy"].max_adaptive_reranker_calls == 4


def test_compare_acurank_b1_uses_budget_one(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, **kwargs: object) -> None:
            captured["strategy"] = kwargs["strategy"]

        def rerank(
            self,
            query: str,
            documents: list[object],
        ) -> list[object]:
            del query
            return [
                type("Result", (), {"document": document})() for document in documents
            ]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setattr("ranksmith.AzureOpenAIReranker", FakeReranker)

    compare_reranking._rank_case(
        case=BenchmarkCase(
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
        ),
        algorithm="acurank_b1",
        window_size=3,
        stride=2,
        passes=10,
        tourrank_rounds=2,
    )

    assert captured["strategy"].max_adaptive_reranker_calls == 1


def test_compare_acurank_k5_b1_uses_target_rank_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, **kwargs: object) -> None:
            captured["strategy"] = kwargs["strategy"]

        def rerank(
            self,
            query: str,
            documents: list[object],
        ) -> list[object]:
            del query
            return [
                type("Result", (), {"document": document})() for document in documents
            ]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setattr("ranksmith.AzureOpenAIReranker", FakeReranker)

    compare_reranking._rank_case(
        case=BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=tuple(
                BenchmarkDocument(id=str(index), title="", text="")
                for index in range(20)
            ),
            qrels={},
        ),
        algorithm="acurank_k5_b1",
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
    )

    assert captured["strategy"].target_rank == 5
    assert captured["strategy"].max_adaptive_reranker_calls == 1


def test_compare_rankgpt_sw_w5_uses_small_sliding_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, **kwargs: object) -> None:
            captured["strategy"] = kwargs["strategy"]

        def rerank(
            self,
            query: str,
            documents: list[object],
        ) -> list[object]:
            del query
            return [
                type("Result", (), {"document": document})() for document in documents
            ]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setattr("ranksmith.AzureOpenAIReranker", FakeReranker)

    compare_reranking._rank_case(
        case=BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=tuple(
                BenchmarkDocument(id=str(index), title="", text="")
                for index in range(20)
            ),
            qrels={},
        ),
        algorithm="rankgpt_sw_w5",
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
    )

    assert captured["strategy"].window_size == 5
    assert captured["strategy"].stride == 2


def test_compare_original_bm25_returns_candidate_order() -> None:
    ranked = compare_reranking._rank_case(
        case=BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=tuple(
                BenchmarkDocument(id=str(index), title="", text="")
                for index in range(3)
            ),
            qrels={},
        ),
        algorithm="original_bm25",
        window_size=3,
        stride=2,
        passes=10,
        tourrank_rounds=2,
    )

    assert ranked == ("0", "1", "2")


def test_compare_original_bm25_only_does_not_require_live_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_reranking.py",
            "--algorithm",
            "original_bm25",
            "--output",
            str(output_path),
        ],
    )
    compare_reranking.main()

    assert "Offline comparison" in capsys.readouterr().err
    assert output_path.exists()


def test_compare_cli_defaults_match_top20_evaluate_at_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare_reranking.py"])

    args = compare_reranking._parse_args()

    assert args.candidate_count == 20
    assert args.window_size == 20
    assert args.stride == 10
    assert args.top_k == 5


def test_compare_live_invalid_output_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        top_k=5,
        checkpoint_output=None,
    )
    case = BenchmarkCase(
        fixture_id="fixture",
        dataset="dataset",
        source="source",
        license="license",
        query_id="q1",
        query="query",
        documents=(
            BenchmarkDocument(id="d1", title="", text=""),
            BenchmarkDocument(id="d2", title="", text=""),
        ),
        qrels={"d1": 1},
    )

    def raise_parse_error(**kwargs: object) -> tuple[str, ...]:
        del kwargs
        raise RerankParseError("ranking must contain exactly 20 items")

    monkeypatch.setattr(compare_reranking, "_rank_case", raise_parse_error)

    evaluations, per_query = compare_reranking._evaluate_cases(
        args=args,
        algorithms=("single_call_listwise@20",),
        cases=(case,),
    )
    aggregate = compare_reranking._aggregate_with_validity(evaluations, per_query)

    assert per_query[0]["valid"] is False
    assert per_query[0]["error_type"] == "RerankParseError"
    assert aggregate[0]["invalid_rate"] == 1.0


def test_compare_checkpoint_output_writes_per_query_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "query_results.jsonl"
    args = argparse.Namespace(
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        top_k=5,
        checkpoint_output=checkpoint_path,
    )
    case = BenchmarkCase(
        fixture_id="fixture",
        dataset="dataset",
        source="source",
        license="license",
        query_id="q1",
        query="query",
        documents=(BenchmarkDocument(id="d1", title="", text=""),),
        qrels={"d1": 1},
    )

    monkeypatch.setattr(
        compare_reranking,
        "_rank_case",
        lambda **kwargs: ("d1",),
    )

    compare_reranking._evaluate_cases(
        args=args,
        algorithms=("original_bm25",),
        cases=(case,),
    )

    assert checkpoint_path.read_text(encoding="utf-8").count("\n") == 1
    assert '"valid": true' in checkpoint_path.read_text(encoding="utf-8")


def test_compare_dataset_name_defaults_to_cache_dir() -> None:
    args = argparse.Namespace(
        dataset="benchmark-cache",
        dataset_name=None,
        cache_dir=Path("/tmp/askubuntu-bm25"),
    )

    assert compare_reranking._dataset_name(args) == "askubuntu-bm25"


def test_compare_filters_cases_by_query_id() -> None:
    cases = [
        BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=(),
            qrels={},
        ),
        BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q2",
            query="query",
            documents=(),
            qrels={},
        ),
    ]

    filtered = compare_reranking._filter_cases_by_query_id(cases, ("q2",))

    assert [case.query_id for case in filtered] == ["q2"]


def test_compare_rejects_unknown_query_id() -> None:
    cases = [
        BenchmarkCase(
            fixture_id="fixture",
            dataset="dataset",
            source="source",
            license="license",
            query_id="q1",
            query="query",
            documents=(),
            qrels={},
        )
    ]

    with pytest.raises(SystemExit, match="--query-id not found"):
        compare_reranking._filter_cases_by_query_id(cases, ("missing",))


def test_compare_beir_scifact_dataset_name_remains_legacy_alias() -> None:
    args = argparse.Namespace(
        dataset="beir-scifact",
        dataset_name=None,
        cache_dir=Path("/tmp/cache"),
    )

    assert compare_reranking._dataset_name(args) == "BEIR/SciFact"
