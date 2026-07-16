from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from benchmarks.benchmark import BenchmarkCase, BenchmarkDocument
from benchmarks.mteb_eval import tourrank_stage_configs_for_candidate_count
from ranksmith import RerankParseError

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
    args = argparse.Namespace(algorithm=["all"])
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
        "setwise_hs_s10",
        "prp_sliding_p1",
    )


def test_compare_all_is_stable_for_100_candidate_cases() -> None:
    args = argparse.Namespace(algorithm=["all"])
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
        "setwise_hs_s10",
        "prp_sliding_p1",
    )


def test_compare_optional_prp_sliding_p3_is_preserved() -> None:
    args = argparse.Namespace(algorithm=["prp_sliding_p3"])
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("prp_sliding_p3",)


def test_compare_optional_prp_sliding_p1_is_preserved() -> None:
    args = argparse.Namespace(algorithm=["prp_sliding_p1"])
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("prp_sliding_p1",)


def test_compare_explicit_tourrank_is_preserved_for_non_100_candidate_cases() -> None:
    args = argparse.Namespace(algorithm=["tourrank_r"])
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("tourrank_r",)


def test_compare_explicit_setwise_heapsort_is_preserved() -> None:
    args = argparse.Namespace(algorithm=["setwise_heapsort"])
    cases: list[BenchmarkCase] = []

    assert compare_reranking._selected_algorithms(args, cases) == ("setwise_heapsort",)


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
    args = argparse.Namespace(algorithm=["acurank"])
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
            20, "setwise_hs_s10", window_size=20, stride=10, top_k=5
        )
        == 12
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
    assert (
        compare_reranking._estimate_provider_calls(
            100, "setwise_hs_s10", window_size=20, stride=10, top_k=5
        )
        == 26
    )


def test_compare_report_records_method_settings() -> None:
    args = argparse.Namespace(
        dataset="benchmark-cache",
        dataset_name="askubuntu-bm25",
        fixture=Path("fixture.jsonl"),
        cache_dir=Path(".benchmark-cache/askubuntu-bm25"),
        candidates=Path("benchmark-results/pyserini/askubuntu-bm25-top20.trec"),
        candidate_strategy="candidate_file",
        candidate_count=20,
        seed=13,
        top_k=5,
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        set_size=3,
        query_id=[],
        timeout=None,
        checkpoint_output=None,
    )

    report = compare_reranking._build_report(
        args=args,
        algorithms=("setwise_hs_s10",),
        cases=(),
        call_estimates={"setwise_hs_s10": 0},
        per_query=(),
        aggregate=(),
    )

    assert report["method_settings"] == {
        "setwise_hs_s10": {
            "candidate_count": 20,
            "set_size": 10,
            "top_k": 5,
            "top_k_early_stop": True,
        }
    }


def test_compare_setwise_heapsort_uses_set_size(
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
        algorithm="setwise_heapsort",
        window_size=3,
        stride=2,
        passes=10,
        tourrank_rounds=2,
        set_size=5,
    )

    assert captured["strategy"].set_size == 5


def test_compare_setwise_hs_s10_uses_fixed_set_size(
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
        algorithm="setwise_hs_s10",
        window_size=3,
        stride=2,
        passes=10,
        tourrank_rounds=2,
        set_size=5,
    )

    assert captured["strategy"].set_size == 10


def test_compare_setwise_heapsort_forwards_top_k(
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
            *,
            top_k: int | None = None,
        ) -> list[object]:
            del query
            captured["top_k"] = top_k
            return [
                type("Result", (), {"document": document})()
                for document in documents[: top_k or len(documents)]
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
        algorithm="setwise_hs_s10",
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        top_k=5,
    )

    assert captured["top_k"] == 5


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
    assert args.set_size == 3
    assert args.top_k == 5


def test_compare_live_invalid_output_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        set_size=3,
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
        set_size=3,
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


def test_compare_default_algorithm_selection_is_the_default_set() -> None:
    args = argparse.Namespace(algorithm=None)

    algorithms = compare_reranking._selected_algorithms(args, [])

    assert algorithms == compare_reranking.ALGORITHMS


def test_compare_repeated_algorithm_flags_run_in_one_invocation() -> None:
    args = argparse.Namespace(
        algorithm=["original_bm25", "single_call_listwise@20", "answer_confidence"]
    )

    algorithms = compare_reranking._selected_algorithms(args, [])

    assert algorithms == (
        "original_bm25",
        "single_call_listwise@20",
        "answer_confidence",
    )


def test_compare_rejects_all_combined_with_other_algorithms() -> None:
    args = argparse.Namespace(algorithm=["all", "answer_confidence"])

    with pytest.raises(SystemExit, match="cannot be combined"):
        compare_reranking._selected_algorithms(args, [])


def test_compare_rejects_duplicate_algorithm_flags() -> None:
    args = argparse.Namespace(algorithm=["tourrank_r2", "tourrank_r2"])

    with pytest.raises(SystemExit, match="must not repeat"):
        compare_reranking._selected_algorithms(args, [])


def _answer_confidence_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "window_size": 20,
        "stride": 10,
        "passes": 10,
        "tourrank_rounds": 2,
        "set_size": 3,
        "top_k": 5,
        "candidate_count": 20,
        "max_cases": None,
        "timeout": None,
        "checkpoint_output": None,
        "output": None,
        "dataset": "fixture",
        "cache_dir": None,
        "dataset_name": None,
        "candidates": None,
        "candidate_strategy": "candidate_file",
        "algorithm": ["answer_confidence"],
        "answer_confidence_artifact": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_compare_answer_confidence_requires_artifact_before_running() -> None:
    args = _answer_confidence_args()

    with pytest.raises(SystemExit, match="requires"):
        compare_reranking._validate_args(args)


def test_compare_answer_confidence_requires_existing_artifact_file() -> None:
    args = _answer_confidence_args(
        answer_confidence_artifact="/nonexistent/scorer.joblib"
    )

    with pytest.raises(SystemExit, match="does not exist"):
        compare_reranking._validate_args(args)
