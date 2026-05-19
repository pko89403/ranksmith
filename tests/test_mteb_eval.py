from __future__ import annotations

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ranksmith._metrics import map_score
from ranksmith._mteb_eval import (
    MtebRerankingCandidate,
    MtebRerankingSample,
    PriceConfig,
    apply_integer_permutation,
    completed_result_keys,
    compute_query_metrics,
    estimate_cost,
    normalize_method_name,
    parse_ranking_with_failure_type,
    percentile,
    rankgpt_window_ranges,
    stable_seed,
    write_jsonl,
)


def _load_mteb_cli_module() -> Any:
    here = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(
        "evaluate_mteb_reranking", here / "scripts" / "evaluate_mteb_reranking.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_map_score_uses_binary_relevance() -> None:
    ranking = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 0, "d2": 2, "d3": 0, "d4": 1}

    assert map_score(ranking, qrels) == (1 / 2 + 2 / 4) / 2


def test_map_score_returns_zero_when_no_relevant_documents() -> None:
    assert map_score(["d1", "d2"], {"d1": 0, "d2": 0}) == 0.0


def test_normalize_method_name_accepts_rankgpt_sliding_window() -> None:
    assert normalize_method_name("rankgpt_sliding_window@50") == (
        "rankgpt_sliding_window@50"
    )


def test_normalize_method_name_rejects_removed_sliding_alias() -> None:
    with pytest.raises(ValueError, match="rankgpt_sliding_window@N"):
        normalize_method_name("sliding@20")


def test_normalize_method_name_rejects_removed_direct_method() -> None:
    with pytest.raises(ValueError, match="rankgpt_sliding_window@N"):
        normalize_method_name("direct@20")


def test_normalize_method_name_accepts_prp_sliding_k() -> None:
    assert normalize_method_name("prp_sliding_k@20") == "prp_sliding_k@20"


def test_parse_ranking_with_failure_type_reports_duplicate() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [1, 1, 2]}', 3)

    assert not parsed.valid
    assert parsed.failure_type == "duplicate_rank"


def test_apply_integer_permutation_is_one_based() -> None:
    candidates = ("a", "b", "c")

    assert apply_integer_permutation(candidates, (3, 1, 2)) == ("c", "a", "b")


def test_rankgpt_window_ranges_do_not_repeat_prefix_window() -> None:
    assert rankgpt_window_ranges(
        document_count=20,
        rank_start=0,
        rank_end=20,
        window_size=20,
        step=10,
    ) == ((0, 20),)


def test_compute_query_metrics_uses_zero_score_for_invalid_result() -> None:
    sample = MtebRerankingSample(
        task_name="task",
        split="test",
        query_id="q1",
        query="query",
        candidates=(
            MtebRerankingCandidate(doc_id="d1", text="a", label=1.0),
            MtebRerankingCandidate(doc_id="d2", text="b", label=0.0),
        ),
    )

    metrics = compute_query_metrics(sample=sample, ranked_doc_ids=None, valid=False)

    assert metrics == {"ndcg@10": 0.0, "mrr@10": 0.0, "map": 0.0, "recall@10": 0.0}


def test_estimate_cost_requires_usage_and_prices() -> None:
    assert estimate_cost(None, PriceConfig(1.0, 2.0)) is None
    assert estimate_cost((1000, 500), None) is None


def test_estimate_cost_uses_price_per_million_tokens() -> None:
    assert estimate_cost((1_000_000, 500_000), PriceConfig(2.0, 8.0)) == 6.0


def test_completed_result_keys_reads_query_results(tmp_path: Path) -> None:
    path = tmp_path / "query_results.jsonl"
    write_jsonl(
        path,
        [
            {
                "task": "Task",
                "split": "test",
                "query_id": "q1",
                "method": "rankgpt_sliding_window@20",
            }
        ],
    )

    assert completed_result_keys(path) == {
        ("Task", "test", "q1", "rankgpt_sliding_window@20")
    }


def test_parse_ranking_json_parse_failure() -> None:
    parsed = parse_ranking_with_failure_type("not json", 3)
    assert not parsed.valid
    assert parsed.failure_type == "json_parse_failure"
    assert parsed.ranking == ()


def test_parse_ranking_missing_ranking_key() -> None:
    parsed = parse_ranking_with_failure_type("{}", 3)
    assert parsed.failure_type == "missing_ranking"


def test_parse_ranking_ranking_not_list() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": "abc"}', 3)
    assert parsed.failure_type == "missing_ranking"


def test_parse_ranking_non_integer_returns_empty_ranking() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [1, "x", 3]}', 3)
    assert parsed.failure_type == "non_integer_rank"
    assert parsed.ranking == ()


def test_parse_ranking_length_mismatch() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [1, 2]}', 3)
    assert parsed.failure_type == "length_mismatch"


def test_parse_ranking_out_of_range_low() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [0, 1, 2]}', 3)
    assert parsed.failure_type == "out_of_range_rank"


def test_parse_ranking_out_of_range_high() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [1, 2, 4]}', 3)
    assert parsed.failure_type == "out_of_range_rank"


def test_parse_ranking_valid_permutation() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [3, 1, 2]}', 3)
    assert parsed.valid
    assert parsed.failure_type is None
    assert parsed.ranking == (3, 1, 2)


def test_rankgpt_window_ranges_validates_rank_end_greater_than_start() -> None:
    with pytest.raises(ValueError):
        rankgpt_window_ranges(
            document_count=10, rank_start=5, rank_end=5, window_size=3, step=2
        )


def test_rankgpt_window_ranges_step_exceeds_window_rejected() -> None:
    with pytest.raises(ValueError):
        rankgpt_window_ranges(
            document_count=10, rank_start=0, rank_end=10, window_size=3, step=5
        )


def test_rankgpt_window_ranges_returns_empty_when_no_documents() -> None:
    assert (
        rankgpt_window_ranges(
            document_count=0, rank_start=0, rank_end=10, window_size=3, step=2
        )
        == ()
    )


def test_rankgpt_window_ranges_clips_to_document_count() -> None:
    ranges = rankgpt_window_ranges(
        document_count=5, rank_start=0, rank_end=20, window_size=4, step=2
    )
    assert ranges[0][1] == 5


def test_rankgpt_window_ranges_back_to_front_with_step() -> None:
    ranges = rankgpt_window_ranges(
        document_count=8, rank_start=0, rank_end=8, window_size=4, step=2
    )
    assert ranges == ((4, 8), (2, 6), (0, 4))


def test_percentile_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == 4.0
    assert percentile([], 50.0) == 0.0


def test_stable_seed_is_deterministic_across_runs() -> None:
    assert stable_seed(13, "Task", "q1") == stable_seed(13, "Task", "q1")
    assert stable_seed(13, "Task", "q1") != stable_seed(13, "Task", "q2")


def test_strip_matched_quotes_symmetric_only() -> None:
    module = _load_mteb_cli_module()
    strip = module._strip_matched_quotes

    assert strip('"hello"') == "hello"
    assert strip("'hello'") == "hello"
    assert strip('"a\'b"') == "a'b"
    assert strip('"unbalanced') == '"unbalanced'
    assert strip("no quotes") == "no quotes"
    assert strip('"') == '"'
    assert strip("") == ""


def test_mteb_samples_fail_on_long_document_without_truncating() -> None:
    module = _load_mteb_cli_module()
    task_dataset = {
        "default": {
            "test": {
                "corpus": [{"id": "d1", "title": "", "text": "too long"}],
                "queries": [{"id": "q1", "text": "query"}],
                "relevant_docs": {"q1": {"d1": 1}},
                "top_ranked": {"q1": ["d1"]},
            }
        }
    }

    with pytest.raises(SystemExit) as error:
        list(
            module._samples_from_task(
                task_name="Task",
                split="test",
                task_dataset=task_dataset,
                max_queries=None,
                max_document_chars=3,
            )
        )

    assert "exceeding --max-document-chars=3" in str(error.value)


def test_mteb_samples_use_qrels_candidates_when_top_ranked_missing() -> None:
    module = _load_mteb_cli_module()
    task_dataset = {
        "default": {
            "test": {
                "corpus": [
                    {"id": "d1", "title": "", "text": "positive"},
                    {"id": "d2", "title": "", "text": "negative"},
                ],
                "queries": [{"id": "q1", "text": "query"}],
                "relevant_docs": {"q1": {"d1": 1, "d2": 0}},
                "top_ranked": None,
            }
        }
    }

    samples = list(
        module._samples_from_task(
            task_name="Task",
            split="test",
            task_dataset=task_dataset,
            max_queries=None,
            max_document_chars=100,
        )
    )

    assert [candidate.doc_id for candidate in samples[0].candidates] == ["d1", "d2"]
    assert [candidate.label for candidate in samples[0].candidates] == [1.0, 0.0]


def test_mteb_sync_llm_method_rejects_prp_method() -> None:
    module = _load_mteb_cli_module()
    sample = MtebRerankingSample(
        task_name="task",
        split="test",
        query_id="q1",
        query="query",
        candidates=(
            MtebRerankingCandidate(doc_id="d1", text="a", label=1.0),
            MtebRerankingCandidate(doc_id="d2", text="b", label=0.0),
            MtebRerankingCandidate(doc_id="d3", text="c", label=0.0),
        ),
    )

    with pytest.raises(ValueError, match="PRP methods require async execution"):
        module._run_llm_method(
            sample=sample,
            method="prp_sliding_k@2",
            rankgpt_window_size=20,
            rankgpt_step=10,
            prp_passes=3,
        )


@pytest.mark.asyncio
async def test_mteb_prp_async_method_uses_async_pairwise_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ranksmith

    module = _load_mteb_cli_module()
    sample = MtebRerankingSample(
        task_name="task",
        split="test",
        query_id="q1",
        query="query",
        candidates=(
            MtebRerankingCandidate(doc_id="d1", text="a", label=1.0),
            MtebRerankingCandidate(doc_id="d2", text="b", label=0.0),
            MtebRerankingCandidate(doc_id="d3", text="c", label=0.0),
        ),
    )
    captured: dict[str, object] = {}

    class FakeAsyncReranker:
        def __init__(self, **kwargs: object) -> None:
            captured["strategy"] = kwargs["strategy"]

        async def rerank(
            self,
            query: str,
            documents: list[ranksmith.Document],
        ) -> list[object]:
            captured["query"] = query
            captured["document_ids"] = [document.id for document in documents]
            return [SimpleNamespace(document=document) for document in documents]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setattr(ranksmith, "AsyncAzureOpenAIReranker", FakeAsyncReranker)

    (
        ranked,
        valid,
        failure,
        usage,
        llm_calls,
        error,
    ) = await module._run_llm_method_async(
        sample=sample,
        method="prp_sliding_k@2",
        rankgpt_window_size=20,
        rankgpt_step=10,
        prp_passes=3,
    )

    assert ranked == ("d1", "d2")
    assert valid is True
    assert failure is None
    assert usage is None
    assert llm_calls == 0
    assert error is None
    assert isinstance(captured["strategy"], ranksmith.AsyncPairwiseStrategy)
    assert captured["strategy"].passes == 3
    assert captured["strategy"].pair_order_parallelism == 2
    assert captured["document_ids"] == ["d1", "d2"]


@pytest.mark.asyncio
async def test_evaluate_pending_methods_respects_concurrency() -> None:
    module = _load_mteb_cli_module()
    samples = [
        MtebRerankingSample(
            task_name="task",
            split="test",
            query_id=f"q{i}",
            query="query",
            candidates=(MtebRerankingCandidate(doc_id=f"d{i}", text="a", label=1.0),),
        )
        for i in range(3)
    ]
    in_flight = 0
    max_in_flight = 0

    async def fake_evaluate(**kwargs: object) -> dict[str, object]:
        nonlocal in_flight, max_in_flight
        sample = kwargs["sample"]
        assert isinstance(sample, MtebRerankingSample)
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {
            "task": sample.task_name,
            "split": sample.split,
            "query_id": sample.query_id,
            "method": "prp_sliding_k@1",
            "metrics": {},
            "valid": True,
            "failure_type": None,
            "error": None,
            "latency_ms": 1.0,
            "usage": None,
            "cost": None,
            "ranked_doc_ids": [sample.candidates[0].doc_id],
        }

    rows = await module._evaluate_pending_methods(
        samples=samples,
        methods=["prp_sliding_k@1"],
        already_done=set(),
        price_config=None,
        rankgpt_window_size=20,
        rankgpt_step=10,
        prp_passes=1,
        concurrency=2,
        evaluate_one=fake_evaluate,
    )

    assert len(rows) == 3
    assert max_in_flight == 2


@pytest.mark.asyncio
async def test_evaluate_pending_methods_creates_only_worker_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mteb_cli_module()
    samples = [
        MtebRerankingSample(
            task_name="task",
            split="test",
            query_id=f"q{i}",
            query="query",
            candidates=(MtebRerankingCandidate(doc_id=f"d{i}", text="a", label=1.0),),
        )
        for i in range(5)
    ]
    created_tasks = 0
    original_create_task = asyncio.create_task

    def counting_create_task(coro: Any) -> asyncio.Task[Any]:
        nonlocal created_tasks
        created_tasks += 1
        return original_create_task(coro)

    async def fake_evaluate(**kwargs: object) -> dict[str, object]:
        sample = kwargs["sample"]
        assert isinstance(sample, MtebRerankingSample)
        await asyncio.sleep(0)
        return {
            "task": sample.task_name,
            "split": sample.split,
            "query_id": sample.query_id,
            "method": "prp_sliding_k@1",
            "metrics": {},
            "valid": True,
            "failure_type": None,
            "error": None,
            "latency_ms": 1.0,
            "usage": None,
            "cost": None,
            "ranked_doc_ids": [sample.candidates[0].doc_id],
        }

    monkeypatch.setattr(module.asyncio, "create_task", counting_create_task)

    rows = await module._evaluate_pending_methods(
        samples=samples,
        methods=["prp_sliding_k@1"],
        already_done=set(),
        price_config=None,
        rankgpt_window_size=20,
        rankgpt_step=10,
        prp_passes=1,
        concurrency=2,
        evaluate_one=fake_evaluate,
    )

    assert len(rows) == 5
    assert created_tasks == 2


def test_mteb_aggregate_includes_llm_call_counts() -> None:
    module = _load_mteb_cli_module()
    rows = [
        {
            "metrics": {"ndcg@10": 1.0},
            "latency_ms": 10.0,
            "valid": True,
            "failure_type": None,
            "usage": [100, 10],
            "cost": 0.01,
            "llm_calls": 9,
        },
        {
            "metrics": {"ndcg@10": 0.0},
            "latency_ms": 20.0,
            "valid": False,
            "failure_type": "llm_output_invalid",
            "usage": [50, 5],
            "cost": 0.005,
            "llm_calls": 3,
        },
    ]

    aggregate = module._aggregate(rows)

    assert aggregate["llm_calls_mean"] == 6.0
    assert aggregate["llm_calls_total"] == 12.0


def test_mteb_result_table_renders_llm_call_counts() -> None:
    module = _load_mteb_cli_module()
    table = module._render_metric_table(
        {
            "prp_sliding_k@20": {
                "ndcg@10": 0.67,
                "ndcg@10_valid_only": 0.67,
                "mrr@10": 0.78,
                "map": 0.61,
                "recall@10": 0.74,
                "latency_ms_p50": 213583.6,
                "latency_ms_p95": 230670.9,
                "invalid_rate": 0.0,
                "llm_calls_mean": 380.0,
                "llm_calls_total": 11400.0,
                "query_count": 30.0,
            }
        }
    )

    rendered = "\n".join(table)

    assert "llm_calls/query" in rendered
    assert "llm_calls_total" in rendered
    assert "380" in rendered
    assert "11400" in rendered


def test_compute_query_metrics_valid_path_uses_graded_qrels() -> None:
    sample = MtebRerankingSample(
        task_name="t",
        split="test",
        query_id="q",
        query="x",
        candidates=(
            MtebRerankingCandidate(doc_id="d1", text="a", label=2.0),
            MtebRerankingCandidate(doc_id="d2", text="b", label=0.0),
        ),
    )
    metrics = compute_query_metrics(
        sample=sample, ranked_doc_ids=("d1", "d2"), valid=True
    )
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@10"] == 1.0
