from __future__ import annotations

from pathlib import Path

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


def test_map_score_uses_binary_relevance() -> None:
    ranking = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 0, "d2": 2, "d3": 0, "d4": 1}

    assert map_score(ranking, qrels) == (1 / 2 + 2 / 4) / 2


def test_map_score_returns_zero_when_no_relevant_documents() -> None:
    assert map_score(["d1", "d2"], {"d1": 0, "d2": 0}) == 0.0


def test_normalize_method_name_accepts_sliding_alias() -> None:
    assert normalize_method_name("sliding@20") == "rankgpt_sliding_window@20"
    assert normalize_method_name("rankgpt_sliding_window@50") == (
        "rankgpt_sliding_window@50"
    )


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
                "method": "direct@20",
            }
        ],
    )

    assert completed_result_keys(path) == {("Task", "test", "q1", "direct@20")}


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
    from importlib.util import module_from_spec, spec_from_file_location

    here = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(
        "evaluate_mteb_reranking", here / "scripts" / "evaluate_mteb_reranking.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    strip = module._strip_matched_quotes

    assert strip('"hello"') == "hello"
    assert strip("'hello'") == "hello"
    assert strip('"a\'b"') == "a'b"
    assert strip('"unbalanced') == '"unbalanced'
    assert strip("no quotes") == "no quotes"
    assert strip('"') == '"'
    assert strip("") == ""


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
