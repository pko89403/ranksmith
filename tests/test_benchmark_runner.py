from __future__ import annotations

from pathlib import Path

import pytest

from ranksmith._benchmark import (
    aggregate_evaluations,
    aggregate_to_dict,
    evaluate_ranked_ids,
    load_beir_cases,
)


def test_load_beir_cases_with_candidate_file(tmp_path: Path) -> None:
    _write_beir_cache(tmp_path)
    candidates_path = tmp_path / "candidates.tsv"
    candidates_path.write_text(
        "query_id\tdocument_id\trank\nq1\td2\t1\nq1\td1\t2\nq1\td3\t3\n",
        encoding="utf-8",
    )

    cases = load_beir_cases(
        tmp_path,
        split="test",
        candidates_path=candidates_path,
    )

    assert len(cases) == 1
    assert cases[0].query_id == "q1"
    assert [document.id for document in cases[0].documents] == ["d2", "d1", "d3"]
    assert cases[0].qrels == {"d1": 2, "d2": 1}


def test_load_beir_cases_requires_candidate_file(tmp_path: Path) -> None:
    _write_beir_cache(tmp_path)

    with pytest.raises(ValueError, match="requires --candidates"):
        load_beir_cases(tmp_path, split="test", candidates_path=None)


def test_load_beir_cases_rejects_unknown_candidate_document(tmp_path: Path) -> None:
    _write_beir_cache(tmp_path)
    candidates_path = tmp_path / "candidates.tsv"
    candidates_path.write_text("q1\tmissing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found in corpus"):
        load_beir_cases(
            tmp_path,
            split="test",
            candidates_path=candidates_path,
        )


def test_diagnostic_candidate_strategy_is_explicit(tmp_path: Path) -> None:
    _write_beir_cache(tmp_path)

    cases = load_beir_cases(
        tmp_path,
        split="test",
        candidates_path=None,
        candidate_strategy="oracle_plus_random",
        candidate_count=3,
        seed=7,
    )

    assert {document.id for document in cases[0].documents} >= {"d1", "d2"}
    assert len(cases[0].documents) == 3


def test_aggregate_evaluations_uses_macro_average(tmp_path: Path) -> None:
    _write_beir_cache(tmp_path)
    candidates_path = tmp_path / "candidates.tsv"
    candidates_path.write_text("q1\td3\nq1\td2\nq1\td1\n", encoding="utf-8")
    case = load_beir_cases(
        tmp_path,
        split="test",
        candidates_path=candidates_path,
    )[0]

    evaluation = evaluate_ranked_ids(
        case=case,
        algorithm="direct",
        ranked_ids=["d3", "d2", "d1"],
        top_k=3,
    )
    aggregate = aggregate_evaluations([evaluation])[0]

    assert evaluation.metrics["mrr@3"] == pytest.approx(0.5)
    assert aggregate.case_count == 1
    assert aggregate.metrics["mrr@3"] == pytest.approx(0.5)
    assert aggregate_to_dict(aggregate)["algorithm"] == "direct"


def _write_beir_cache(root: Path) -> None:
    (root / "qrels").mkdir()
    (root / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"Relevant A","text":"First relevant document."}\n'
        '{"_id":"d2","title":"Relevant B","text":"Second relevant document."}\n'
        '{"_id":"d3","title":"Distractor","text":"Distractor document."}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id":"q1","text":"Which documents are relevant?"}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\n",
        encoding="utf-8",
    )
