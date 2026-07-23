from __future__ import annotations

import pytest

from benchmarks.bm25 import bm25_search, build_bm25_index, tokenize


def test_tokenize_lowercases_and_splits_on_non_alphanumerics() -> None:
    assert tokenize("Vitamin-C deficiency, causes SCURVY!") == [
        "vitamin",
        "c",
        "deficiency",
        "causes",
        "scurvy",
    ]


def test_bm25_ranks_matching_document_first() -> None:
    index = build_bm25_index(
        {
            "d1": "scurvy is caused by vitamin c deficiency",
            "d2": "sleep deprivation weakens the immune system",
            "d3": "apples contain vitamins and fiber",
        }
    )

    results = bm25_search(index, "what causes scurvy", top_k=3)

    assert [document_id for document_id, _ in results][0] == "d1"
    assert all(score > 0 for _, score in results)


def test_bm25_omits_documents_without_query_terms() -> None:
    index = build_bm25_index(
        {
            "d1": "scurvy is caused by vitamin c deficiency",
            "d2": "sleep deprivation weakens the immune system",
        }
    )

    results = bm25_search(index, "scurvy", top_k=5)

    assert [document_id for document_id, _ in results] == ["d1"]


def test_bm25_returns_empty_for_out_of_vocabulary_query() -> None:
    index = build_bm25_index({"d1": "alpha beta"})

    assert bm25_search(index, "zzz", top_k=3) == []


def test_bm25_excludes_requested_documents() -> None:
    index = build_bm25_index(
        {
            "d1": "vitamin c deficiency",
            "d2": "vitamin c deficiency",
        }
    )

    results = bm25_search(index, "vitamin c", top_k=2, exclude=("d1",))

    assert [document_id for document_id, _ in results] == ["d2"]


def test_bm25_breaks_score_ties_by_document_id() -> None:
    index = build_bm25_index(
        {
            "d2": "vitamin c deficiency",
            "d1": "vitamin c deficiency",
        }
    )

    results = bm25_search(index, "vitamin", top_k=2)

    assert [document_id for document_id, _ in results] == ["d1", "d2"]
    assert results[0][1] == results[1][1]


def test_bm25_respects_top_k() -> None:
    index = build_bm25_index({f"d{i}": f"vitamin document {i}" for i in range(10)})

    assert len(bm25_search(index, "vitamin", top_k=4)) == 4


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_bm25_index({})


def test_bm25_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k1"):
        build_bm25_index({"d1": "text"}, k1=0)
    with pytest.raises(ValueError, match="b"):
        build_bm25_index({"d1": "text"}, b=1.5)
    index = build_bm25_index({"d1": "text"})
    with pytest.raises(ValueError, match="top_k"):
        bm25_search(index, "text", top_k=0)
