from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import pytest

from ranksmith import AzureOpenAIReranker, Document, ListwiseStrategy
from ranksmith._metrics import mrr_at_k, ndcg_at_k, recall_at_k

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reranking_smoke_fixture.jsonl"


class FixtureDocument(TypedDict):
    id: str
    title: str
    text: str


class FixtureCase(TypedDict):
    schema_version: int
    fixture_id: str
    dataset: str
    source: str
    license: str
    query_id: str
    query: str
    documents: list[FixtureDocument]
    qrels: dict[str, int]


class RelevanceProvider:
    def __init__(self, qrels: dict[str, int]) -> None:
        self.qrels = qrels
        self.calls: list[list[str]] = []

    def rank(self, query: str, documents: list[Document]) -> str:
        del query
        ids = [document.id or "" for document in documents]
        self.calls.append(ids)
        ranking = sorted(
            range(len(documents)),
            key=lambda index: (-self.qrels.get(ids[index], 0), index),
        )
        return json.dumps({"ranking": [index + 1 for index in ranking]})


def test_reranking_smoke_fixture_schema_is_valid() -> None:
    cases = _load_fixture_cases()

    assert cases
    for case in cases:
        assert case["schema_version"] == 1
        assert case["fixture_id"]
        assert case["query"]
        assert case["documents"]
        assert case["qrels"]
        document_ids = [document["id"] for document in case["documents"]]
        assert len(document_ids) == len(set(document_ids))
        assert set(case["qrels"]).issubset(document_ids)
        assert all(score > 0 for score in case["qrels"].values())
        assert all(document["title"] for document in case["documents"])
        assert all(document["text"] for document in case["documents"])


def test_metrics_have_known_values() -> None:
    qrels = {"a": 3, "b": 2}

    assert ndcg_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(1.0)
    assert mrr_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(1.0)

    assert ndcg_at_k(["c", "b", "a"], qrels, 3) == pytest.approx(0.6480408150365517)
    assert mrr_at_k(["c", "b", "a"], qrels, 3) == pytest.approx(0.5)
    assert recall_at_k(["c", "b", "a"], qrels, 2) == pytest.approx(0.5)


def test_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError):
        mrr_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 1}, 0)


def test_rankgpt_sliding_window_with_real_fixture_reaches_relevant_docs() -> None:
    for case in _load_fixture_cases():
        provider = RelevanceProvider(case["qrels"])
        reranker = AzureOpenAIReranker(
            api_key="key",
            azure_endpoint="https://example.openai.azure.com",
            azure_deployment="gpt-4o-mini",
            provider=provider,
            strategy=ListwiseStrategy(
                algorithm="rankgpt_sliding_window", window_size=3, stride=2
            ),
        )
        documents = [
            Document(
                id=document["id"],
                text=f"{document['title']}\n\n{document['text']}",
            )
            for document in case["documents"]
        ]

        results = reranker.rerank(case["query"], documents)
        ranked_ids = [result.document.id or "" for result in results]

        assert ndcg_at_k(ranked_ids, case["qrels"], 3) == pytest.approx(1.0)
        assert mrr_at_k(ranked_ids, case["qrels"], 3) == pytest.approx(1.0)
        assert recall_at_k(ranked_ids, case["qrels"], 3) == pytest.approx(1.0)
        assert provider.calls[0] == [
            document["id"] for document in case["documents"][2:5]
        ]


def _load_fixture_cases() -> list[FixtureCase]:
    cases: list[FixtureCase] = []
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        cases.append(cast(FixtureCase, json.loads(line)))
    return cases
