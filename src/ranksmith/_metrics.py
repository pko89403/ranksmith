from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    gains = [_relevance(document_id, qrels) for document_id in ranking[:k]]
    relevant_scores = (score for score in qrels.values() if score > 0)
    ideal_gains = sorted(relevant_scores, reverse=True)[:k]
    ideal = _dcg(ideal_gains)
    if ideal == 0:
        return 0.0
    return _dcg(gains) / ideal


def mrr_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    for index, document_id in enumerate(ranking[:k], start=1):
        if _relevance(document_id, qrels) > 0:
            return 1.0 / index
    return 0.0


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    relevant_documents = {
        document_id for document_id, score in qrels.items() if score > 0
    }
    if not relevant_documents:
        return 0.0
    retrieved_relevant = {
        document_id for document_id in ranking[:k] if document_id in relevant_documents
    }
    return len(retrieved_relevant) / len(relevant_documents)


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(index + 1) for index, gain in enumerate(gains, start=1))


def _relevance(document_id: str, qrels: Mapping[str, int]) -> int:
    return qrels.get(document_id, 0)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be greater than 0")
