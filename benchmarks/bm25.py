"""Pure-Python Okapi BM25 for building first-stage candidate runs.

This exists so benchmark tooling can produce BM25 hard negatives without an
external dependency such as Pyserini. Defaults match Pyserini's Lucene BM25
(k1=0.9, b=0.4) so runs stay comparable with the documented
``bm25_top20_reranking`` setup; the tokenizer is a simple lowercase
alphanumeric split, so absolute scores are not identical to Lucene's.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass

DEFAULT_K1 = 0.9
DEFAULT_B = 0.4

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Index:
    k1: float
    b: float
    document_ids: tuple[str, ...]
    document_lengths: tuple[int, ...]
    average_document_length: float
    # term -> ((document position, term frequency), ...)
    postings: Mapping[str, tuple[tuple[int, int], ...]]
    inverse_document_frequency: Mapping[str, float]


def build_bm25_index(
    documents: Mapping[str, str],
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25Index:
    if not documents:
        raise ValueError("documents must not be empty")
    if k1 <= 0:
        raise ValueError("k1 must be greater than 0")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")

    document_ids = tuple(sorted(documents))
    document_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, document_id in enumerate(document_ids):
        tokens = tokenize(documents[document_id])
        document_lengths.append(len(tokens))
        for term, frequency in sorted(Counter(tokens).items()):
            postings[term].append((position, frequency))

    total_documents = len(document_ids)
    inverse_document_frequency = {
        # Lucene-style BM25 idf: always positive, robust for df close to N.
        term: math.log(
            1.0
            + (total_documents - len(term_postings) + 0.5) / (len(term_postings) + 0.5)
        )
        for term, term_postings in postings.items()
    }
    return BM25Index(
        k1=k1,
        b=b,
        document_ids=document_ids,
        document_lengths=tuple(document_lengths),
        average_document_length=sum(document_lengths) / total_documents,
        postings={
            term: tuple(term_postings) for term, term_postings in postings.items()
        },
        inverse_document_frequency=inverse_document_frequency,
    )


def bm25_search(
    index: BM25Index,
    query: str,
    *,
    top_k: int,
    exclude: Collection[str] = (),
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(document_id, score)`` pairs, best first.

    Documents sharing no term with the query are never returned, so the
    result can be shorter than ``top_k`` (or empty). Ties break on document
    id so runs are deterministic.
    """
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")
    excluded_ids = frozenset(exclude)
    scores: dict[int, float] = defaultdict(float)
    average_length = index.average_document_length
    for term in tokenize(query):
        term_postings = index.postings.get(term)
        if term_postings is None:
            continue
        idf = index.inverse_document_frequency[term]
        for position, frequency in term_postings:
            length_norm = (
                1.0
                - index.b
                + index.b * (index.document_lengths[position] / average_length)
            )
            scores[position] += (
                idf
                * frequency
                * (index.k1 + 1.0)
                / (frequency + index.k1 * length_norm)
            )
    ranked = sorted(
        (
            (index.document_ids[position], score)
            for position, score in scores.items()
            if index.document_ids[position] not in excluded_ids
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked[:top_k]
