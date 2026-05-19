from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from ranksmith._metrics import map_score, mrr_at_k, ndcg_at_k, recall_at_k

T = TypeVar("T")


@dataclass(frozen=True)
class MtebRerankingCandidate:
    doc_id: str
    text: str
    label: float


@dataclass(frozen=True)
class MtebRerankingSample:
    task_name: str
    split: str
    query_id: str
    query: str
    candidates: tuple[MtebRerankingCandidate, ...]


@dataclass(frozen=True)
class ParsedRanking:
    ranking: tuple[int, ...]
    valid: bool
    failure_type: str | None


@dataclass(frozen=True)
class PriceConfig:
    input_token_price_per_1m: float
    output_token_price_per_1m: float


def normalize_method_name(method: str) -> str:
    if method == "original":
        return method
    if method.startswith("rankgpt_sliding_window@"):
        _parse_positive_suffix(method, "rankgpt_sliding_window@")
        return method
    if method.startswith("prp_sliding_k@"):
        _parse_positive_suffix(method, "prp_sliding_k@")
        return method
    raise ValueError(
        "method must be original, rankgpt_sliding_window@N, or prp_sliding_k@N"
    )


def parse_ranking_with_failure_type(
    raw_response: str,
    expected_count: int,
) -> ParsedRanking:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return ParsedRanking((), False, "json_parse_failure")
    if not isinstance(data, dict) or "ranking" not in data:
        return ParsedRanking((), False, "missing_ranking")
    ranking = data["ranking"]
    if not isinstance(ranking, list):
        return ParsedRanking((), False, "missing_ranking")
    if not all(isinstance(item, int) for item in ranking):
        return ParsedRanking((), False, "non_integer_rank")
    parsed = tuple(ranking)
    if len(parsed) != expected_count:
        return ParsedRanking(parsed, False, "length_mismatch")
    if any(rank < 1 or rank > expected_count for rank in parsed):
        return ParsedRanking(parsed, False, "out_of_range_rank")
    if len(set(parsed)) != len(parsed):
        return ParsedRanking(parsed, False, "duplicate_rank")
    return ParsedRanking(parsed, True, None)


def apply_integer_permutation(
    items: Sequence[T],
    ranking: Sequence[int],
) -> tuple[T, ...]:
    return tuple(items[index - 1] for index in ranking)


def rankgpt_window_ranges(
    *,
    document_count: int,
    rank_start: int,
    rank_end: int,
    window_size: int,
    step: int,
) -> tuple[tuple[int, int], ...]:
    if document_count < 1:
        return ()
    if rank_start < 0 or rank_end < 1 or window_size < 1 or step < 1:
        raise ValueError("rank_start, rank_end, window_size, and step are invalid")
    if step > window_size:
        raise ValueError("step must be less than or equal to window_size")
    if rank_end <= rank_start:
        raise ValueError("rank_end must be greater than rank_start")

    effective_end = min(rank_end, document_count)
    ranges: list[tuple[int, int]] = []
    end = effective_end
    max_iterations = (effective_end - rank_start) // step + 2
    for _ in range(max_iterations):
        start = max(rank_start, end - window_size)
        ranges.append((start, end))
        if start == rank_start:
            return tuple(ranges)
        end -= step
    raise RuntimeError("rankgpt_window_ranges exceeded iteration cap")


def compute_query_metrics(
    *,
    sample: MtebRerankingSample,
    ranked_doc_ids: Sequence[str] | None,
    valid: bool,
) -> dict[str, float]:
    if not valid or ranked_doc_ids is None:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "map": 0.0, "recall@10": 0.0}
    qrels = {candidate.doc_id: int(candidate.label) for candidate in sample.candidates}
    binary_qrels = {
        candidate.doc_id: 1 if candidate.label > 0 else 0
        for candidate in sample.candidates
    }
    return {
        "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels, 10),
        "mrr@10": mrr_at_k(ranked_doc_ids, binary_qrels, 10),
        "map": map_score(ranked_doc_ids, binary_qrels),
        "recall@10": recall_at_k(ranked_doc_ids, binary_qrels, 10),
    }


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Nearest-rank percentile (not NumPy-style linear interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((percentile_value / 100) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from string parts, immune to PYTHONHASHSEED."""
    key = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def estimate_cost(
    usage: tuple[int, int] | None,
    price_config: PriceConfig | None,
) -> float | None:
    if usage is None or price_config is None:
        return None
    input_tokens, output_tokens = usage
    return (
        input_tokens / 1_000_000 * price_config.input_token_price_per_1m
        + output_tokens / 1_000_000 * price_config.output_token_price_per_1m
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def completed_result_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        row = json.loads(line)
        completed.add(
            (
                str(row["task"]),
                str(row["split"]),
                str(row["query_id"]),
                str(row["method"]),
            )
        )
    return completed


def _parse_positive_suffix(method: str, prefix: str) -> int:
    suffix = method.removeprefix(prefix)
    if suffix == "":
        raise ValueError(f"{method} is missing a numeric suffix")
    value = int(suffix)
    if value < 1:
        raise ValueError(f"{method} suffix must be greater than 0")
    return value
