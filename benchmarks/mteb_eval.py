from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from benchmarks.metrics import map_score, mrr_at_k, ndcg_at_k, recall_at_k
from ranksmith.strategies import TourRankStageConfig

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
class PriceConfig:
    input_token_price_per_1m: float
    output_token_price_per_1m: float


MethodKind = Literal[
    "original",
    "rankgpt_sliding_window",
    "prp_sliding_k",
    "tourrank_r",
    "acurank",
]


@dataclass(frozen=True)
class MethodConfig:
    kind: MethodKind
    candidate_count: int | None
    rounds: int | None
    passes: int | None
    canonical_name: str


def normalize_method_name(method: str) -> str:
    return parse_method_config(method).canonical_name


def parse_method_config(method: str) -> MethodConfig:
    if method == "original":
        return MethodConfig("original", None, None, None, "original")
    if method.startswith("rankgpt_sliding_window@"):
        candidate_count = _parse_positive_suffix(method, "rankgpt_sliding_window@")
        return MethodConfig(
            "rankgpt_sliding_window",
            candidate_count,
            None,
            None,
            f"rankgpt_sliding_window@{candidate_count}",
        )
    if method.startswith("prp_sliding_k@"):
        return _parse_prp_method(method)
    if method.startswith("tourrank_r@"):
        return _parse_tourrank_method(method)
    if method.startswith("acurank@"):
        candidate_count = _parse_positive_suffix(method, "acurank@")
        return MethodConfig(
            "acurank",
            candidate_count,
            None,
            None,
            f"acurank@{candidate_count}",
        )
    raise ValueError(
        "method must be original, rankgpt_sliding_window@N, prp_sliding_k@N, "
        "prp_sliding_k@N:pP, tourrank_r@N:rR, or acurank@N"
    )


def _parse_prp_method(method: str) -> MethodConfig:
    prefix = "prp_sliding_k@"
    suffix = method.removeprefix(prefix)
    count_text, separator, passes_text = suffix.partition(":p")
    if count_text == "" or (separator and passes_text == ""):
        raise ValueError("PRP method must use prp_sliding_k@N or prp_sliding_k@N:pP")
    candidate_count = _parse_positive_int(count_text, method)
    if candidate_count < 1:
        raise ValueError("PRP candidate count must be greater than 0")
    passes = _parse_positive_int(passes_text, method) if separator else None
    if passes is not None and passes < 1:
        raise ValueError("PRP passes must be greater than 0")
    suffix_text = f":p{passes}" if passes is not None else ""
    return MethodConfig(
        "prp_sliding_k",
        candidate_count=candidate_count,
        rounds=None,
        passes=passes,
        canonical_name=f"prp_sliding_k@{candidate_count}{suffix_text}",
    )


def _parse_tourrank_method(method: str) -> MethodConfig:
    prefix = "tourrank_r@"
    if not method.startswith(prefix):
        raise ValueError("TourRank method must use tourrank_r@N:rR")
    suffix = method.removeprefix(prefix)
    count_text, separator, rounds_text = suffix.partition(":r")
    if count_text == "" or (separator and rounds_text == ""):
        raise ValueError("TourRank method must use tourrank_r@N:rR")
    candidate_count = _parse_positive_int(count_text, method)
    if candidate_count < 1:
        raise ValueError("TourRank candidate count must be greater than 0")
    rounds = _parse_positive_int(rounds_text, method) if separator else 2
    if rounds < 1:
        raise ValueError("TourRank rounds must be greater than 0")
    return MethodConfig(
        "tourrank_r",
        candidate_count=candidate_count,
        rounds=rounds,
        passes=None,
        canonical_name=f"tourrank_r@{candidate_count}:r{rounds}",
    )


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


def tourrank_stage_configs_for_candidate_count(
    candidate_count: int,
) -> tuple[TourRankStageConfig, ...]:
    if candidate_count < 2:
        raise ValueError("tourrank_r requires at least 2 candidates")
    if candidate_count == 100:
        return (
            TourRankStageConfig(group_count=5, group_size=20, selected_count=10),
            TourRankStageConfig(group_count=5, group_size=10, selected_count=4),
            TourRankStageConfig(group_count=1, group_size=20, selected_count=10),
            TourRankStageConfig(group_count=1, group_size=10, selected_count=5),
            TourRankStageConfig(group_count=1, group_size=5, selected_count=2),
        )

    stages: list[TourRankStageConfig] = []
    current_count = candidate_count
    while current_count > 1:
        selected_count = max(1, current_count // 2)
        stages.append(
            TourRankStageConfig(
                group_count=1,
                group_size=current_count,
                selected_count=selected_count,
            )
        )
        current_count = selected_count
    return tuple(stages)


def estimate_tourrank_llm_calls(method: str) -> int:
    config = parse_method_config(method)
    if config.kind != "tourrank_r" or config.candidate_count is None:
        raise ValueError("method must be tourrank_r@N:rR")
    if config.rounds is None:
        raise ValueError("TourRank method config is incomplete")
    return config.rounds * sum(
        stage.group_count
        for stage in tourrank_stage_configs_for_candidate_count(config.candidate_count)
    )


def estimate_acurank_llm_calls(method: str, *, window_size: int) -> int:
    config = parse_method_config(method)
    if config.kind != "acurank" or config.candidate_count is None:
        raise ValueError("method must be acurank@N")
    return max(1, math.ceil(config.candidate_count / window_size) + 1)


def estimate_prp_llm_calls(method: str, *, default_passes: int) -> int:
    config = parse_method_config(method)
    if config.kind != "prp_sliding_k" or config.candidate_count is None:
        raise ValueError("method must be prp_sliding_k@N or prp_sliding_k@N:pP")
    passes = config.passes if config.passes is not None else default_passes
    if passes < 1:
        raise ValueError("PRP passes must be greater than 0")
    return 2 * passes * max(config.candidate_count - 1, 0)


def _parse_positive_suffix(method: str, prefix: str) -> int:
    suffix = method.removeprefix(prefix)
    if suffix == "":
        raise ValueError(f"{method} is missing a numeric suffix")
    value = _parse_positive_int(suffix, method)
    if value < 1:
        raise ValueError(f"{method} suffix must be greater than 0")
    return value


def _parse_positive_int(value: str, method: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{method} must contain positive integer values") from exc
    if parsed < 1:
        raise ValueError(f"{method} values must be greater than 0")
    return parsed
