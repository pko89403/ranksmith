from __future__ import annotations

import json

from ranksmith.errors import RerankInputError, RerankParseError

__all__ = ["parse_ranking_response"]


def parse_ranking_response(raw_response: str, *, expected_count: int) -> list[int]:
    """Parse and validate a 1-based ranking permutation from provider JSON."""
    if expected_count < 0:
        raise RerankInputError("expected_count must be greater than or equal to 0")

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RerankParseError("LLM response is not valid JSON.", raw_response) from exc

    ranking = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(ranking, list):
        raise RerankParseError(
            'LLM response must contain a "ranking" list.',
            raw_response,
        )
    if not all(isinstance(item, int) for item in ranking):
        raise RerankParseError("ranking must contain only integers.", raw_response)

    expected = set(range(1, expected_count + 1))
    actual = set(ranking)
    if len(ranking) != expected_count:
        raise RerankParseError(
            f"ranking must contain exactly {expected_count} items.",
            raw_response,
        )
    if actual != expected:
        raise RerankParseError(
            f"ranking must be a permutation of 1..{expected_count}.",
            raw_response,
        )
    return ranking
