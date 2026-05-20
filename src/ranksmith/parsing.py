from __future__ import annotations

import json

from ranksmith.errors import RerankInputError, RerankParseError

__all__ = ["parse_ranking_response", "parse_selection_response"]


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


def parse_selection_response(
    raw_response: str,
    *,
    expected_count: int,
    selected_count: int,
) -> list[int]:
    """Parse and validate 1-based selected indexes from provider JSON."""
    if expected_count < 0:
        raise RerankInputError("expected_count must be greater than or equal to 0")
    if selected_count < 0:
        raise RerankInputError("selected_count must be greater than or equal to 0")
    if selected_count > expected_count:
        raise RerankInputError(
            "selected_count must be less than or equal to expected_count"
        )

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RerankParseError("LLM response is not valid JSON.", raw_response) from exc

    selected = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(selected, list):
        raise RerankParseError(
            'LLM response must contain a "selected" list.',
            raw_response,
        )
    if not all(isinstance(item, int) for item in selected):
        raise RerankParseError("selected must contain only integers.", raw_response)
    if len(selected) != selected_count:
        raise RerankParseError(
            f"selected must contain exactly {selected_count} items.",
            raw_response,
        )
    if len(set(selected)) != len(selected):
        raise RerankParseError(
            "selected must not contain duplicate items.",
            raw_response,
        )

    valid_range = set(range(1, expected_count + 1))
    if not set(selected).issubset(valid_range):
        raise RerankParseError(
            f"selected must contain values in 1..{expected_count}.",
            raw_response,
        )
    return selected
