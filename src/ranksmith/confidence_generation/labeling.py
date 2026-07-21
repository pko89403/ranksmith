from __future__ import annotations

import math
import re
from typing import Literal

from ranksmith.confidence_generation.errors import ConfidenceGenerationInputError

JudgmentValue = Literal["relevant", "not_relevant"]


def normalized_exact_match(
    answer: str,
    gold_answer: str | list[str],
    *,
    no_answer_value: str = "__NO_ANSWER__",
) -> bool:
    if not isinstance(no_answer_value, str) or not no_answer_value.strip():
        raise ConfidenceGenerationInputError(
            "no_answer_value must be a non-empty string"
        )
    normalized_answer = _normalize(answer)
    if normalized_answer == _normalize(no_answer_value):
        return False

    candidates = [gold_answer] if isinstance(gold_answer, str) else gold_answer
    return any(normalized_answer == _normalize(candidate) for candidate in candidates)


def relevance_truth(
    value: int | float | bool,
    *,
    threshold: float,
    operator: str,
) -> JudgmentValue:
    if operator not in {"gt", "gte"}:
        raise ConfidenceGenerationInputError('operator must be "gt" or "gte"')
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        raise ConfidenceGenerationInputError("threshold must be a finite number")

    if isinstance(value, bool):
        return "relevant" if value else "not_relevant"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfidenceGenerationInputError("relevance_label must be numeric or bool")
    if operator == "gt":
        return "relevant" if value > threshold else "not_relevant"
    return "relevant" if value >= threshold else "not_relevant"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
