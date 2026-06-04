from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ranksmith.confidence_generation._errors import ConfidenceGenerationParseError
from ranksmith.confidence_generation._labeling import JudgmentValue


def parse_answer_output(content: str) -> str:
    data = _parse_json_object(content)
    _require_exact_keys(data, {"answer"})
    answer = data["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ConfidenceGenerationParseError("answer must be a non-empty string")
    return answer


def parse_relevance_output(content: str) -> JudgmentValue:
    data = _parse_json_object(content)
    _require_exact_keys(data, {"judgment"})
    judgment = data["judgment"]
    if judgment == "relevant":
        return "relevant"
    if judgment == "not_relevant":
        return "not_relevant"
    raise ConfidenceGenerationParseError(
        'judgment must be "relevant" or "not_relevant"'
    )


def _parse_json_object(content: str) -> Mapping[str, object]:
    if content == "":
        raise ConfidenceGenerationParseError("model output must not be empty")
    try:
        value: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConfidenceGenerationParseError("model output must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ConfidenceGenerationParseError("model output must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_exact_keys(data: Mapping[str, object], expected: set[str]) -> None:
    keys = set(data)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    if missing:
        raise ConfidenceGenerationParseError(
            f"model output missing required field: {missing[0]}"
        )
    unexpected = sorted(keys - expected)
    raise ConfidenceGenerationParseError(
        f"model output has unexpected field: {unexpected[0]}"
    )
