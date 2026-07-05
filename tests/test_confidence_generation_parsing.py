from __future__ import annotations

import pytest

from ranksmith.confidence_generation.errors import ConfidenceGenerationParseError
from ranksmith.confidence_generation.parsing import (
    parse_answer_output,
    parse_relevance_output,
)


def test_parse_answer_output_accepts_exact_shape() -> None:
    assert parse_answer_output('{"answer":"Nancy Travis"}') == "Nancy Travis"


def test_parse_answer_output_rejects_duplicate_keys() -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_answer_output('{"answer":"first","answer":"second"}')


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"answer":""}',
        '{"answer":" "}',
        '{"answer":"x","rationale":"extra"}',
    ],
)
def test_parse_answer_output_rejects_invalid_shape(content: str) -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_answer_output(content)


def test_parse_relevance_output_accepts_supported_values() -> None:
    assert parse_relevance_output('{"judgment":"relevant"}') == "relevant"
    assert parse_relevance_output('{"judgment":"not_relevant"}') == "not_relevant"


def test_parse_relevance_output_rejects_duplicate_keys() -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_relevance_output('{"judgment":"relevant","judgment":"not_relevant"}')


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"judgment":"maybe"}',
        '{"judgment":"relevant","confidence":0.9}',
    ],
)
def test_parse_relevance_output_rejects_invalid_shape(content: str) -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_relevance_output(content)
