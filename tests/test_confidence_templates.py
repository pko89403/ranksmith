from __future__ import annotations

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceInputError,
    JudgmentConfidenceInput,
)
from ranksmith.confidence._templates import (
    format_confidence_input,
)


def test_formats_answer_confidence_template() -> None:
    text = format_confidence_input(
        "answer_confidence",
        AnswerConfidenceInput(context="passage", answer="answer"),
    )

    assert text == "Context:\npassage\n\nAnswer:\nanswer"


def test_formats_judgment_confidence_template() -> None:
    text = format_confidence_input(
        "judgment_confidence",
        JudgmentConfidenceInput(
            query="query",
            document="document",
            judgment="direct evidence",
        ),
    )

    assert text == "Query:\nquery\n\nDocument:\ndocument\n\nJudgment:\ndirect evidence"


def test_rejects_mismatched_input_type() -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(
            "answer_confidence",
            JudgmentConfidenceInput(
                query="query",
                document="document",
                judgment="direct evidence",
            ),
        )


def test_rejects_whitespace_required_field() -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(
            "answer_confidence",
            AnswerConfidenceInput(context="  ", answer="answer"),
        )
