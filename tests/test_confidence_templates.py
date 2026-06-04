from __future__ import annotations

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    TaskType,
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


def test_formats_query_answerability_confidence_template() -> None:
    text = format_confidence_input(
        "query_answerability_confidence",
        QueryAnswerabilityConfidenceInput(query="Who?", answer="Nancy Travis"),
    )

    assert text == "Query:\nWho?\n\nAnswer:\nNancy Travis"


def test_formats_query_context_answerability_confidence_template() -> None:
    text = format_confidence_input(
        "query_context_answerability_confidence",
        QueryContextAnswerabilityConfidenceInput(
            query="Who?",
            context="Karen was played by Nancy Travis.",
            answer="Nancy Travis",
        ),
    )

    assert text == (
        "Query:\nWho?\n\nContext:\nKaren was played by Nancy Travis."
        "\n\nAnswer:\nNancy Travis"
    )


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


@pytest.mark.parametrize(
    ("task_type", "item"),
    [
        (
            "query_answerability_confidence",
            QueryContextAnswerabilityConfidenceInput(
                query="Who?",
                context="Karen was played by Nancy Travis.",
                answer="Nancy Travis",
            ),
        ),
        (
            "query_context_answerability_confidence",
            QueryAnswerabilityConfidenceInput(query="Who?", answer="Nancy Travis"),
        ),
    ],
)
def test_rejects_mismatched_answerability_input_type(
    task_type: TaskType,
    item: QueryAnswerabilityConfidenceInput | QueryContextAnswerabilityConfidenceInput,
) -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(task_type, item)


def test_rejects_whitespace_required_field() -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(
            "answer_confidence",
            AnswerConfidenceInput(context="  ", answer="answer"),
        )


@pytest.mark.parametrize(
    ("task_type", "item"),
    [
        (
            "query_answerability_confidence",
            QueryAnswerabilityConfidenceInput(query=" ", answer="Nancy Travis"),
        ),
        (
            "query_answerability_confidence",
            QueryAnswerabilityConfidenceInput(query="Who?", answer="\t"),
        ),
        (
            "query_context_answerability_confidence",
            QueryContextAnswerabilityConfidenceInput(
                query=" ",
                context="Karen was played by Nancy Travis.",
                answer="Nancy Travis",
            ),
        ),
        (
            "query_context_answerability_confidence",
            QueryContextAnswerabilityConfidenceInput(
                query="Who?",
                context="\n",
                answer="Nancy Travis",
            ),
        ),
        (
            "query_context_answerability_confidence",
            QueryContextAnswerabilityConfidenceInput(
                query="Who?",
                context="Karen was played by Nancy Travis.",
                answer="\t",
            ),
        ),
    ],
)
def test_answerability_templates_reject_whitespace_required_fields(
    task_type: TaskType,
    item: QueryAnswerabilityConfidenceInput | QueryContextAnswerabilityConfidenceInput,
) -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(task_type, item)
