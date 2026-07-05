from __future__ import annotations

from ranksmith.confidence.errors import ConfidenceInputError
from ranksmith.confidence.types import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    StructuralConfidenceInput,
    TaskType,
)

INPUT_TEMPLATE_VERSION = "structural-template-v1"


def _require_non_empty(value: str, *, field_name: str) -> str:
    if value.strip() == "":
        raise ConfidenceInputError(f"{field_name} must not be empty")
    return value


def format_confidence_input(
    task_type: TaskType,
    item: StructuralConfidenceInput,
) -> str:
    if task_type == "answer_confidence":
        if not isinstance(item, AnswerConfidenceInput):
            raise ConfidenceInputError(
                "answer_confidence requires AnswerConfidenceInput"
            )
        context = _require_non_empty(item.context, field_name="context")
        answer = _require_non_empty(item.answer, field_name="answer")
        return f"Context:\n{context}\n\nAnswer:\n{answer}"

    if task_type == "judgment_confidence":
        if not isinstance(item, JudgmentConfidenceInput):
            raise ConfidenceInputError(
                "judgment_confidence requires JudgmentConfidenceInput"
            )
        query = _require_non_empty(item.query, field_name="query")
        document = _require_non_empty(item.document, field_name="document")
        judgment = _require_non_empty(item.judgment, field_name="judgment")
        return f"Query:\n{query}\n\nDocument:\n{document}\n\nJudgment:\n{judgment}"

    raise ConfidenceInputError(f"unsupported task_type: {task_type!r}")
