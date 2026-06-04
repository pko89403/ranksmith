from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ranksmith.confidence import TaskType
from ranksmith.confidence_training._errors import (
    ConfidenceDatasetError,
    ConfidenceLabelError,
)
from ranksmith.confidence_training._types import CanonicalConfidenceSample

_ANSWER_REQUIRED = ("id", "context", "answer", "label")
_JUDGMENT_REQUIRED = ("id", "query", "document", "judgment", "label")
_QUERY_ANSWERABILITY_REQUIRED = ("id", "query", "answer", "label")
_QUERY_CONTEXT_ANSWERABILITY_REQUIRED = ("id", "query", "context", "answer", "label")
_ANSWER_ALLOWED = {*_ANSWER_REQUIRED, "gold_answer", "source", "group_id", "metadata"}
_JUDGMENT_ALLOWED = {
    *_JUDGMENT_REQUIRED,
    "relevance_label",
    "source",
    "group_id",
    "metadata",
}
_ANSWERABILITY_OPTIONAL_ALLOWED = {
    "task_type",
    "gold_answer",
    "source",
    "group_id",
    "metadata",
}
_QUERY_ANSWERABILITY_ALLOWED = {
    *_QUERY_ANSWERABILITY_REQUIRED,
    *_ANSWERABILITY_OPTIONAL_ALLOWED,
}
_QUERY_CONTEXT_ANSWERABILITY_ALLOWED = {
    *_QUERY_CONTEXT_ANSWERABILITY_REQUIRED,
    *_ANSWERABILITY_OPTIONAL_ALLOWED,
}


@dataclass(frozen=True)
class _TaskDatasetSchema:
    required: tuple[str, ...]
    allowed: set[str]
    parser: Callable[[Mapping[str, Any]], CanonicalConfidenceSample]


_TASK_SCHEMAS: dict[str, _TaskDatasetSchema] = {
    "answer_confidence": _TaskDatasetSchema(
        required=_ANSWER_REQUIRED,
        allowed=_ANSWER_ALLOWED,
        parser=lambda row: CanonicalConfidenceSample(
            id=_required_text(row, "id"),
            task_type="answer_confidence",
            label=_label(row["label"]),
            context=_required_text(row, "context"),
            answer=_required_text(row, "answer"),
            gold_answer=_optional_gold_answer(row.get("gold_answer")),
            source=_optional_text(row.get("source"), "source"),
            group_id=_optional_text(row.get("group_id"), "group_id"),
            metadata=_metadata(row.get("metadata")),
        ),
    ),
    "judgment_confidence": _TaskDatasetSchema(
        required=_JUDGMENT_REQUIRED,
        allowed=_JUDGMENT_ALLOWED,
        parser=lambda row: CanonicalConfidenceSample(
            id=_required_text(row, "id"),
            task_type="judgment_confidence",
            label=_label(row["label"]),
            query=_required_text(row, "query"),
            document=_required_text(row, "document"),
            judgment=_required_text(row, "judgment"),
            relevance_label=_optional_relevance_label(row.get("relevance_label")),
            source=_optional_text(row.get("source"), "source"),
            group_id=_optional_text(row.get("group_id"), "group_id"),
            metadata=_metadata(row.get("metadata")),
        ),
    ),
    "query_answerability_confidence": _TaskDatasetSchema(
        required=_QUERY_ANSWERABILITY_REQUIRED,
        allowed=_QUERY_ANSWERABILITY_ALLOWED,
        parser=lambda row: CanonicalConfidenceSample(
            id=_required_text(row, "id"),
            task_type="query_answerability_confidence",
            label=_label(row["label"]),
            query=_required_text(row, "query"),
            answer=_required_text(row, "answer"),
            gold_answer=_optional_gold_answer(row.get("gold_answer")),
            source=_optional_text(row.get("source"), "source"),
            group_id=_optional_text(row.get("group_id"), "group_id"),
            metadata=_metadata(row.get("metadata")),
        ),
    ),
    "query_context_answerability_confidence": _TaskDatasetSchema(
        required=_QUERY_CONTEXT_ANSWERABILITY_REQUIRED,
        allowed=_QUERY_CONTEXT_ANSWERABILITY_ALLOWED,
        parser=lambda row: CanonicalConfidenceSample(
            id=_required_text(row, "id"),
            task_type="query_context_answerability_confidence",
            label=_label(row["label"]),
            query=_required_text(row, "query"),
            context=_required_text(row, "context"),
            answer=_required_text(row, "answer"),
            gold_answer=_optional_gold_answer(row.get("gold_answer")),
            source=_optional_text(row.get("source"), "source"),
            group_id=_optional_text(row.get("group_id"), "group_id"),
            metadata=_metadata(row.get("metadata")),
        ),
    ),
}


def load_canonical_dataset(
    path: str | Path,
    *,
    task_type: TaskType,
) -> list[CanonicalConfidenceSample]:
    _validate_task_type(task_type)
    rows = _read_jsonl(Path(path))
    samples: list[CanonicalConfidenceSample] = []
    seen_ids: set[str] = set()
    for line_number, row in rows:
        sample = _parse_row(row, task_type=task_type, line_number=line_number)
        if sample.id in seen_ids:
            raise ConfidenceDatasetError(f"duplicate id: {sample.id}")
        seen_ids.add(sample.id)
        samples.append(sample)
    return samples


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfidenceDatasetError("dataset could not be read") from exc

    rows: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfidenceDatasetError(
                f"dataset line {line_number} must be valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ConfidenceDatasetError(
                f"dataset line {line_number} must be a JSON object"
            )
        rows.append((line_number, value))
    return rows


def _parse_row(
    row: Mapping[str, Any],
    *,
    task_type: TaskType,
    line_number: int,
) -> CanonicalConfidenceSample:
    schema = _TASK_SCHEMAS.get(task_type)
    if schema is None:
        raise ConfidenceDatasetError(f"unsupported task_type: {task_type}")
    try:
        _validate_keys(row, required=schema.required, allowed=schema.allowed)
        _validate_row_task_type(row, task_type=task_type)
        return schema.parser(row)
    except (ConfidenceDatasetError, ConfidenceLabelError) as exc:
        raise type(exc)(f"line {line_number}: {exc}") from exc


def _validate_task_type(task_type: object) -> None:
    if task_type not in _TASK_SCHEMAS:
        raise ConfidenceDatasetError(f"unsupported task_type: {task_type}")


def _validate_keys(
    row: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    allowed: set[str],
) -> None:
    for name in required:
        if name not in row:
            raise ConfidenceDatasetError(f"missing required field: {name}")
    unexpected = sorted(key for key in row if key not in allowed)
    if unexpected:
        raise ConfidenceDatasetError(f"unexpected field for task: {unexpected[0]}")


def _validate_row_task_type(row: Mapping[str, Any], *, task_type: TaskType) -> None:
    value = row.get("task_type")
    if value is None:
        return
    if value != task_type:
        raise ConfidenceDatasetError("task_type field must match requested task_type")


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ConfidenceDatasetError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ConfidenceDatasetError(f"{name} must not be empty")
    return stripped


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfidenceDatasetError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ConfidenceDatasetError(f"{name} must not be empty")
    return stripped


def _label(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ConfidenceLabelError("label must be 0 or 1")
    return value


def _optional_gold_answer(value: object) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ConfidenceDatasetError("gold_answer must not be empty")
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if any(not item.strip() for item in value):
            raise ConfidenceDatasetError("gold_answer must not contain empty strings")
        return list(value)
    raise ConfidenceDatasetError("gold_answer must be a string or list of strings")


def _optional_relevance_label(value: object) -> int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float):
        return value
    raise ConfidenceDatasetError("relevance_label must be numeric or boolean")


def _metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfidenceDatasetError("metadata must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ConfidenceDatasetError("metadata keys must be strings")
    try:
        json.dumps(dict(value))
    except (TypeError, ValueError) as exc:
        raise ConfidenceDatasetError("metadata must be JSON-serializable") from exc
    return dict(value)
