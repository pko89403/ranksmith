from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Literal

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation._types import (
    AnswerGenerationSample,
    RelevanceGenerationSample,
)

TaskType = Literal["answer_confidence", "judgment_confidence"]

_ANSWER_REQUIRED = ("id", "query", "context", "gold_answer")
_ANSWER_ALLOWED = {*_ANSWER_REQUIRED, "source", "group_id", "metadata"}
_RELEVANCE_REQUIRED = ("id", "query", "document", "relevance_label")
_RELEVANCE_ALLOWED = {*_RELEVANCE_REQUIRED, "source", "group_id", "metadata"}
_ANSWER_OUTPUT_REQUIRED = ("id", "context", "answer", "label")
_ANSWER_OUTPUT_ALLOWED = {
    *_ANSWER_OUTPUT_REQUIRED,
    "gold_answer",
    "source",
    "group_id",
    "metadata",
}
_JUDGMENT_OUTPUT_REQUIRED = (
    "id",
    "query",
    "document",
    "judgment",
    "relevance_label",
    "label",
)
_JUDGMENT_OUTPUT_ALLOWED = {
    *_JUDGMENT_OUTPUT_REQUIRED,
    "relevance_label",
    "source",
    "group_id",
    "metadata",
}


def load_answer_generation_samples(
    path: str | Path,
    *,
    max_context_chars: int,
) -> list[AnswerGenerationSample]:
    samples: list[AnswerGenerationSample] = []
    seen: set[str] = set()
    for line_number, row in _read_jsonl_objects(Path(path)):
        try:
            _validate_keys(row, required=_ANSWER_REQUIRED, allowed=_ANSWER_ALLOWED)
            sample = AnswerGenerationSample(
                id=_required_text(row, "id"),
                query=_required_text(row, "query"),
                context=_bounded_text(row, "context", max_context_chars),
                gold_answer=_gold_answer(row["gold_answer"]),
                source=_optional_text(row.get("source"), "source"),
                group_id=_optional_text(row.get("group_id"), "group_id"),
                metadata=_metadata(row.get("metadata")),
            )
        except ConfidenceGenerationInputError as exc:
            raise ConfidenceGenerationInputError(f"line {line_number}: {exc}") from exc
        _check_duplicate(sample.id, seen)
        samples.append(sample)
    return samples


def load_relevance_generation_samples(
    path: str | Path,
    *,
    max_document_chars: int,
) -> list[RelevanceGenerationSample]:
    samples: list[RelevanceGenerationSample] = []
    seen: set[str] = set()
    for line_number, row in _read_jsonl_objects(Path(path)):
        try:
            _validate_keys(
                row,
                required=_RELEVANCE_REQUIRED,
                allowed=_RELEVANCE_ALLOWED,
            )
            sample = RelevanceGenerationSample(
                id=_required_text(row, "id"),
                query=_required_text(row, "query"),
                document=_bounded_text(row, "document", max_document_chars),
                relevance_label=_relevance_label(row["relevance_label"]),
                source=_optional_text(row.get("source"), "source"),
                group_id=_optional_text(row.get("group_id"), "group_id"),
                metadata=_metadata(row.get("metadata")),
            )
        except ConfidenceGenerationInputError as exc:
            raise ConfidenceGenerationInputError(f"line {line_number}: {exc}") from exc
        _check_duplicate(sample.id, seen)
        samples.append(sample)
    return samples


def open_output_path(
    path: str | Path,
    *,
    overwrite: bool,
    resume: bool,
) -> IO[str]:
    if overwrite and resume:
        raise ConfidenceGenerationInputError("overwrite and resume cannot both be true")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and not resume:
        raise ConfidenceGenerationInputError(
            "output_path already exists; use overwrite or resume"
        )

    if resume:
        _ensure_trailing_newline_for_append(output_path)
    mode = "a" if resume else "w"
    return output_path.open(mode, encoding="utf-8")


def write_jsonl_row(handle: IO[str], row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def load_completed_ids(path: str | Path, *, task_type: TaskType) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()

    ids: set[str] = set()
    for line_number, row in _read_jsonl_objects(output_path):
        try:
            row_id = _validate_output_task(row, task_type)
        except ConfidenceGenerationInputError as exc:
            raise ConfidenceGenerationInputError(f"line {line_number}: {exc}") from exc
        _check_duplicate(row_id, ids, location="output")
    return ids


def _ensure_trailing_newline_for_append(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return
        with path.open("ab") as handle:
            handle.write(b"\n")
    except OSError as exc:
        raise ConfidenceGenerationInputError(
            f"output file could not be prepared for resume append: {path}"
        ) from exc


def _read_jsonl_objects(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfidenceGenerationInputError(
            f"jsonl file could not be read: {path}"
        ) from exc

    rows: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise ConfidenceGenerationInputError(
                f"line {line_number}: must be valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ConfidenceGenerationInputError(
                f"line {line_number}: must be a JSON object"
            )
        rows.append((line_number, value))
    return rows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _validate_keys(
    row: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    allowed: set[str],
) -> None:
    for field in required:
        if field not in row:
            raise ConfidenceGenerationInputError(f"missing required field: {field}")

    unexpected = sorted(key for key in row if key not in allowed)
    if unexpected:
        raise ConfidenceGenerationInputError(f"unexpected field: {unexpected[0]}")


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ConfidenceGenerationInputError(f"{name} must be a string")
    if not value.strip():
        raise ConfidenceGenerationInputError(f"{name} must not be empty")
    return value


def _bounded_text(row: Mapping[str, Any], name: str, max_chars: int) -> str:
    value = _required_text(row, name)
    if len(value) > max_chars:
        raise ConfidenceGenerationInputError(f"{name} exceeds max chars")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfidenceGenerationInputError(f"{name} must be a string")
    if not value.strip():
        raise ConfidenceGenerationInputError(f"{name} must not be empty")
    return value


def _gold_answer(value: object) -> str | list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ConfidenceGenerationInputError("gold_answer must not be empty")
        return value

    if isinstance(value, list):
        if not value:
            raise ConfidenceGenerationInputError("gold_answer list must not be empty")
        if not all(isinstance(item, str) for item in value):
            raise ConfidenceGenerationInputError(
                "gold_answer list must contain only strings"
            )
        if any(not item.strip() for item in value):
            raise ConfidenceGenerationInputError(
                "gold_answer list must not contain empty strings"
            )
        return list(value)

    raise ConfidenceGenerationInputError(
        "gold_answer must be a string or non-empty list of strings"
    )


def _relevance_label(value: object) -> int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if not math.isfinite(value):
            raise ConfidenceGenerationInputError("relevance_label must be finite")
        return value
    raise ConfidenceGenerationInputError("relevance_label must be numeric or bool")


def _metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfidenceGenerationInputError("metadata must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ConfidenceGenerationInputError("metadata keys must be strings")

    metadata = dict(value)
    try:
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfidenceGenerationInputError(
            "metadata values must be JSON-serializable"
        ) from exc
    return metadata


def _validate_output_task(row: Mapping[str, Any], task_type: TaskType) -> str:
    if task_type == "answer_confidence":
        _validate_keys(
            row,
            required=_ANSWER_OUTPUT_REQUIRED,
            allowed=_ANSWER_OUTPUT_ALLOWED,
        )
        row_id = _required_text(row, "id")
        _required_text(row, "context")
        _required_text(row, "answer")
        _label(row["label"])
        if "gold_answer" in row:
            _gold_answer(row["gold_answer"])
    else:
        _validate_keys(
            row,
            required=_JUDGMENT_OUTPUT_REQUIRED,
            allowed=_JUDGMENT_OUTPUT_ALLOWED,
        )
        row_id = _required_text(row, "id")
        _required_text(row, "query")
        _required_text(row, "document")
        _judgment(row["judgment"])
        _label(row["label"])
        _relevance_label(row["relevance_label"])

    if "source" in row:
        _required_text(row, "source")
    if "group_id" in row:
        _required_text(row, "group_id")
    if "metadata" in row:
        _metadata_present(row["metadata"])
    return row_id


def _label(value: object) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ConfidenceGenerationInputError("label must be integer 0 or 1")
    return value


def _judgment(value: object) -> str:
    if value not in {"relevant", "not_relevant"}:
        raise ConfidenceGenerationInputError(
            'judgment must be "relevant" or "not_relevant"'
        )
    return str(value)


def _metadata_present(value: object) -> dict[str, Any]:
    if value is None:
        raise ConfidenceGenerationInputError("metadata must be a JSON object")
    return _metadata(value)


def _check_duplicate(
    row_id: str,
    seen: set[str],
    *,
    location: str = "input",
) -> None:
    if row_id in seen:
        raise ConfidenceGenerationInputError(f"duplicate id in {location}: {row_id}")
    seen.add(row_id)
