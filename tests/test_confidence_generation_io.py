from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence_generation.errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation.io import (
    _metadata,
    load_answer_generation_samples,
    load_completed_ids,
    load_relevance_generation_samples,
    open_output_path,
    write_jsonl_row,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_answer_generation_samples_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a1",
                "query": "q",
                "context": "ctx",
                "gold_answer": ["gold"],
                "source": "dataset",
                "group_id": "g1",
                "metadata": {"dataset": "d", "nested": {"k": [1, True, None]}},
            }
        ],
    )

    samples = load_answer_generation_samples(path, max_context_chars=10)

    assert len(samples) == 1
    assert samples[0].id == "a1"
    assert samples[0].context == "ctx"
    assert samples[0].gold_answer == ["gold"]
    assert samples[0].source == "dataset"
    assert samples[0].group_id == "g1"
    assert samples[0].metadata["dataset"] == "d"


def test_load_relevance_generation_samples_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "relevance.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "relevance_label": 1,
                "metadata": {"dataset": "d"},
            }
        ],
    )

    samples = load_relevance_generation_samples(path, max_document_chars=10)

    assert len(samples) == 1
    assert samples[0].id == "j1"
    assert samples[0].document == "doc"
    assert samples[0].relevance_label == 1
    assert samples[0].metadata["dataset"] == "d"


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a1", "query": "q", "context": "ctx"},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": "g", "x": 1},
        {"id": "a1", "query": " ", "context": "ctx", "gold_answer": "g"},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": []},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": [""]},
        {
            "id": "a1",
            "query": "q",
            "context": "ctx",
            "gold_answer": "g",
            "metadata": [],
        },
    ],
)
def test_load_answer_generation_samples_rejects_invalid_rows(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ConfidenceGenerationInputError):
        load_answer_generation_samples(path, max_context_chars=4000)


@pytest.mark.parametrize(
    "row",
    [
        {"id": "j1", "query": "q", "document": "doc"},
        {
            "id": "j1",
            "query": "q",
            "document": "doc",
            "relevance_label": 1,
            "extra": "nope",
        },
        {"id": "j1", "query": "q", "document": "doc", "relevance_label": "1"},
        {"id": "j1", "query": "q", "document": "doc", "relevance_label": None},
        {"id": "j1", "query": "q", "document": " ", "relevance_label": 1},
    ],
)
def test_load_relevance_generation_samples_rejects_invalid_rows(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    path = tmp_path / "relevance.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ConfidenceGenerationInputError):
        load_relevance_generation_samples(path, max_document_chars=4000)


def test_load_answer_generation_samples_rejects_too_long_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [{"id": "a1", "query": "q", "context": "12345", "gold_answer": "g"}],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="context"):
        load_answer_generation_samples(path, max_context_chars=4)


def test_load_answer_generation_samples_uses_raw_context_length_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [{"id": "a1", "query": "q", "context": " 12345 ", "gold_answer": "g"}],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="context"):
        load_answer_generation_samples(path, max_context_chars=5)


def test_load_relevance_generation_samples_rejects_too_long_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rel.jsonl"
    _write_jsonl(
        path,
        [{"id": "j1", "query": "q", "document": "12345", "relevance_label": 1}],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="document"):
        load_relevance_generation_samples(path, max_document_chars=4)


def test_duplicate_input_ids_fail(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
        ],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="duplicate id"):
        load_answer_generation_samples(path, max_context_chars=4000)


def test_duplicate_input_ids_use_raw_id_and_preserve_id(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {"id": " a1 ", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
        ],
    )

    samples = load_answer_generation_samples(path, max_context_chars=4000)

    assert [sample.id for sample in samples] == [" a1 ", "a1"]


def test_source_and_group_id_preserve_surrounding_spaces(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a1",
                "query": "q",
                "context": "c",
                "gold_answer": "g",
                "source": " source ",
                "group_id": " group ",
            }
        ],
    )

    samples = load_answer_generation_samples(path, max_context_chars=4000)

    assert samples[0].source == " source "
    assert samples[0].group_id == " group "


def test_metadata_must_have_string_keys_and_json_serializable_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answer.jsonl"
    path.write_text(
        '{"id":"a1","query":"q","context":"c","gold_answer":"g",'
        '"metadata":{"ok":1,"bad":NaN}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfidenceGenerationInputError, match="valid JSON"):
        load_answer_generation_samples(path, max_context_chars=4000)

    metadata_with_non_string_key: dict[Any, Any] = {1: "bad"}
    with pytest.raises(ConfidenceGenerationInputError, match="metadata keys"):
        _metadata(metadata_with_non_string_key)


def test_output_policy_rejects_existing_file_without_overwrite_or_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfidenceGenerationInputError):
        open_output_path(path, overwrite=False, resume=False)


def test_output_policy_supports_parent_creation_overwrite_and_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "out.jsonl"

    with open_output_path(path, overwrite=False, resume=True) as handle:
        write_jsonl_row(handle, {"id": "first"})
    assert path.read_text(encoding="utf-8") == '{"id": "first"}\n'

    with open_output_path(path, overwrite=True, resume=False) as handle:
        write_jsonl_row(handle, {"id": "new"})
    assert path.read_text(encoding="utf-8") == '{"id": "new"}\n'

    with open_output_path(path, overwrite=False, resume=True) as handle:
        write_jsonl_row(handle, {"id": "next"})
    assert path.read_text(encoding="utf-8").endswith('{"id": "next"}\n')


def test_resume_append_preserves_jsonl_when_existing_file_has_no_trailing_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "first"}', encoding="utf-8")

    with open_output_path(path, overwrite=False, resume=True) as handle:
        write_jsonl_row(handle, {"id": "next"})

    assert path.read_text(encoding="utf-8") == ('{"id": "first"}\n{"id": "next"}\n')


def test_output_policy_rejects_overwrite_and_resume_together(tmp_path: Path) -> None:
    with pytest.raises(ConfidenceGenerationInputError):
        open_output_path(tmp_path / "out.jsonl", overwrite=True, resume=True)


def test_load_completed_ids_rejects_duplicates_and_task_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 1}])

    assert load_completed_ids(path, task_type="answer_confidence") == {"a1"}

    with pytest.raises(ConfidenceGenerationInputError):
        load_completed_ids(path, task_type="judgment_confidence")

    _write_jsonl(
        path,
        [
            {"id": "a1", "context": "c", "answer": "a", "label": 1},
            {"id": "a1", "context": "c", "answer": "a", "label": 0},
        ],
    )
    with pytest.raises(ConfidenceGenerationInputError, match="duplicate id"):
        load_completed_ids(path, task_type="answer_confidence")


def test_load_completed_ids_accepts_missing_output_path(tmp_path: Path) -> None:
    assert (
        load_completed_ids(tmp_path / "missing.jsonl", task_type="answer_confidence")
        == set()
    )


@pytest.mark.parametrize("label", ["oops", True])
def test_load_completed_ids_rejects_invalid_label(
    tmp_path: Path,
    label: object,
) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": label}])

    with pytest.raises(ConfidenceGenerationInputError, match="label"):
        load_completed_ids(path, task_type="answer_confidence")


def test_load_completed_ids_rejects_unexpected_field(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(
        path,
        [{"id": "a1", "context": "c", "answer": "a", "label": 1, "extra": "nope"}],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="unexpected field"):
        load_completed_ids(path, task_type="answer_confidence")


def test_load_completed_ids_accepts_valid_judgment_canonical_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "judgment": "relevant",
                "label": 1,
                "relevance_label": 0.5,
            }
        ],
    )

    assert load_completed_ids(path, task_type="judgment_confidence") == {"j1"}


@pytest.mark.parametrize(
    "row, match",
    [
        (
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "judgment": "direct evidence",
                "label": 1,
                "relevance_label": 0.5,
            },
            "judgment",
        ),
        (
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "judgment": "relevant",
                "label": 1,
            },
            "missing required field: relevance_label",
        ),
    ],
)
def test_load_completed_ids_rejects_invalid_judgment_canonical_row(
    tmp_path: Path,
    row: dict[str, object],
    match: str,
) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ConfidenceGenerationInputError, match=match):
        load_completed_ids(path, task_type="judgment_confidence")
