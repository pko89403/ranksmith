from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence_generation._errors import (
    ConfidenceGenerationInputError,
    ConfidenceGenerationParseError,
)
from ranksmith.confidence_generation._io import (
    load_completed_ids,
    load_query_answerability_generation_samples,
    load_query_context_answerability_generation_samples,
)
from ranksmith.confidence_generation._prompts import (
    QUERY_ANSWERABILITY_SYSTEM_PROMPT,
    QUERY_CONTEXT_ANSWERABILITY_SYSTEM_PROMPT,
    build_query_answerability_prompt,
    build_query_context_answerability_prompt,
)
from ranksmith.confidence_generation._types import (
    QueryAnswerabilityGenerationSample,
    QueryContextAnswerabilityGenerationSample,
)
from ranksmith.model import ModelMessage, ModelRequest, ModelResponse
from ranksmith.types import RerankUsage

ANSWERABILITY_PUBLIC_NAMES = (
    "QueryAnswerabilityGenerationConfig",
    "QueryContextAnswerabilityGenerationConfig",
    "generate_query_answerability_confidence_dataset",
    "generate_query_context_answerability_confidence_dataset",
)


class RecordingProvider:
    def __init__(self, outputs: list[ModelResponse]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outputs.pop(0)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_query_answerability_prompt_uses_parametric_knowledge_contract() -> None:
    prompt = build_query_answerability_prompt(
        QueryAnswerabilityGenerationSample(
            id="q1",
            query="Who played Karen?",
            gold_answer="Nancy Travis",
        ),
        no_answer_value="UNKNOWN",
    )

    assert "Question:\nWho played Karen?" in prompt
    assert "Context:" not in prompt
    assert '{"answer":"short answer"}' in prompt
    assert '{"answer":"UNKNOWN"}' in prompt
    assert "parametric knowledge" in prompt


def test_query_context_answerability_prompt_uses_context_contract() -> None:
    prompt = build_query_context_answerability_prompt(
        QueryContextAnswerabilityGenerationSample(
            id="c1",
            query="Who played Karen?",
            context="Nancy Travis played Karen.",
            gold_answer="Nancy Travis",
        ),
        no_answer_value="NO_CONTEXT_ANSWER",
    )

    assert "Question:\nWho played Karen?" in prompt
    assert "Context:\nNancy Travis played Karen." in prompt
    assert '{"answer":"short answer"}' in prompt
    assert '{"answer":"NO_CONTEXT_ANSWER"}' in prompt
    assert "Use only the context" in prompt


def test_answerability_public_exports_are_submodule_only() -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    ranksmith = importlib.import_module("ranksmith")

    for name in ANSWERABILITY_PUBLIC_NAMES:
        assert hasattr(generation, name)
        assert not hasattr(ranksmith, name)


def test_query_answerability_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    provider = RecordingProvider([])

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.QueryAnswerabilityGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=object(),
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.QueryAnswerabilityGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=provider,
            overwrite=True,
            resume=True,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.QueryAnswerabilityGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=provider,
            max_items=0,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.QueryAnswerabilityGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=provider,
            no_answer_value=" ",
        )


def test_query_context_answerability_config_rejects_invalid_options(
    tmp_path: Path,
) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.QueryContextAnswerabilityGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=RecordingProvider([]),
            max_context_chars=0,
        )


def test_answerability_raw_loaders_validate_and_preserve_text(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "query.jsonl"
    context_path = tmp_path / "context.jsonl"
    _write_jsonl(
        query_path,
        [
            {
                "id": " q1 ",
                "query": " Who? ",
                "gold_answer": [" Gold "],
                "source": " source ",
                "group_id": " group ",
                "metadata": {"dataset": "unit"},
            }
        ],
    )
    _write_jsonl(
        context_path,
        [
            {
                "id": "c1",
                "query": "q",
                "context": " 12345 ",
                "gold_answer": "g",
            }
        ],
    )

    query_samples = load_query_answerability_generation_samples(query_path)
    assert query_samples[0].id == " q1 "
    assert query_samples[0].query == " Who? "
    assert query_samples[0].gold_answer == [" Gold "]
    assert query_samples[0].source == " source "
    assert query_samples[0].group_id == " group "
    assert query_samples[0].metadata["dataset"] == "unit"

    with pytest.raises(ConfidenceGenerationInputError, match="context"):
        load_query_context_answerability_generation_samples(
            context_path,
            max_context_chars=5,
        )


@pytest.mark.parametrize(
    "row",
    [
        {"id": "q1", "query": "q"},
        {"id": "q1", "query": " ", "gold_answer": "g"},
        {"id": "q1", "query": "q", "gold_answer": "", "extra": "nope"},
        {"id": "q1", "query": "q", "gold_answer": "g", "metadata": []},
    ],
)
def test_query_answerability_loader_rejects_invalid_rows(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    path = tmp_path / "query.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ConfidenceGenerationInputError):
        load_query_answerability_generation_samples(path)


def test_generate_answerability_datasets_write_canonical_rows_and_usage(
    tmp_path: Path,
) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    query_input = tmp_path / "query_in.jsonl"
    query_output = tmp_path / "query_out.jsonl"
    context_input = tmp_path / "context_in.jsonl"
    context_output = tmp_path / "context_out.jsonl"
    _write_jsonl(
        query_input,
        [
            {
                "id": "q1",
                "query": "Who played Karen?",
                "gold_answer": ["nancy travis"],
                "metadata": {"dataset": "unit"},
            }
        ],
    )
    _write_jsonl(
        context_input,
        [
            {
                "id": "c1",
                "query": "Who played Karen?",
                "context": "Nancy Travis played Karen.",
                "gold_answer": "Nancy Travis",
                "source": "row-source",
            }
        ],
    )
    usage = RerankUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    query_provider = RecordingProvider(
        [ModelResponse(content='{"answer":" Nancy   Travis "}', usage=usage)]
    )
    context_provider = RecordingProvider(
        [ModelResponse(content='{"answer":"__NO_ANSWER__"}')]
    )
    seen_usage: list[RerankUsage] = []

    query_result = generation.generate_query_answerability_confidence_dataset(
        generation.QueryAnswerabilityGenerationConfig(
            input_path=query_input,
            output_path=query_output,
            provider=query_provider,
            source="config-source",
            on_usage=seen_usage.append,
        )
    )
    context_result = generation.generate_query_context_answerability_confidence_dataset(
        generation.QueryContextAnswerabilityGenerationConfig(
            input_path=context_input,
            output_path=context_output,
            provider=context_provider,
            include_raw_model_output=False,
        )
    )

    query_rows = _read_jsonl(query_output)
    context_rows = _read_jsonl(context_output)
    assert query_result.generated_count == 1
    assert query_result.positive_count == 1
    assert query_rows[0]["task_type"] == "query_answerability_confidence"
    assert query_rows[0]["id"] == "q1"
    assert query_rows[0]["query"] == "Who played Karen?"
    assert query_rows[0]["answer"] == " Nancy   Travis "
    assert query_rows[0]["gold_answer"] == ["nancy travis"]
    assert query_rows[0]["label"] == 1
    assert query_rows[0]["source"] == "config-source"
    assert query_rows[0]["metadata"]["input_metadata"] == {"dataset": "unit"}
    assert query_rows[0]["metadata"]["generation"]["generation_task"] == (
        "query_answerability"
    )
    assert query_rows[0]["metadata"]["generation"]["match_policy"] == (
        "normalized_exact"
    )
    assert query_rows[0]["metadata"]["generation"]["raw_model_output"] == (
        '{"answer":" Nancy   Travis "}'
    )
    assert context_result.negative_count == 1
    assert context_rows[0]["task_type"] == "query_context_answerability_confidence"
    assert context_rows[0]["context"] == "Nancy Travis played Karen."
    assert context_rows[0]["source"] == "row-source"
    assert "raw_model_output" not in context_rows[0]["metadata"]["generation"]
    assert seen_usage == [usage]


def test_answerability_pipelines_use_expected_model_request_messages(
    tmp_path: Path,
) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    query_input = tmp_path / "query_in.jsonl"
    query_output = tmp_path / "query_out.jsonl"
    context_input = tmp_path / "context_in.jsonl"
    context_output = tmp_path / "context_out.jsonl"
    _write_jsonl(
        query_input,
        [{"id": "q1", "query": "Who played Karen?", "gold_answer": "Nancy Travis"}],
    )
    _write_jsonl(
        context_input,
        [
            {
                "id": "c1",
                "query": "Who played Karen?",
                "context": "Nancy Travis played Karen.",
                "gold_answer": "Nancy Travis",
            }
        ],
    )
    query_provider = RecordingProvider([ModelResponse(content='{"answer":"Nancy"}')])
    context_provider = RecordingProvider([ModelResponse(content='{"answer":"Nancy"}')])

    generation.generate_query_answerability_confidence_dataset(
        generation.QueryAnswerabilityGenerationConfig(
            input_path=query_input,
            output_path=query_output,
            provider=query_provider,
            no_answer_value="UNKNOWN",
        )
    )
    generation.generate_query_context_answerability_confidence_dataset(
        generation.QueryContextAnswerabilityGenerationConfig(
            input_path=context_input,
            output_path=context_output,
            provider=context_provider,
            no_answer_value="NO_CONTEXT_ANSWER",
        )
    )

    assert query_provider.requests[0].messages == [
        ModelMessage(role="system", content=QUERY_ANSWERABILITY_SYSTEM_PROMPT),
        ModelMessage(
            role="user",
            content=build_query_answerability_prompt(
                QueryAnswerabilityGenerationSample(
                    id="q1",
                    query="Who played Karen?",
                    gold_answer="Nancy Travis",
                ),
                no_answer_value="UNKNOWN",
            ),
        ),
    ]
    assert context_provider.requests[0].messages == [
        ModelMessage(
            role="system",
            content=QUERY_CONTEXT_ANSWERABILITY_SYSTEM_PROMPT,
        ),
        ModelMessage(
            role="user",
            content=build_query_context_answerability_prompt(
                QueryContextAnswerabilityGenerationSample(
                    id="c1",
                    query="Who played Karen?",
                    context="Nancy Travis played Karen.",
                    gold_answer="Nancy Travis",
                ),
                no_answer_value="NO_CONTEXT_ANSWER",
            ),
        ),
    ]


def test_answerability_resume_accepts_completed_rows_and_skips(
    tmp_path: Path,
) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    input_path = tmp_path / "query_in.jsonl"
    output_path = tmp_path / "query_out.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "q1", "query": "q", "gold_answer": "g"},
            {"id": "q2", "query": "q", "gold_answer": "g"},
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "id": "q1",
                "task_type": "query_answerability_confidence",
                "query": "q",
                "answer": "g",
                "label": 1,
            }
        ],
    )
    provider = RecordingProvider([ModelResponse(content='{"answer":"g"}')])

    result = generation.generate_query_answerability_confidence_dataset(
        generation.QueryAnswerabilityGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            resume=True,
        )
    )

    rows = _read_jsonl(output_path)
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert [row["id"] for row in rows] == ["q1", "q2"]


def test_query_context_answerability_resume_after_partial_parse_failure(
    tmp_path: Path,
) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")
    input_path = tmp_path / "context_in.jsonl"
    output_path = tmp_path / "context_out.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "c1",
                "query": "q",
                "context": "gold answer appears here",
                "gold_answer": "gold",
            },
            {
                "id": "c2",
                "query": "q",
                "context": "next answer appears here",
                "gold_answer": "next",
            },
        ],
    )
    first_provider = RecordingProvider(
        [
            ModelResponse(content='{"answer":"gold"}'),
            ModelResponse(content='{"answer":}'),
        ]
    )

    with pytest.raises(ConfidenceGenerationParseError, match="valid JSON"):
        generation.generate_query_context_answerability_confidence_dataset(
            generation.QueryContextAnswerabilityGenerationConfig(
                input_path=input_path,
                output_path=output_path,
                provider=first_provider,
            )
        )

    assert len(first_provider.requests) == 2
    assert [row["id"] for row in _read_jsonl(output_path)] == ["c1"]

    second_provider = RecordingProvider([ModelResponse(content='{"answer":"next"}')])
    result = generation.generate_query_context_answerability_confidence_dataset(
        generation.QueryContextAnswerabilityGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=second_provider,
            resume=True,
        )
    )

    rows = _read_jsonl(output_path)
    assert len(second_provider.requests) == 1
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert [row["id"] for row in rows] == ["c1", "c2"]


def test_answerability_resume_rejects_mismatched_task_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "q1",
                "task_type": "query_context_answerability_confidence",
                "query": "q",
                "answer": "a",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="task_type"):
        load_completed_ids(path, task_type="query_answerability_confidence")
