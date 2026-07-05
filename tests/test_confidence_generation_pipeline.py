from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence_generation import (
    AnswerGenerationConfig,
    ConfidenceGenerationParseError,
    RelevanceGenerationConfig,
)
from ranksmith.confidence_generation.pipeline import _call_provider
from ranksmith.confidence_generation.prompts import build_relevance_prompt
from ranksmith.confidence_generation.types import (
    RelevanceGenerationSample,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelMessage, ModelRequest, ModelResponse, _answer_messages
from ranksmith.types import RerankUsage


class RecordingProvider:
    def __init__(self, outputs: list[ModelResponse]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outputs.pop(0)


class ProviderErrorProvider:
    def __init__(self, error: RerankProviderError) -> None:
        self.error = error

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise self.error


class BrokenProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("boom")


def test_answer_prompt_includes_configured_no_answer_value() -> None:
    _system, prompt = _answer_messages(
        "Who played Karen?",
        "Nancy Travis played Karen.",
        no_answer_value="NO_CONTEXT_ANSWER",
    )

    assert "Question:\nWho played Karen?" in prompt
    assert "Context:\nNancy Travis played Karen." in prompt
    assert '{"answer":"..."}' in prompt
    assert '{"answer":"NO_CONTEXT_ANSWER"}' in prompt


def test_relevance_prompt_includes_relevant_not_relevant_json_contract() -> None:
    prompt = build_relevance_prompt(
        RelevanceGenerationSample(
            id="j1",
            query="Who played Karen?",
            document="Nancy Travis played Karen.",
            relevance_label=1,
        )
    )

    assert "Query:\nWho played Karen?" in prompt
    assert "Document:\nNancy Travis played Karen." in prompt
    assert '{"judgment":"relevant"}' in prompt
    assert '{"judgment":"not_relevant"}' in prompt


def test_call_provider_uses_json_request_and_emits_usage() -> None:
    usage = RerankUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    provider = RecordingProvider(
        [ModelResponse(content='{"answer":"ok"}', usage=usage)]
    )
    seen: list[RerankUsage] = []

    content = _call_provider(
        provider,
        system="system",
        user="user",
        on_usage=seen.append,
    )

    assert content == '{"answer":"ok"}'
    assert provider.requests[0] == ModelRequest(
        messages=[
            ModelMessage(role="system", content="system"),
            ModelMessage(role="user", content="user"),
        ],
    )
    assert seen == [usage]


def test_call_provider_preserves_provider_errors() -> None:
    error = RerankProviderError("provider failed")

    with pytest.raises(RerankProviderError) as exc_info:
        _call_provider(
            ProviderErrorProvider(error),
            system="system",
            user="user",
            on_usage=None,
        )

    assert exc_info.value is error


def test_call_provider_wraps_unexpected_errors() -> None:
    with pytest.raises(RerankProviderError, match="boom") as exc_info:
        _call_provider(
            BrokenProvider(),
            system="system",
            user="user",
            on_usage=None,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_call_provider_rejects_empty_content() -> None:
    with pytest.raises(RerankProviderError, match="empty response"):
        _call_provider(
            RecordingProvider([ModelResponse(content="")]),
            system="system",
            user="user",
            on_usage=None,
        )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_generate_answer_confidence_dataset_writes_canonical_rows(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "a1",
                "query": "Who?",
                "context": "Nancy Travis played Karen.",
                "gold_answer": ["nancy travis"],
                "metadata": {"dataset": "unit"},
            },
            {
                "id": "a2",
                "query": "Who?",
                "context": "No answer here.",
                "gold_answer": "Nancy Travis",
                "source": "row-source",
            },
        ],
    )
    provider = RecordingProvider(
        [
            ModelResponse(content='{"answer":" Nancy   Travis "}'),
            ModelResponse(content='{"answer":"__NO_ANSWER__"}'),
        ]
    )

    result = generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            source="config-source",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 2
    assert result.generated_count == 2
    assert result.skipped_count == 0
    assert result.positive_count == 1
    assert result.negative_count == 1
    assert rows[0]["label"] == 1
    assert rows[0]["source"] == "config-source"
    assert rows[0]["metadata"]["input_metadata"] == {"dataset": "unit"}
    assert rows[0]["metadata"]["generation"]["generation_task"] == "answer_oriented"
    assert rows[0]["metadata"]["generation"]["match_policy"] == "normalized_exact"
    assert rows[0]["metadata"]["generation"]["query"] == "Who?"
    assert rows[0]["metadata"]["generation"]["raw_model_output"] == (
        '{"answer":" Nancy   Travis "}'
    )
    assert rows[1]["label"] == 0
    assert rows[1]["source"] == "row-source"
    assert provider.requests[0].messages == [
        ModelMessage(
            role="system",
            content=(
                "You answer questions using only the provided context. "
                'Return only JSON with an "answer" string.'
            ),
        ),
        ModelMessage(
            role="user",
            content=_answer_messages(
                "Who?",
                "Nancy Travis played Karen.",
                no_answer_value="__NO_ANSWER__",
            )[1],
        ),
    ]


def test_generate_answer_confidence_dataset_respects_resume_and_max_items(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a2", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a3", "query": "q", "context": "c", "gold_answer": "g"},
        ],
    )
    _write_jsonl(
        output_path,
        [{"id": "a1", "context": "c", "answer": "g", "label": 1}],
    )
    provider = RecordingProvider([ModelResponse(content='{"answer":"g"}')])

    result = generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            resume=True,
            max_items=1,
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 3
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert result.positive_count == 1
    assert result.negative_count == 0
    assert [row["id"] for row in rows] == ["a1", "a2"]


def test_generate_answer_confidence_dataset_can_omit_raw_output(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(
        input_path,
        [{"id": "a1", "query": "q", "context": "c", "gold_answer": "g"}],
    )

    generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=RecordingProvider([ModelResponse(content='{"answer":"g"}')]),
            include_raw_model_output=False,
        )
    )

    row = _read_jsonl(output_path)[0]
    assert "raw_model_output" not in row["metadata"]["generation"]


def test_generate_judgment_confidence_dataset_writes_canonical_rows(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "relevance_label": 1,
                "metadata": {"dataset": "unit"},
            },
            {
                "id": "j2",
                "query": "q",
                "document": "doc",
                "relevance_label": 0,
            },
        ],
    )
    provider = RecordingProvider(
        [
            ModelResponse(content='{"judgment":"relevant"}'),
            ModelResponse(content='{"judgment":"relevant"}'),
        ]
    )

    result = generate_judgment_confidence_dataset(
        RelevanceGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            source="config-source",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 2
    assert result.generated_count == 2
    assert result.skipped_count == 0
    assert result.positive_count == 1
    assert result.negative_count == 1
    assert rows[0]["label"] == 1
    assert rows[0]["source"] == "config-source"
    assert rows[0]["metadata"]["input_metadata"] == {"dataset": "unit"}
    assert rows[0]["metadata"]["generation"]["generation_task"] == (
        "relevance_oriented"
    )
    assert rows[0]["metadata"]["generation"]["parsed_judgment"] == "relevant"
    assert rows[0]["metadata"]["generation"]["truth_judgment"] == "relevant"
    assert rows[0]["metadata"]["generation"]["raw_model_output"] == (
        '{"judgment":"relevant"}'
    )
    assert rows[1]["label"] == 0
    assert rows[1]["metadata"]["generation"]["truth_judgment"] == "not_relevant"


def test_generate_judgment_confidence_dataset_respects_threshold_and_resume(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "j1", "query": "q", "document": "d", "relevance_label": 2},
            {"id": "j2", "query": "q", "document": "d", "relevance_label": 2},
            {"id": "j3", "query": "q", "document": "d", "relevance_label": 1},
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "d",
                "judgment": "relevant",
                "relevance_label": 2,
                "label": 1,
            }
        ],
    )
    provider = RecordingProvider([ModelResponse(content='{"judgment":"relevant"}')])

    result = generate_judgment_confidence_dataset(
        RelevanceGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            resume=True,
            max_items=1,
            truth_positive_threshold=2.0,
            truth_positive_operator="gte",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 3
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert result.positive_count == 1
    assert result.negative_count == 0
    assert [row["id"] for row in rows] == ["j1", "j2"]
    assert rows[1]["metadata"]["generation"]["truth_positive_threshold"] == 2.0
    assert rows[1]["metadata"]["generation"]["truth_positive_operator"] == "gte"


def test_generate_judgment_confidence_dataset_can_omit_raw_output(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [{"id": "j1", "query": "q", "document": "d", "relevance_label": True}],
    )

    generate_judgment_confidence_dataset(
        RelevanceGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=RecordingProvider(
                [ModelResponse(content='{"judgment":"relevant"}')]
            ),
            include_raw_model_output=False,
        )
    )

    row = _read_jsonl(output_path)[0]
    assert "raw_model_output" not in row["metadata"]["generation"]


def test_generate_judgment_confidence_dataset_keeps_only_completed_rows_on_failure(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "j1", "query": "q", "document": "d", "relevance_label": True},
            {"id": "j2", "query": "q", "document": "d", "relevance_label": True},
        ],
    )

    with pytest.raises(ConfidenceGenerationParseError):
        generate_judgment_confidence_dataset(
            RelevanceGenerationConfig(
                input_path=input_path,
                output_path=output_path,
                provider=RecordingProvider(
                    [
                        ModelResponse(content='{"judgment":"relevant"}'),
                        ModelResponse(content='{"judgment":"maybe"}'),
                    ]
                ),
            )
        )

    rows = _read_jsonl(output_path)
    assert [row["id"] for row in rows] == ["j1"]
    assert rows[0]["label"] == 1
