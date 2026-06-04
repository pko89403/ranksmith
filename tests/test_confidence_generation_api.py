from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import pytest

from ranksmith.model import ModelResponse

CONFIDENCE_GENERATION_PUBLIC_NAMES = (
    "AnswerGenerationConfig",
    "RelevanceGenerationConfig",
    "ConfidenceGenerationResult",
    "ConfidenceGenerationError",
    "ConfidenceGenerationInputError",
    "ConfidenceGenerationParseError",
    "generate_answer_confidence_dataset",
    "generate_judgment_confidence_dataset",
)


class FakeProvider:
    def complete(self, request: object) -> ModelResponse:
        return ModelResponse(content='{"answer":"ok"}')


def test_confidence_generation_public_submodule_exports_are_available() -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    for name in CONFIDENCE_GENERATION_PUBLIC_NAMES:
        assert hasattr(generation, name)


def test_confidence_generation_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    for name in CONFIDENCE_GENERATION_PUBLIC_NAMES:
        assert not hasattr(ranksmith, name)


def test_answer_generation_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=cast(Any, object()),
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            overwrite=True,
            resume=True,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_items=0,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_context_chars=0,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            no_answer_value=" ",
        )


def test_relevance_generation_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.RelevanceGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=cast(Any, object()),
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.RelevanceGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            truth_positive_operator="eq",
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.RelevanceGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_document_chars=0,
        )

    for invalid_threshold in (True, "0"):
        with pytest.raises(generation.ConfidenceGenerationInputError):
            generation.RelevanceGenerationConfig(
                input_path=tmp_path / "in.jsonl",
                output_path=tmp_path / "out.jsonl",
                provider=FakeProvider(),
                truth_positive_threshold=cast(Any, invalid_threshold),
            )

    for non_finite_threshold in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(generation.ConfidenceGenerationInputError):
            generation.RelevanceGenerationConfig(
                input_path=tmp_path / "in.jsonl",
                output_path=tmp_path / "out.jsonl",
                provider=FakeProvider(),
                truth_positive_threshold=non_finite_threshold,
            )
