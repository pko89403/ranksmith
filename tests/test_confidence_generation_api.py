from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ranksmith.model import ModelResponse


class FakeProvider:
    def complete(self, request: object) -> ModelResponse:
        return ModelResponse(content='{"answer":"ok"}')


def test_confidence_generation_public_submodule_exports_are_available() -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    assert hasattr(generation, "AnswerGenerationConfig")
    assert hasattr(generation, "RelevanceGenerationConfig")
    assert hasattr(generation, "ConfidenceGenerationResult")
    assert hasattr(generation, "ConfidenceGenerationError")
    assert hasattr(generation, "ConfidenceGenerationInputError")
    assert hasattr(generation, "ConfidenceGenerationParseError")
    assert hasattr(generation, "generate_answer_confidence_dataset")
    assert hasattr(generation, "generate_judgment_confidence_dataset")


def test_confidence_generation_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    assert not hasattr(ranksmith, "AnswerGenerationConfig")
    assert not hasattr(ranksmith, "RelevanceGenerationConfig")
    assert not hasattr(ranksmith, "ConfidenceGenerationError")


def test_answer_generation_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

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
