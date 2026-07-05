from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from ranksmith.confidence_generation.io import (
    load_answer_generation_samples,
    load_completed_ids,
    load_relevance_generation_samples,
    open_output_path,
    write_jsonl_row,
)
from ranksmith.confidence_generation.labeling import (
    JudgmentValue,
    normalized_exact_match,
    relevance_truth,
)
from ranksmith.confidence_generation.parsing import (
    parse_answer_output,
    parse_relevance_output,
)
from ranksmith.confidence_generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    build_answer_prompt,
    build_relevance_prompt,
)
from ranksmith.confidence_generation.types import (
    AnswerGenerationConfig,
    AnswerGenerationSample,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
    RelevanceGenerationSample,
    UsageCallback,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest
from ranksmith.types import RerankUsage

SampleT = TypeVar("SampleT")


def generate_answer_confidence_dataset(
    config: AnswerGenerationConfig,
) -> ConfidenceGenerationResult:
    samples = load_answer_generation_samples(
        config.input_path,
        max_context_chars=config.max_context_chars,
    )
    output_path = Path(config.output_path)
    completed_ids = (
        load_completed_ids(output_path, task_type="answer_confidence")
        if config.resume
        else set()
    )

    return _write_generation_dataset(
        samples=samples,
        output_path=output_path,
        completed_ids=completed_ids,
        overwrite=config.overwrite,
        resume=config.resume,
        max_items=config.max_items,
        get_id=lambda sample: sample.id,
        build_row=lambda sample: _build_answer_generated_row(sample, config),
    )


def generate_judgment_confidence_dataset(
    config: RelevanceGenerationConfig,
) -> ConfidenceGenerationResult:
    samples = load_relevance_generation_samples(
        config.input_path,
        max_document_chars=config.max_document_chars,
    )
    output_path = Path(config.output_path)
    completed_ids = (
        load_completed_ids(output_path, task_type="judgment_confidence")
        if config.resume
        else set()
    )

    return _write_generation_dataset(
        samples=samples,
        output_path=output_path,
        completed_ids=completed_ids,
        overwrite=config.overwrite,
        resume=config.resume,
        max_items=config.max_items,
        get_id=lambda sample: sample.id,
        build_row=lambda sample: _build_judgment_generated_row(sample, config),
    )


def _call_provider(
    provider: ModelProvider,
    *,
    system: str,
    user: str,
    on_usage: UsageCallback | None,
) -> str:
    try:
        response = provider.complete(
            ModelRequest(
                messages=[
                    ModelMessage(role="system", content=system),
                    ModelMessage(role="user", content=user),
                ],
            )
        )
    except RerankProviderError:
        raise
    except Exception as exc:
        raise RerankProviderError(str(exc)) from exc

    _emit_usage(response.usage, on_usage)
    if response.content == "":
        raise RerankProviderError("Model provider returned an empty response.")
    return response.content


def _emit_usage(usage: RerankUsage | None, callback: UsageCallback | None) -> None:
    if usage is not None and callback is not None:
        callback(usage)


def _write_generation_dataset(
    *,
    samples: Sequence[SampleT],
    output_path: Path,
    completed_ids: set[str],
    overwrite: bool,
    resume: bool,
    max_items: int | None,
    get_id: Callable[[SampleT], str],
    build_row: Callable[[SampleT], Mapping[str, Any]],
) -> ConfidenceGenerationResult:
    generated_count = 0
    skipped_count = 0
    positive_count = 0
    negative_count = 0

    with open_output_path(
        output_path,
        overwrite=overwrite,
        resume=resume,
    ) as handle:
        for sample in samples:
            if get_id(sample) in completed_ids:
                skipped_count += 1
                continue
            if max_items is not None and generated_count >= max_items:
                break

            row = build_row(sample)
            label = _generated_label(row)
            write_jsonl_row(handle, row)
            generated_count += 1
            positive_count += label
            negative_count += 1 - label

    return ConfidenceGenerationResult(
        output_path=output_path,
        input_count=len(samples),
        generated_count=generated_count,
        skipped_count=skipped_count,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def _generated_label(row: Mapping[str, Any]) -> int:
    label = row.get("label")
    if type(label) is not int or label not in {0, 1}:
        raise AssertionError("generated row label must be integer 0 or 1")
    return label


def _build_answer_generated_row(
    sample: AnswerGenerationSample,
    config: AnswerGenerationConfig,
) -> Mapping[str, Any]:
    raw_output = _call_provider(
        config.provider,
        system=ANSWER_SYSTEM_PROMPT,
        user=build_answer_prompt(
            sample,
            no_answer_value=config.no_answer_value,
        ),
        on_usage=config.on_usage,
    )
    answer = parse_answer_output(raw_output)
    label = int(
        normalized_exact_match(
            answer,
            sample.gold_answer,
            no_answer_value=config.no_answer_value,
        )
    )
    return _answer_canonical_row(
        sample,
        answer=answer,
        label=label,
        raw_output=raw_output,
        config=config,
    )


def _answer_canonical_row(
    sample: AnswerGenerationSample,
    *,
    answer: str,
    label: int,
    raw_output: str,
    config: AnswerGenerationConfig,
) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "generation_task": "answer_oriented",
        "query": sample.query,
        "match_policy": "normalized_exact",
        "no_answer_value": config.no_answer_value,
    }
    if config.include_raw_model_output:
        generation["raw_model_output"] = raw_output

    row: dict[str, Any] = {
        "id": sample.id,
        "context": sample.context,
        "answer": answer,
        "gold_answer": sample.gold_answer,
        "label": label,
        "metadata": {
            "input_metadata": dict(sample.metadata),
            "generation": generation,
        },
    }
    source = sample.source if sample.source is not None else config.source
    if source is not None:
        row["source"] = source
    if sample.group_id is not None:
        row["group_id"] = sample.group_id
    return row


def _build_judgment_generated_row(
    sample: RelevanceGenerationSample,
    config: RelevanceGenerationConfig,
) -> Mapping[str, Any]:
    raw_output = _call_provider(
        config.provider,
        system=RELEVANCE_SYSTEM_PROMPT,
        user=build_relevance_prompt(sample),
        on_usage=config.on_usage,
    )
    judgment = parse_relevance_output(raw_output)
    truth = relevance_truth(
        sample.relevance_label,
        threshold=float(config.truth_positive_threshold),
        operator=config.truth_positive_operator,
    )
    label = int(judgment == truth)
    return _judgment_canonical_row(
        sample,
        judgment=judgment,
        truth=truth,
        label=label,
        raw_output=raw_output,
        config=config,
    )


def _judgment_canonical_row(
    sample: RelevanceGenerationSample,
    *,
    judgment: JudgmentValue,
    truth: JudgmentValue,
    label: int,
    raw_output: str,
    config: RelevanceGenerationConfig,
) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "generation_task": "relevance_oriented",
        "parsed_judgment": judgment,
        "truth_judgment": truth,
        "truth_positive_threshold": float(config.truth_positive_threshold),
        "truth_positive_operator": config.truth_positive_operator,
    }
    if config.include_raw_model_output:
        generation["raw_model_output"] = raw_output

    row: dict[str, Any] = {
        "id": sample.id,
        "query": sample.query,
        "document": sample.document,
        "judgment": judgment,
        "relevance_label": sample.relevance_label,
        "label": label,
        "metadata": {
            "input_metadata": dict(sample.metadata),
            "generation": generation,
        },
    }
    source = sample.source if sample.source is not None else config.source
    if source is not None:
        row["source"] = source
    if sample.group_id is not None:
        row["group_id"] = sample.group_id
    return row
