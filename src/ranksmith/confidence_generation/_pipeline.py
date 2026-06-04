from __future__ import annotations

from pathlib import Path
from typing import Any

from ranksmith.confidence_generation._io import (
    load_answer_generation_samples,
    load_completed_ids,
    open_output_path,
    write_jsonl_row,
)
from ranksmith.confidence_generation._labeling import normalized_exact_match
from ranksmith.confidence_generation._parsing import parse_answer_output
from ranksmith.confidence_generation._prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
)
from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
    AnswerGenerationSample,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
    UsageCallback,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest
from ranksmith.types import RerankUsage


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
    generated_count = 0
    skipped_count = 0
    positive_count = 0
    negative_count = 0

    with open_output_path(
        output_path,
        overwrite=config.overwrite,
        resume=config.resume,
    ) as handle:
        for sample in samples:
            if sample.id in completed_ids:
                skipped_count += 1
                continue
            if config.max_items is not None and generated_count >= config.max_items:
                break

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
            write_jsonl_row(
                handle,
                _answer_canonical_row(
                    sample,
                    answer=answer,
                    label=label,
                    raw_output=raw_output,
                    config=config,
                ),
            )
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


def generate_judgment_confidence_dataset(
    config: RelevanceGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError


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
                response_format="json_object",
                temperature=0,
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
