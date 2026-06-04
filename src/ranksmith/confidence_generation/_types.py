from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError
from ranksmith.model import ModelProvider
from ranksmith.types import RerankUsage

TruthPositiveOperator = Literal["gt", "gte"]
UsageCallback = Callable[[RerankUsage], None]


@dataclass(frozen=True)
class AnswerGenerationConfig:
    input_path: str | Path
    output_path: str | Path
    provider: ModelProvider
    overwrite: bool = False
    resume: bool = False
    max_items: int | None = None
    max_context_chars: int = 4000
    include_raw_model_output: bool = True
    on_usage: UsageCallback | None = None
    source: str | None = None
    no_answer_value: str = "__NO_ANSWER__"

    def __post_init__(self) -> None:
        _validate_common_config(
            provider=self.provider,
            overwrite=self.overwrite,
            resume=self.resume,
            max_items=self.max_items,
            source=self.source,
        )
        _validate_positive_int("max_context_chars", self.max_context_chars)
        if (
            not isinstance(self.no_answer_value, str)
            or not self.no_answer_value.strip()
        ):
            raise ConfidenceGenerationInputError(
                "no_answer_value must be a non-empty string"
            )


@dataclass(frozen=True)
class QueryAnswerabilityGenerationConfig:
    input_path: str | Path
    output_path: str | Path
    provider: ModelProvider
    overwrite: bool = False
    resume: bool = False
    max_items: int | None = None
    include_raw_model_output: bool = True
    on_usage: UsageCallback | None = None
    source: str | None = None
    no_answer_value: str = "__NO_ANSWER__"

    def __post_init__(self) -> None:
        _validate_common_config(
            provider=self.provider,
            overwrite=self.overwrite,
            resume=self.resume,
            max_items=self.max_items,
            source=self.source,
        )
        _validate_no_answer_value(self.no_answer_value)


@dataclass(frozen=True)
class QueryContextAnswerabilityGenerationConfig:
    input_path: str | Path
    output_path: str | Path
    provider: ModelProvider
    overwrite: bool = False
    resume: bool = False
    max_items: int | None = None
    max_context_chars: int = 4000
    include_raw_model_output: bool = True
    on_usage: UsageCallback | None = None
    source: str | None = None
    no_answer_value: str = "__NO_ANSWER__"

    def __post_init__(self) -> None:
        _validate_common_config(
            provider=self.provider,
            overwrite=self.overwrite,
            resume=self.resume,
            max_items=self.max_items,
            source=self.source,
        )
        _validate_positive_int("max_context_chars", self.max_context_chars)
        _validate_no_answer_value(self.no_answer_value)


@dataclass(frozen=True)
class RelevanceGenerationConfig:
    input_path: str | Path
    output_path: str | Path
    provider: ModelProvider
    truth_positive_threshold: float = 0.0
    truth_positive_operator: TruthPositiveOperator = "gt"
    overwrite: bool = False
    resume: bool = False
    max_items: int | None = None
    max_document_chars: int = 4000
    include_raw_model_output: bool = True
    on_usage: UsageCallback | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        _validate_common_config(
            provider=self.provider,
            overwrite=self.overwrite,
            resume=self.resume,
            max_items=self.max_items,
            source=self.source,
        )
        if self.truth_positive_operator not in {"gt", "gte"}:
            raise ConfidenceGenerationInputError(
                'truth_positive_operator must be "gt" or "gte"'
            )
        if isinstance(self.truth_positive_threshold, bool) or not isinstance(
            self.truth_positive_threshold,
            (int, float),
        ):
            raise ConfidenceGenerationInputError(
                "truth_positive_threshold must be numeric"
            )
        if not math.isfinite(self.truth_positive_threshold):
            raise ConfidenceGenerationInputError(
                "truth_positive_threshold must be finite"
            )
        _validate_positive_int("max_document_chars", self.max_document_chars)


@dataclass(frozen=True)
class ConfidenceGenerationResult:
    output_path: Path
    input_count: int
    generated_count: int
    skipped_count: int
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class AnswerGenerationSample:
    id: str
    query: str
    context: str
    gold_answer: str | list[str]
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class QueryAnswerabilityGenerationSample:
    id: str
    query: str
    gold_answer: str | list[str]
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class QueryContextAnswerabilityGenerationSample:
    id: str
    query: str
    context: str
    gold_answer: str | list[str]
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RelevanceGenerationSample:
    id: str
    query: str
    document: str
    relevance_label: int | float | bool
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _validate_common_config(
    *,
    provider: object,
    overwrite: bool,
    resume: bool,
    max_items: int | None,
    source: str | None,
) -> None:
    if not callable(getattr(provider, "complete", None)):
        raise ConfidenceGenerationInputError("provider must define complete()")
    if overwrite and resume:
        raise ConfidenceGenerationInputError("overwrite and resume cannot both be true")
    if max_items is not None:
        _validate_positive_int("max_items", max_items)
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ConfidenceGenerationInputError("source must be a non-empty string")


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfidenceGenerationInputError(f"{name} must be an int")
    if value < 1:
        raise ConfidenceGenerationInputError(f"{name} must be >= 1")


def _validate_no_answer_value(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfidenceGenerationInputError(
            "no_answer_value must be a non-empty string"
        )
