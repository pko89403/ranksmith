from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

TaskType: TypeAlias = Literal["answer_confidence", "judgment_confidence"]
ScoreOutput: TypeAlias = Literal["probability"]


@dataclass(frozen=True)
class AnswerConfidenceInput:
    context: str
    answer: str


@dataclass(frozen=True)
class JudgmentConfidenceInput:
    query: str
    document: str
    judgment: str


StructuralConfidenceInput: TypeAlias = AnswerConfidenceInput | JudgmentConfidenceInput


@dataclass(frozen=True)
class StructuralConfidenceResult:
    score: float
    task_type: TaskType
    feature_schema_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ScorerMetadata:
    artifact_schema_version: str
    scorer_type: str
    task_type: TaskType
    encoder_name: str
    encoder_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None
    input_template_version: str
    feature_schema_version: str
    feature_dim: int
    feature_dtype: str
    max_length: int
    granularity: str
    local_window_size: int
    local_stride: int
    score_output: ScoreOutput
    positive_class_index: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", dict(self.extra))


class StructuralConfidenceScorer(Protocol):
    metadata: ScorerMetadata

    def predict_confidence(self, features: Sequence[float]) -> float:
        """Return calibrated confidence probability for one feature vector."""
