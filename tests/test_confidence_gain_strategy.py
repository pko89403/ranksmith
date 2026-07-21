from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from ranksmith import AzureOpenAIReranker
from ranksmith.confidence import (
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    StructuralConfidenceEstimator,
    StructuralConfidenceResult,
    TaskType,
)
from ranksmith.confidence.scorer import ARTIFACT_SCHEMA_VERSION
from ranksmith.errors import (
    DocumentTooLongError,
    RerankInputError,
    RerankProviderError,
    RerankStrategyError,
)
from ranksmith.strategies import ConfidenceGainStrategy
from ranksmith.types import Document


@dataclass
class FakeEstimator:
    task_type: TaskType
    scores: list[Any]
    calls: list[object] | None = None

    def score(self, item: object) -> StructuralConfidenceResult:
        if self.calls is not None:
            self.calls.append(item)
        value = self.scores.pop(0)
        if isinstance(value, BaseException):
            raise value
        return StructuralConfidenceResult(
            score=value,
            task_type=self.task_type,
            feature_schema_version="structural-v1",
        )


class FakeGenerator:
    def __init__(
        self,
        *,
        base_answer: object = "base answer",
        context_answers: list[object] | None = None,
    ) -> None:
        self.base_answer = base_answer
        self.context_answers = context_answers or []
        self.query_calls: list[str] = []
        self.context_calls: list[tuple[str, str]] = []

    def answer_query(self, query: str) -> Any:
        self.query_calls.append(query)
        if isinstance(self.base_answer, BaseException):
            raise self.base_answer
        return self.base_answer

    def answer_with_context(self, query: str, context: str) -> Any:
        self.context_calls.append((query, context))
        answer = self.context_answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class ArtifactScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict_confidence(self, features: object) -> float:
        del features
        return self.scores.pop(0)


class ArtifactEncoder:
    encoder_name = "bert-base-uncased"
    encoder_revision = None
    tokenizer_name = "bert-base-uncased"
    tokenizer_revision = None

    def __init__(self, *, max_length: int) -> None:
        self.max_length = max_length

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        seed = float(len(text) % 7 + 1)
        hidden = [[seed + row * 0.01, row * 0.02, seed * 0.03] for row in range(40)]
        return hidden, [1] * len(hidden)


def _artifact_metadata(task_type: TaskType) -> dict[str, object]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scorer_type": "joblib-wrapper",
        "task_type": task_type,
        "encoder_name": "bert-base-uncased",
        "encoder_revision": None,
        "tokenizer_name": "bert-base-uncased",
        "tokenizer_revision": None,
        "input_template_version": "structural-template-v1",
        "feature_schema_version": "structural-v1",
        "feature_dim": 70,
        "feature_dtype": "float64",
        "max_length": 64,
        "granularity": "two_scale",
        "local_window_size": 5,
        "local_stride": 2,
        "score_output": "probability",
        "positive_class_index": 1,
    }


def _install_artifact_joblib(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: dict[Path, object],
) -> None:
    module = ModuleType("joblib")

    def load(path: str | Path) -> object:
        return artifacts[Path(path)]

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joblib", module)


def _strategy(
    *,
    base_scores: list[Any] | None = None,
    context_scores: list[Any] | None = None,
    generator: FakeGenerator | None = None,
) -> ConfidenceGainStrategy:
    return ConfidenceGainStrategy(
        base_estimator=FakeEstimator(
            task_type="query_answerability_confidence",
            scores=base_scores or [0.2],
        ),
        context_estimator=FakeEstimator(
            task_type="query_context_answerability_confidence",
            scores=context_scores or [0.7],
        ),
        answer_generator=generator or FakeGenerator(context_answers=["context answer"]),
    )


def _unused_model_client() -> Any:
    return object()


def test_confidence_gain_exports_are_submodule_only() -> None:
    strategies = importlib.import_module("ranksmith.strategies")
    root = importlib.import_module("ranksmith")

    assert strategies.AnswerGenerator is not None
    assert strategies.ConfidenceEstimator is not None
    assert strategies.ConfidenceGainResult is not None
    assert strategies.ConfidenceGainStrategy is not None
    assert not hasattr(root, "AnswerGenerator")
    assert not hasattr(root, "ConfidenceEstimator")
    assert not hasattr(root, "ConfidenceGainResult")
    assert not hasattr(root, "ConfidenceGainStrategy")


def test_confidence_gain_sorts_by_gain_desc_and_preserves_ties() -> None:
    generator = FakeGenerator(context_answers=["answer a", "answer b", "answer c"])
    strategy = _strategy(
        base_scores=[0.4],
        context_scores=[0.6, 0.8, 0.8],
        generator=generator,
    )
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
        Document(id="c", text="gamma"),
    ]

    results = strategy.rerank(
        query="Who?",
        documents=documents,
        model_client=object(),
    )

    assert [result.document.id for result in results] == ["b", "c", "a"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [1, 2, 0]
    assert [dict(result.metadata) for result in results] == [
        {
            "strategy": "confidence_gain",
            "algorithm": "confidence_gain",
            "base_confidence": 0.4,
            "context_confidence": 0.8,
            "confidence_gain": 0.4,
        },
        {
            "strategy": "confidence_gain",
            "algorithm": "confidence_gain",
            "base_confidence": 0.4,
            "context_confidence": 0.8,
            "confidence_gain": 0.4,
        },
        {
            "strategy": "confidence_gain",
            "algorithm": "confidence_gain",
            "base_confidence": 0.4,
            "context_confidence": 0.6,
            "confidence_gain": 0.19999999999999996,
        },
    ]


def test_confidence_gain_applies_top_k_after_sorting() -> None:
    strategy = _strategy(
        base_scores=[0.1],
        context_scores=[0.2, 0.9, 0.5],
        generator=FakeGenerator(context_answers=["a", "b", "c"]),
    )

    results = strategy.rerank(
        query="Who?",
        documents=[Document(text="a"), Document(text="b"), Document(text="c")],
        model_client=object(),
        top_k=2,
    )

    assert [result.original_index for result in results] == [1, 2]
    assert [result.rank for result in results] == [1, 2]


def test_confidence_gain_calls_answer_generator_expected_number_of_times() -> None:
    generator = FakeGenerator(context_answers=["a", "b"])
    strategy = _strategy(
        base_scores=[0.3],
        context_scores=[0.4, 0.5],
        generator=generator,
    )

    strategy.rerank(
        query="Who?",
        documents=[Document(text="alpha"), Document(text="beta")],
        model_client=object(),
    )

    assert generator.query_calls == ["Who?"]
    assert generator.context_calls == [("Who?", "alpha"), ("Who?", "beta")]


def test_confidence_gain_task_mismatch_fails_for_base_and_context() -> None:
    from ranksmith.strategies import ConfidenceGainStrategy  # noqa: PLC0415

    with pytest.raises(RerankInputError, match="base_estimator"):
        ConfidenceGainStrategy(
            base_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
        )

    with pytest.raises(RerankInputError, match="context_estimator"):
        ConfidenceGainStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
        )


def test_confidence_gain_invalid_algorithm_fails() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        ConfidenceGainStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
            algorithm=cast(Any, "other"),
        )


def test_confidence_gain_invalid_max_document_chars_fails() -> None:
    with pytest.raises(ValueError, match="max_document_chars"):
        ConfidenceGainStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
            max_document_chars=0,
        )


def test_confidence_gain_negative_top_k_fails() -> None:
    with pytest.raises(RerankInputError, match="top_k"):
        _strategy().rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
            top_k=-1,
        )


def test_confidence_gain_empty_query_fails() -> None:
    with pytest.raises(RerankInputError, match="query"):
        _strategy().rerank(
            query="  ",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_confidence_gain_empty_documents_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(generator=generator)

    assert strategy.rerank(query="Who?", documents=[], model_client=object()) == []
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_confidence_gain_long_document_fails() -> None:
    from ranksmith.strategies import ConfidenceGainStrategy  # noqa: PLC0415

    strategy = ConfidenceGainStrategy(
        base_estimator=FakeEstimator(
            task_type="query_answerability_confidence",
            scores=[0.2],
        ),
        context_estimator=FakeEstimator(
            task_type="query_context_answerability_confidence",
            scores=[0.5],
        ),
        answer_generator=FakeGenerator(context_answers=["a"]),
        max_document_chars=3,
    )

    with pytest.raises(DocumentTooLongError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="abcdef")],
            model_client=object(),
            top_k=None,
        )


@pytest.mark.parametrize("base_answer", ["", "   ", 123])
def test_confidence_gain_empty_or_non_string_answer_query_fails(
    base_answer: object,
) -> None:
    strategy = _strategy(
        generator=FakeGenerator(base_answer=base_answer, context_answers=["a"])
    )

    with pytest.raises(RerankProviderError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


@pytest.mark.parametrize("context_answer", ["", "   ", 123])
def test_confidence_gain_empty_or_non_string_answer_with_context_fails(
    context_answer: object,
) -> None:
    strategy = _strategy(generator=FakeGenerator(context_answers=[context_answer]))

    with pytest.raises(RerankProviderError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_confidence_gain_generator_unexpected_exception_wraps_provider_error() -> None:
    strategy = _strategy(
        generator=FakeGenerator(
            base_answer=TimeoutError("timeout"),
            context_answers=["a"],
        )
    )

    with pytest.raises(RerankProviderError) as exc_info:
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )

    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_confidence_gain_preserves_answer_query_provider_error() -> None:
    error = RerankProviderError("provider failed")
    strategy = _strategy(
        generator=FakeGenerator(
            base_answer=error,
            context_answers=["a"],
        )
    )

    with pytest.raises(RerankProviderError) as exc_info:
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_confidence_gain_preserves_answer_with_context_provider_error() -> None:
    error = RerankProviderError("provider failed")
    strategy = _strategy(
        generator=FakeGenerator(
            context_answers=[error],
        )
    )

    with pytest.raises(RerankProviderError) as exc_info:
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_confidence_gain_facade_preserves_generator_provider_error() -> None:
    error = RerankProviderError("provider failed")
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(
            generator=FakeGenerator(
                base_answer=error,
                context_answers=["a"],
            )
        ),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert exc_info.value is error


def test_confidence_gain_facade_wraps_unexpected_generator_error() -> None:
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(
            generator=FakeGenerator(
                base_answer=RuntimeError("generation failed"),
                context_answers=["a"],
            )
        ),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_confidence_gain_facade_preserves_estimator_provider_error() -> None:
    error = RerankProviderError("confidence provider failed")
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(base_scores=[error]),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert exc_info.value is error


def test_confidence_gain_facade_wraps_unexpected_estimator_error() -> None:
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(base_scores=[RuntimeError("confidence failed")]),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_confidence_gain_invalid_base_score_fails(score: object) -> None:
    strategy = _strategy(base_scores=[score], context_scores=[0.5])

    with pytest.raises(RerankStrategyError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_confidence_gain_invalid_context_score_fails(score: object) -> None:
    strategy = _strategy(context_scores=[score])

    with pytest.raises(RerankStrategyError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_confidence_gain_estimator_confidence_error_propagates() -> None:
    error = RuntimeError("confidence scorer failed")
    strategy = _strategy(base_scores=[error])

    with pytest.raises(RuntimeError, match="confidence scorer failed"):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_confidence_gain_passes_expected_confidence_inputs() -> None:
    base_calls: list[object] = []
    context_calls: list[object] = []
    from ranksmith.strategies import ConfidenceGainStrategy  # noqa: PLC0415

    strategy = ConfidenceGainStrategy(
        base_estimator=FakeEstimator(
            task_type="query_answerability_confidence",
            scores=[0.2],
            calls=base_calls,
        ),
        context_estimator=FakeEstimator(
            task_type="query_context_answerability_confidence",
            scores=[0.5],
            calls=context_calls,
        ),
        answer_generator=FakeGenerator(
            base_answer="base",
            context_answers=["context"],
        ),
    )

    strategy.rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )

    assert base_calls == [
        QueryAnswerabilityConfidenceInput(query="Who?", answer="base")
    ]
    assert context_calls == [
        QueryContextAnswerabilityConfidenceInput(
            query="Who?",
            context="alpha",
            answer="context",
        )
    ]


def test_confidence_gain_e2e_smoke_from_artifacts_through_azure_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_artifact_path = tmp_path / "query_answerability.joblib"
    context_artifact_path = tmp_path / "query_context_answerability.joblib"
    _install_artifact_joblib(
        monkeypatch,
        {
            base_artifact_path: {
                "metadata": _artifact_metadata("query_answerability_confidence"),
                "scorer": ArtifactScorer([0.3]),
            },
            context_artifact_path: {
                "metadata": _artifact_metadata(
                    "query_context_answerability_confidence"
                ),
                "scorer": ArtifactScorer([0.45, 0.9]),
            },
        },
    )

    def fake_from_pretrained(**kwargs: object) -> ArtifactEncoder:
        return ArtifactEncoder(max_length=cast(int, kwargs["max_length"]))

    monkeypatch.setattr(
        "ranksmith.confidence.structural.FrozenAutoEncoder.from_pretrained",
        fake_from_pretrained,
    )

    strategy = ConfidenceGainStrategy(
        base_estimator=StructuralConfidenceEstimator.from_artifact(base_artifact_path),
        context_estimator=StructuralConfidenceEstimator.from_artifact(
            context_artifact_path
        ),
        answer_generator=FakeGenerator(
            base_answer="base answer",
            context_answers=["low answer", "high answer"],
        ),
    )
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=strategy,
    )

    results = reranker.rerank(
        "who played karen in married to the mob?",
        [
            Document(
                id="similar-but-weak",
                text="Michelle Pfeiffer appears in the film.",
            ),
            Document(id="direct-evidence", text="Nancy Travis played Karen."),
        ],
    )

    assert [result.document.id for result in results] == [
        "direct-evidence",
        "similar-but-weak",
    ]
    assert [result.metadata["base_confidence"] for result in results] == [0.3, 0.3]
    assert [result.metadata["context_confidence"] for result in results] == [0.9, 0.45]
    assert [result.metadata["confidence_gain"] for result in results] == pytest.approx(
        [0.6, 0.15]
    )
