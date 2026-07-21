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
from ranksmith.strategies import CBDRStrategy
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


def _artifact_strategy(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    base_scores: list[float],
    context_scores: list[float],
    generator: FakeGenerator,
    skip_threshold: float,
) -> CBDRStrategy:
    base_artifact_path = tmp_path / "query_answerability.joblib"
    context_artifact_path = tmp_path / "query_context_answerability.joblib"
    _install_artifact_joblib(
        monkeypatch,
        {
            base_artifact_path: {
                "metadata": _artifact_metadata("query_answerability_confidence"),
                "scorer": ArtifactScorer(base_scores),
            },
            context_artifact_path: {
                "metadata": _artifact_metadata(
                    "query_context_answerability_confidence"
                ),
                "scorer": ArtifactScorer(context_scores),
            },
        },
    )

    def fake_from_pretrained(**kwargs: object) -> ArtifactEncoder:
        return ArtifactEncoder(max_length=cast(int, kwargs["max_length"]))

    monkeypatch.setattr(
        "ranksmith.confidence.structural.FrozenAutoEncoder.from_pretrained",
        fake_from_pretrained,
    )

    return CBDRStrategy(
        base_estimator=StructuralConfidenceEstimator.from_artifact(base_artifact_path),
        context_estimator=StructuralConfidenceEstimator.from_artifact(
            context_artifact_path
        ),
        answer_generator=generator,
        skip_threshold=skip_threshold,
    )


def _strategy(
    *,
    base_scores: list[Any] | None = None,
    context_scores: list[Any] | None = None,
    generator: FakeGenerator | None = None,
    skip_threshold: float = 0.8,
    max_document_chars: int = 4000,
) -> CBDRStrategy:
    return CBDRStrategy(
        base_estimator=FakeEstimator(
            task_type="query_answerability_confidence",
            scores=base_scores or [0.2],
        ),
        context_estimator=FakeEstimator(
            task_type="query_context_answerability_confidence",
            scores=context_scores or [0.7],
        ),
        answer_generator=generator or FakeGenerator(context_answers=["context answer"]),
        skip_threshold=skip_threshold,
        max_document_chars=max_document_chars,
    )


def _unused_model_client() -> Any:
    return object()


def test_cbdr_exports_are_submodule_only() -> None:
    strategies = importlib.import_module("ranksmith.strategies")
    root = importlib.import_module("ranksmith")

    assert strategies.CBDRStrategy is not None
    assert not hasattr(root, "CBDRStrategy")


def test_cbdr_from_artifacts_builds_estimators_with_hf_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    base_estimator = FakeEstimator(
        task_type="query_answerability_confidence",
        scores=[0.9],
    )
    context_estimator = FakeEstimator(
        task_type="query_context_answerability_confidence",
        scores=[0.5],
    )

    def fake_from_artifact(path: str | Path, **kwargs: object) -> FakeEstimator:
        calls.append({"path": Path(path), **kwargs})
        return base_estimator if len(calls) == 1 else context_estimator

    monkeypatch.setattr(
        "ranksmith.confidence.StructuralConfidenceEstimator.from_artifact",
        fake_from_artifact,
    )
    generator = FakeGenerator(context_answers=[])

    strategy = CBDRStrategy.from_artifacts(
        base_artifact_path=tmp_path / "base.joblib",
        context_artifact_path=tmp_path / "context.joblib",
        base_metadata_path=tmp_path / "base.metadata.json",
        context_metadata_path=tmp_path / "context.metadata.json",
        answer_generator=generator,
        skip_threshold=0.7,
        max_document_chars=123,
        hf_token="token",
        cache_dir="/tmp/hf",
        device="cpu",
        local_files_only=True,
        max_length=128,
        allow_truncation=True,
    )

    assert strategy.base_estimator is base_estimator
    assert strategy.context_estimator is context_estimator
    assert strategy.answer_generator is generator
    assert strategy.skip_threshold == 0.7
    assert strategy.max_document_chars == 123
    assert calls == [
        {
            "path": tmp_path / "base.joblib",
            "metadata_path": tmp_path / "base.metadata.json",
            "task_type": "query_answerability_confidence",
            "hf_token": "token",
            "cache_dir": "/tmp/hf",
            "device": "cpu",
            "local_files_only": True,
            "max_length": 128,
            "allow_truncation": True,
        },
        {
            "path": tmp_path / "context.joblib",
            "metadata_path": tmp_path / "context.metadata.json",
            "task_type": "query_context_answerability_confidence",
            "hf_token": "token",
            "cache_dir": "/tmp/hf",
            "device": "cpu",
            "local_files_only": True,
            "max_length": 128,
            "allow_truncation": True,
        },
    ]


def test_cbdr_empty_documents_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(generator=generator)

    assert (
        strategy.rerank(
            query="Who?",
            documents=[],
            model_client=object(),
        )
        == []
    )
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_cbdr_top_k_zero_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=["a"])
    strategy = _strategy(generator=generator)

    assert (
        strategy.rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
            top_k=0,
        )
        == []
    )
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_cbdr_skip_path_preserves_original_order_and_metadata() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(base_scores=[0.91], generator=generator, skip_threshold=0.8)
    documents = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
    ]

    results = strategy.rerank(query="Who?", documents=documents, model_client=object())

    assert [result.document.id for result in results] == ["a", "b"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [0, 1]
    assert [dict(result.metadata) for result in results] == [
        {
            "strategy": "cbdr",
            "algorithm": "cbdr",
            "cbdr_skipped": True,
            "base_confidence": 0.91,
            "skip_threshold": 0.8,
            "context_confidence": None,
            "confidence_gain": None,
        },
        {
            "strategy": "cbdr",
            "algorithm": "cbdr",
            "cbdr_skipped": True,
            "base_confidence": 0.91,
            "skip_threshold": 0.8,
            "context_confidence": None,
            "confidence_gain": None,
        },
    ]
    assert generator.query_calls == ["Who?"]
    assert generator.context_calls == []


def test_cbdr_skip_path_applies_top_k_after_original_order() -> None:
    strategy = _strategy(base_scores=[0.9], skip_threshold=0.8)

    results = strategy.rerank(
        query="Who?",
        documents=[
            Document(id="a", text="alpha"),
            Document(id="b", text="beta"),
        ],
        model_client=object(),
        top_k=1,
    )

    assert [result.document.id for result in results] == ["a"]
    assert [result.rank for result in results] == [1]
    assert [result.original_index for result in results] == [0]


def test_cbdr_skip_path_does_not_validate_long_documents() -> None:
    strategy = _strategy(
        base_scores=[0.9],
        context_scores=[],
        generator=FakeGenerator(context_answers=[]),
        skip_threshold=0.8,
        max_document_chars=3,
    )

    results = strategy.rerank(
        query="Who?",
        documents=[Document(text="abcdef")],
        model_client=object(),
    )

    assert len(results) == 1
    assert results[0].metadata["cbdr_skipped"] is True


def test_cbdr_rerank_path_sorts_by_gain_and_preserves_ties() -> None:
    generator = FakeGenerator(context_answers=["answer a", "answer b", "answer c"])
    strategy = _strategy(
        base_scores=[0.4],
        context_scores=[0.6, 0.8, 0.8],
        generator=generator,
        skip_threshold=0.9,
    )

    results = strategy.rerank(
        query="Who?",
        documents=[
            Document(id="a", text="alpha"),
            Document(id="b", text="beta"),
            Document(id="c", text="gamma"),
        ],
        model_client=object(),
        top_k=2,
    )

    assert [result.document.id for result in results] == ["b", "c"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [1, 2]
    assert [result.metadata["cbdr_skipped"] for result in results] == [False, False]
    assert [result.metadata["base_confidence"] for result in results] == [0.4, 0.4]
    assert [result.metadata["context_confidence"] for result in results] == [0.8, 0.8]
    assert [result.metadata["confidence_gain"] for result in results] == pytest.approx(
        [0.4, 0.4]
    )
    assert generator.context_calls == [
        ("Who?", "alpha"),
        ("Who?", "beta"),
        ("Who?", "gamma"),
    ]


@pytest.mark.parametrize("skip_threshold", [math.nan, math.inf, -0.1, 1.1, True])
def test_cbdr_invalid_skip_threshold_fails(skip_threshold: object) -> None:
    with pytest.raises(ValueError, match="skip_threshold"):
        CBDRStrategy(
            base_estimator=FakeEstimator(
                task_type="query_answerability_confidence",
                scores=[0.1],
            ),
            context_estimator=FakeEstimator(
                task_type="query_context_answerability_confidence",
                scores=[0.2],
            ),
            answer_generator=FakeGenerator(),
            skip_threshold=cast(float, skip_threshold),
        )


def test_cbdr_threshold_zero_always_skips_non_empty_documents() -> None:
    strategy = _strategy(base_scores=[0.0], skip_threshold=0.0)

    results = strategy.rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )

    assert results[0].metadata["cbdr_skipped"] is True


def test_cbdr_threshold_one_skips_only_at_exact_one() -> None:
    skipped = _strategy(base_scores=[1.0], skip_threshold=1.0).rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )
    reranked = _strategy(
        base_scores=[0.999],
        context_scores=[1.0],
        generator=FakeGenerator(context_answers=["a"]),
        skip_threshold=1.0,
    ).rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
    )

    assert skipped[0].metadata["cbdr_skipped"] is True
    assert reranked[0].metadata["cbdr_skipped"] is False


def test_cbdr_rerank_path_validates_long_documents() -> None:
    strategy = _strategy(base_scores=[0.2], skip_threshold=0.8, max_document_chars=3)

    with pytest.raises(DocumentTooLongError):
        strategy.rerank(
            query="Who?",
            documents=[Document(text="abcdef")],
            model_client=object(),
        )


def test_cbdr_empty_query_fails() -> None:
    with pytest.raises(RerankInputError, match="query"):
        _strategy().rerank(
            query="  ",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_negative_top_k_fails() -> None:
    with pytest.raises(RerankInputError, match="top_k"):
        _strategy().rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
            top_k=-1,
        )


def test_cbdr_invalid_task_types_fail() -> None:
    with pytest.raises(RerankInputError, match="base_estimator"):
        CBDRStrategy(
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
        CBDRStrategy(
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


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_cbdr_invalid_base_score_fails(score: object) -> None:
    with pytest.raises(RerankStrategyError):
        _strategy(base_scores=[score]).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, "0.5", True])
def test_cbdr_invalid_context_score_fails(score: object) -> None:
    with pytest.raises(RerankStrategyError):
        _strategy(
            base_scores=[0.1],
            context_scores=[score],
            skip_threshold=0.8,
        ).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_answer_generator_empty_output_fails() -> None:
    with pytest.raises(RerankProviderError):
        _strategy(generator=FakeGenerator(base_answer="")).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_generator_unexpected_exception_wraps_provider_error() -> None:
    with pytest.raises(RerankProviderError) as exc_info:
        _strategy(generator=FakeGenerator(base_answer=RuntimeError("boom"))).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_cbdr_direct_estimator_unexpected_exception_propagates() -> None:
    error = RuntimeError("confidence failed")

    with pytest.raises(RuntimeError, match="confidence failed"):
        _strategy(base_scores=[error]).rerank(
            query="Who?",
            documents=[Document(text="alpha")],
            model_client=object(),
        )


def test_cbdr_facade_wraps_unexpected_estimator_error() -> None:
    reranker = AzureOpenAIReranker(
        model_client=_unused_model_client(),
        strategy=_strategy(base_scores=[RuntimeError("confidence failed")]),
    )

    with pytest.raises(RerankProviderError) as exc_info:
        reranker.rerank("Who?", [Document(text="alpha")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_cbdr_artifact_e2e_skip_path_through_azure_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = _artifact_strategy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        base_scores=[0.91],
        context_scores=[],
        generator=FakeGenerator(context_answers=[]),
        skip_threshold=0.8,
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
        "similar-but-weak",
        "direct-evidence",
    ]
    assert [result.metadata["cbdr_skipped"] for result in results] == [True, True]
    assert [result.metadata["base_confidence"] for result in results] == [0.91, 0.91]
    assert [result.metadata["context_confidence"] for result in results] == [
        None,
        None,
    ]
    assert [result.metadata["confidence_gain"] for result in results] == [None, None]


def test_cbdr_artifact_e2e_rerank_path_through_azure_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = _artifact_strategy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        base_scores=[0.3],
        context_scores=[0.45, 0.9],
        generator=FakeGenerator(
            base_answer="base answer",
            context_answers=["low answer", "high answer"],
        ),
        skip_threshold=0.8,
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
    assert [result.metadata["cbdr_skipped"] for result in results] == [False, False]
    assert [result.metadata["base_confidence"] for result in results] == [0.3, 0.3]
    assert [result.metadata["context_confidence"] for result in results] == [0.9, 0.45]
    assert [result.metadata["confidence_gain"] for result in results] == pytest.approx(
        [0.6, 0.15]
    )
