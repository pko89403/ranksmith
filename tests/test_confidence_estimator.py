from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceArtifactError,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceEstimator,
    StructuralConfidenceResult,
    TaskType,
    _encoder,
)
from ranksmith.confidence._structural import _require_result


@dataclass(frozen=True)
class FakeEncoder:
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_revision: str | None = None
    max_length: int = 64
    cache_dir: str | None = "/tmp/private-cache"

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        assert text
        hidden = [[float(row + col) for col in range(4)] for row in range(8)]
        mask = [1] * 8
        return hidden, mask


class FakeScorer:
    def __init__(
        self,
        *,
        score: object = 0.75,
        task_type: TaskType = "answer_confidence",
        max_length: int = 64,
        scorer_type: str = "fake",
        encoder_name: str = "bert-base-uncased",
        tokenizer_name: str = "bert-base-uncased",
    ) -> None:
        self.metadata = ScorerMetadata(
            artifact_schema_version="structural-artifact-v1",
            scorer_type=scorer_type,
            task_type=task_type,
            encoder_name=encoder_name,
            encoder_revision=None,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=None,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
            feature_dtype="float64",
            max_length=max_length,
            granularity="two_scale",
            local_window_size=5,
            local_stride=2,
            score_output="probability",
            positive_class_index=1,
        )
        self.score = score
        self.last_features: list[float] | None = None

    def predict_confidence(self, features: Sequence[float]) -> float:
        assert len(features) == 70
        self.last_features = list(features)
        return self.score  # type: ignore[return-value]


def test_estimator_is_frozen() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(FrozenInstanceError):
        estimator.task_type = "judgment_confidence"  # type: ignore[misc]


def test_estimator_scores_answer_input() -> None:
    scorer = FakeScorer(score=0.75)
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=scorer,
        task_type="answer_confidence",
    )

    result = estimator.score(AnswerConfidenceInput(context="context", answer="answer"))

    assert result.score == 0.75
    assert result.task_type == "answer_confidence"
    assert result.feature_schema_version == "structural-v1"
    assert scorer.last_features is not None
    assert result.metadata["encoder_name"] == "bert-base-uncased"
    assert result.metadata["tokenizer_name"] == "bert-base-uncased"
    assert result.metadata["max_length"] == 64
    assert result.metadata["feature_dim"] == 70
    assert result.metadata["feature_dtype"] == "float64"
    assert result.metadata["granularity"] == "two_scale"
    assert result.metadata["local_window_size"] == 5
    assert result.metadata["local_stride"] == 2
    assert result.metadata["input_template_version"] == "structural-template-v1"
    assert result.metadata["scorer_type"] == "fake"
    assert result.metadata["artifact_schema_version"] == "structural-artifact-v1"


def test_estimator_scores_judgment_input_with_matching_metadata() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(score=0.6, task_type="judgment_confidence"),
        task_type="judgment_confidence",
    )

    result = estimator.score(
        JudgmentConfidenceInput(
            query="query",
            document="document",
            judgment="direct evidence",
        )
    )

    assert result.score == 0.6
    assert result.task_type == "judgment_confidence"


def test_direct_constructor_rejects_task_type_metadata_mismatch() -> None:
    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator(
            encoder=FakeEncoder(),
            scorer=FakeScorer(task_type="judgment_confidence"),
            task_type="answer_confidence",
        )


def test_direct_constructor_rejects_max_length_metadata_mismatch() -> None:
    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator(
            encoder=FakeEncoder(max_length=64),
            scorer=FakeScorer(max_length=128),
            task_type="answer_confidence",
        )


def test_estimator_rejects_wrong_input_type() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score(
            JudgmentConfidenceInput(
                query="query",
                document="document",
                judgment="direct evidence",
            )
        )


def test_from_pretrained_rejects_scorer_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        captured.update(kwargs)
        return FakeEncoder()

    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator.from_pretrained(
            encoder_name="bert-base-uncased",
            scorer=FakeScorer(max_length=128),
            task_type="answer_confidence",
            max_length=64,
            hf_token="secret-token",
            cache_dir="/tmp/private-cache",
        )

    assert captured["hf_token"] == "secret-token"
    assert captured["cache_dir"] == "/tmp/private-cache"


def test_from_pretrained_accepts_matching_scorer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        assert kwargs["tokenizer_name"] is None
        return FakeEncoder()

    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    estimator = StructuralConfidenceEstimator.from_pretrained(
        encoder_name="bert-base-uncased",
        scorer=FakeScorer(),
        task_type="answer_confidence",
        max_length=64,
    )

    assert estimator.encoder.encoder_name == "bert-base-uncased"


def test_from_artifact_uses_scorer_metadata_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_encoder_kwargs: dict[str, object] = {}
    captured_loader_kwargs: dict[str, object] = {}
    scorer = FakeScorer(task_type="judgment_confidence", max_length=64)

    def fake_load_lightgbm_scorer(
        artifact_path: object,
        *,
        metadata_path: object = None,
    ) -> FakeScorer:
        captured_loader_kwargs["artifact_path"] = artifact_path
        captured_loader_kwargs["metadata_path"] = metadata_path
        return scorer

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        captured_encoder_kwargs.update(kwargs)
        return FakeEncoder(max_length=64)

    monkeypatch.setattr(
        "ranksmith.confidence._structural.load_lightgbm_scorer",
        fake_load_lightgbm_scorer,
    )
    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    artifact_path = tmp_path / "judgment_confidence.joblib"
    metadata_path = tmp_path / "metadata.json"
    estimator = StructuralConfidenceEstimator.from_artifact(
        artifact_path,
        metadata_path=metadata_path,
        hf_token="secret-token",
        cache_dir="/tmp/private-cache",
    )

    assert estimator.task_type == "judgment_confidence"
    assert captured_loader_kwargs == {
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
    }
    assert captured_encoder_kwargs["encoder_name"] == "bert-base-uncased"
    assert captured_encoder_kwargs["encoder_revision"] is None
    assert captured_encoder_kwargs["tokenizer_name"] == "bert-base-uncased"
    assert captured_encoder_kwargs["tokenizer_revision"] is None
    assert captured_encoder_kwargs["max_length"] == 64
    assert captured_encoder_kwargs["hf_token"] == "secret-token"
    assert captured_encoder_kwargs["cache_dir"] == "/tmp/private-cache"


def test_from_artifact_rejects_override_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_lightgbm_scorer(
        artifact_path: object,
        *,
        metadata_path: object = None,
    ) -> FakeScorer:
        del artifact_path, metadata_path
        return FakeScorer(max_length=64)

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        del kwargs
        raise AssertionError("encoder should not load before metadata validation")

    monkeypatch.setattr(
        "ranksmith.confidence._structural.load_lightgbm_scorer",
        fake_load_lightgbm_scorer,
    )
    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator.from_artifact(
            tmp_path / "answer_confidence.joblib",
            max_length=128,
        )


def test_from_artifact_rejects_empty_encoder_name_override_before_encoder_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_lightgbm_scorer(
        artifact_path: object,
        *,
        metadata_path: object = None,
    ) -> FakeScorer:
        del artifact_path, metadata_path
        return FakeScorer(encoder_name="bert-base-uncased")

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        del kwargs
        raise AssertionError("encoder should not load before metadata validation")

    monkeypatch.setattr(
        "ranksmith.confidence._structural.load_lightgbm_scorer",
        fake_load_lightgbm_scorer,
    )
    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator.from_artifact(
            tmp_path / "answer_confidence.joblib",
            encoder_name="",
        )


def test_from_artifact_rejects_empty_tokenizer_name_override_before_encoder_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_lightgbm_scorer(
        artifact_path: object,
        *,
        metadata_path: object = None,
    ) -> FakeScorer:
        del artifact_path, metadata_path
        return FakeScorer(tokenizer_name="bert-base-uncased")

    def fake_from_pretrained(**kwargs: object) -> FakeEncoder:
        del kwargs
        raise AssertionError("encoder should not load before metadata validation")

    monkeypatch.setattr(
        "ranksmith.confidence._structural.load_lightgbm_scorer",
        fake_load_lightgbm_scorer,
    )
    monkeypatch.setattr(
        _encoder.FrozenAutoEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(ConfidenceArtifactError):
        StructuralConfidenceEstimator.from_artifact(
            tmp_path / "answer_confidence.joblib",
            tokenizer_name="",
        )


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, "0.5", None])
def test_estimator_rejects_invalid_scores(score: object) -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(score=score),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceArtifactError):
        estimator.score(AnswerConfidenceInput(context="context", answer="answer"))


def test_result_metadata_excludes_sensitive_and_heavy_values() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    result = estimator.score(AnswerConfidenceInput(context="context", answer="answer"))

    forbidden = {
        "hf_token",
        "token",
        "cache_dir",
        "model",
        "tokenizer",
        "features",
        "feature_vector",
        "local_path",
    }
    assert forbidden.isdisjoint(result.metadata)


@dataclass(frozen=True)
class CountingEncoder:
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_revision: str | None = None
    max_length: int = 64
    calls: list[str] | None = None

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        assert self.calls is not None
        self.calls.append(text)
        signal = float(len(self.calls))
        hidden = [[signal + float(row + col) for col in range(4)] for row in range(8)]
        mask = [1] * 8
        return hidden, mask


class CyclingScorer(FakeScorer):
    def __init__(self, scores: list[float], *, task_type: TaskType) -> None:
        super().__init__(score=0.0, task_type=task_type)
        self._scores = scores
        self._index = 0

    def predict_confidence(self, features: Sequence[float]) -> float:
        assert len(features) == 70
        score = self._scores[self._index]
        self._index += 1
        return score


def test_score_batch_preserves_input_order_with_chunking() -> None:
    calls: list[str] = []
    estimator = StructuralConfidenceEstimator(
        encoder=CountingEncoder(calls=calls),
        scorer=CyclingScorer([0.1, 0.2, 0.3], task_type="judgment_confidence"),
        task_type="judgment_confidence",
    )

    results = estimator.score_batch(
        [
            JudgmentConfidenceInput(query="q", document="doc a", judgment="direct"),
            JudgmentConfidenceInput(query="q", document="doc b", judgment="partial"),
            JudgmentConfidenceInput(query="q", document="doc c", judgment="none"),
        ],
        batch_size=2,
    )

    assert [result.score for result in results] == [0.1, 0.2, 0.3]
    assert len(calls) == 3
    assert "doc a" in calls[0]
    assert "doc b" in calls[1]
    assert "doc c" in calls[2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"max_workers": 0},
        {"max_batch_items": 0},
        {"batch_size": True},
        {"max_workers": True},
        {"max_batch_items": True},
        {"batch_size": 1.5},
        {"max_workers": 1.5},
        {"max_batch_items": 1.5},
    ],
)
def test_score_batch_rejects_invalid_options(kwargs: dict[str, object]) -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score_batch(
            [AnswerConfidenceInput(context="context", answer="answer")],
            **kwargs,
        )


def test_score_batch_rejects_empty_items() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score_batch([])


def test_score_batch_rejects_too_many_items() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score_batch(
            [
                AnswerConfidenceInput(context="context 1", answer="answer 1"),
                AnswerConfidenceInput(context="context 2", answer="answer 2"),
            ],
            max_batch_items=1,
        )


def test_score_batch_parallel_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[int] = []

    class FakeFuture:
        def __init__(
            self,
            index: int,
            result: tuple[int, StructuralConfidenceResult],
        ) -> None:
            self.index = index
            self._result = result

        def result(self) -> tuple[int, StructuralConfidenceResult]:
            return self._result

        def cancel(self) -> bool:
            return False

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def submit(
            self,
            fn: object,
            value: tuple[int, JudgmentConfidenceInput],
        ) -> FakeFuture:
            submitted.append(value[0])
            return FakeFuture(value[0], fn(value))  # type: ignore[misc]

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool = False,
        ) -> None:
            assert wait is True
            assert cancel_futures is False

    def fake_as_completed(futures: object) -> list[FakeFuture]:
        return list(reversed(list(futures)))  # type: ignore[arg-type]

    def fake_score(
        self: StructuralConfidenceEstimator,
        item: JudgmentConfidenceInput,
    ) -> StructuralConfidenceResult:
        del self
        index = int(item.document.removeprefix("doc "))
        return StructuralConfidenceResult(
            score=(index + 1) / 10,
            task_type="judgment_confidence",
            feature_schema_version="structural-v1",
            metadata={"document": item.document},
        )

    monkeypatch.setattr(StructuralConfidenceEstimator, "score", fake_score)
    monkeypatch.setattr(
        "ranksmith.confidence._structural.ThreadPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        "ranksmith.confidence._structural.as_completed",
        fake_as_completed,
        raising=False,
    )
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(task_type="judgment_confidence"),
        task_type="judgment_confidence",
    )

    results = estimator.score_batch(
        [
            JudgmentConfidenceInput(query="q", document=f"doc {index}", judgment="j")
            for index in range(6)
        ],
        batch_size=3,
        max_workers=2,
    )

    assert len(results) == 6
    assert [result.score for result in results] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert [result.task_type for result in results] == ["judgment_confidence"] * 6
    assert submitted == [0, 1, 2, 0, 1, 2]


def test_score_batch_parallel_propagates_worker_error() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(task_type="answer_confidence"),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score_batch(
            [
                AnswerConfidenceInput(context="context", answer="answer"),
                JudgmentConfidenceInput(
                    query="query",
                    document="document",
                    judgment="direct evidence",
                ),
            ],
            max_workers=2,
        )


def test_require_result_rejects_missing_parallel_result() -> None:
    with pytest.raises(
        ConfidenceArtifactError,
        match=r"^parallel confidence result is missing\.$",
    ):
        _require_result(None)


def test_score_batch_parallel_fast_fails_without_waiting_for_slow_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(
        self: StructuralConfidenceEstimator,
        item: AnswerConfidenceInput,
    ) -> StructuralConfidenceResult:
        del self
        if item.context == "slow":
            time.sleep(0.5)
            return StructuralConfidenceResult(
                score=0.1,
                task_type="answer_confidence",
                feature_schema_version="structural-v1",
                metadata={},
            )
        raise ConfidenceInputError("fast failure")

    monkeypatch.setattr(StructuralConfidenceEstimator, "score", fake_score)
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    started_at = time.perf_counter()
    with pytest.raises(ConfidenceInputError, match="fast failure"):
        estimator.score_batch(
            [
                AnswerConfidenceInput(context="slow", answer="answer 0"),
                AnswerConfidenceInput(context="fast", answer="answer 1"),
            ],
            max_workers=2,
        )
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.3


def test_score_batch_parallel_raises_first_completed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[int] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeFuture:
        def __init__(self, index: int) -> None:
            self.index = index

        def result(self) -> tuple[int, StructuralConfidenceResult]:
            if self.index == 0:
                raise AssertionError("slow earlier future should not be consumed first")
            raise ConfidenceInputError("fast failure")

        def cancel(self) -> bool:
            cancelled.append(self.index)
            return True

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def submit(
            self,
            fn: object,
            value: tuple[int, AnswerConfidenceInput],
        ) -> FakeFuture:
            del fn
            return FakeFuture(value[0])

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def fake_as_completed(futures: object) -> list[FakeFuture]:
        by_index = {future.index: future for future in futures}  # type: ignore[union-attr]
        return [by_index[1], by_index[0]]

    monkeypatch.setattr(
        "ranksmith.confidence._structural.ThreadPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        "ranksmith.confidence._structural.as_completed",
        fake_as_completed,
        raising=False,
    )
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceInputError, match="fast failure"):
        estimator.score_batch(
            [
                AnswerConfidenceInput(context="context 0", answer="answer 0"),
                AnswerConfidenceInput(context="context 1", answer="answer 1"),
            ],
            max_workers=2,
        )

    assert cancelled == [0, 1]
    assert shutdown_calls == [(False, True)]


def test_score_batch_parallel_uses_executor_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeFuture:
        def __init__(self, result: object) -> None:
            self._result = result

        def result(self) -> object:
            return self._result

        def cancel(self) -> bool:
            return False

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            called.append(max_workers)

        def submit(self, fn: object, value: object) -> FakeFuture:
            return FakeFuture(fn(value))  # type: ignore[misc]

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def fake_as_completed(futures: object) -> object:
        return futures

    monkeypatch.setattr(
        "ranksmith.confidence._structural.ThreadPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        "ranksmith.confidence._structural.as_completed",
        fake_as_completed,
        raising=False,
    )
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    estimator.score_batch(
        [AnswerConfidenceInput(context="context", answer="answer")],
        max_workers=3,
    )

    assert called == [3]
    assert shutdown_calls == [(True, False)]
