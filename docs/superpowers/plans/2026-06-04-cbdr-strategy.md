# CBDR Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sync `CBDRStrategy` that skips context reranking when `Conf(Q) >= skip_threshold`, otherwise reranks by `Conf(Q+C)-Conf(Q)`.

**Architecture:** Reuse the existing query-only and query+context answerability confidence contracts. Keep CBDR as a separate Strategy, not a subclass of `ConfidenceGainStrategy`, while sharing private helper functions for answer calls, task validation, score validation, and confidence gain scoring. Export only through `ranksmith.strategies`, and register it as a built-in sync strategy in `AzureOpenAIReranker`.

**Tech Stack:** Python 3.10+, dataclasses, pytest, ranksmith confidence runtime, existing `AzureOpenAIReranker` facade.

---

## File Structure

- Modify: `src/ranksmith/strategies/_confidence_gain.py`
  - Expose private helper functions for task validation, answer generation calls, base scoring, context scoring, and confidence score validation.
  - Keep public exports unchanged.
- Create: `src/ranksmith/strategies/_cbdr.py`
  - Implement sync `CBDRStrategy`.
  - Use shared helpers from `_confidence_gain.py`.
- Modify: `src/ranksmith/strategies/__init__.py`
  - Export `CBDRStrategy` from the strategies submodule only.
- Modify: `src/ranksmith/azure.py`
  - Import `CBDRStrategy`.
  - Treat `CBDRStrategy` as built-in sync strategy for facade exception handling.
- Create: `tests/test_cbdr_strategy.py`
  - Cover skip path, rerank path, no-call paths, invalid inputs, facade behavior, and artifact-load E2E smoke for both paths.
- Modify: `docs/wiki/02_architecture.md`
  - Add CBDR strategy and algorithm placement.
- Modify: `README.md`
  - Add benchmark-free CBDR usage docs.
- Modify: `README.ko.md`
  - Mirror README structure and content in Korean.
- Modify: `docs/specs/spec_cbdr_strategy.md`
  - Mark implementation checklist items completed only after exit code 0 with no test failures.

---

### Task 1: Shared Confidence Gain Helpers

**Files:**
- Modify: `src/ranksmith/strategies/_confidence_gain.py`
- Test: `tests/test_confidence_gain_strategy.py`

- [ ] **Step 1: Run existing confidence gain tests before changing helpers**

Run:

```bash
uv run pytest tests/test_confidence_gain_strategy.py -q
```

Expected:

```text
39 passed
```

- [ ] **Step 2: Refactor helper functions without behavior change**

Modify `src/ranksmith/strategies/_confidence_gain.py` so shared private helpers have these signatures and behavior:

```python
def _validate_estimator_tasks(
    *,
    base_estimator: ConfidenceEstimator,
    context_estimator: ConfidenceEstimator,
) -> None:
    if base_estimator.task_type != QUERY_ANSWERABILITY_TASK:
        raise RerankInputError(
            f"base_estimator task_type must be {QUERY_ANSWERABILITY_TASK!r}"
        )
    if context_estimator.task_type != QUERY_CONTEXT_ANSWERABILITY_TASK:
        raise RerankInputError(
            "context_estimator task_type must be "
            f"{QUERY_CONTEXT_ANSWERABILITY_TASK!r}"
        )


def _score_base_answerability(
    *,
    estimator: ConfidenceEstimator,
    query: str,
    answer: str,
) -> StructuralConfidenceResult:
    return estimator.score(
        QueryAnswerabilityConfidenceInput(query=query, answer=answer)
    )


def _score_context_answerability(
    *,
    estimator: ConfidenceEstimator,
    query: str,
    context: str,
    answer: str,
) -> StructuralConfidenceResult:
    return estimator.score(
        QueryContextAnswerabilityConfidenceInput(
            query=query,
            context=context,
            answer=answer,
        )
    )


def _confidence_gain(
    *,
    base_score: float,
    context_score: float,
) -> float:
    gain = context_score - base_score
    if not math.isfinite(gain) or gain < -1.0 or gain > 1.0:
        raise RerankStrategyError("confidence gain must be finite in [-1, 1]")
    return gain
```

Keep `_call_answer_query`, `_call_answer_with_context`, `_validate_answer`, and `_validate_confidence_score` private and reusable from `_cbdr.py`.

Update `ConfidenceGainStrategy.__post_init__()` to call `_validate_estimator_tasks(...)`.

Update `ConfidenceGainStrategy.rerank()` to call `_score_base_answerability(...)`, `_score_context_answerability(...)`, and `_confidence_gain(...)`.

- [ ] **Step 3: Run confidence gain regression tests**

Run:

```bash
uv run pytest tests/test_confidence_gain_strategy.py -q
```

Expected:

```text
39 passed
```

- [ ] **Step 4: Commit helper refactor**

Run:

```bash
git add src/ranksmith/strategies/_confidence_gain.py tests/test_confidence_gain_strategy.py
git commit -m "refactor: share confidence gain helpers"
```

Expected:

```text
commit succeeds
```

---

### Task 2: CBDR Unit Tests and Strategy

**Files:**
- Create: `src/ranksmith/strategies/_cbdr.py`
- Modify: `src/ranksmith/strategies/__init__.py`
- Modify: `src/ranksmith/azure.py`
- Create: `tests/test_cbdr_strategy.py`

- [ ] **Step 1: Write failing CBDR tests**

Create `tests/test_cbdr_strategy.py` with this structure:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pytest

from ranksmith import AzureOpenAIReranker
from ranksmith.confidence import StructuralConfidenceResult, TaskType
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
```

Add these tests in the same file:

```python
def test_cbdr_exports_are_submodule_only() -> None:
    import importlib

    strategies = importlib.import_module("ranksmith.strategies")
    root = importlib.import_module("ranksmith")

    assert strategies.CBDRStrategy is not None
    assert not hasattr(root, "CBDRStrategy")


def test_cbdr_empty_documents_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=[])
    strategy = _strategy(generator=generator)

    assert strategy.rerank(
        query="Who?",
        documents=[],
        model_client=object(),
    ) == []
    assert generator.query_calls == []
    assert generator.context_calls == []


def test_cbdr_top_k_zero_returns_empty_without_calls() -> None:
    generator = FakeGenerator(context_answers=["a"])
    strategy = _strategy(generator=generator)

    assert strategy.rerank(
        query="Who?",
        documents=[Document(text="alpha")],
        model_client=object(),
        top_k=0,
    ) == []
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
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
uv run pytest tests/test_cbdr_strategy.py -q
```

Expected:

```text
ImportError: cannot import name 'CBDRStrategy'
```

- [ ] **Step 3: Implement `CBDRStrategy`**

Create `src/ranksmith/strategies/_cbdr.py`:

```python
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from ranksmith.errors import RerankInputError, RerankStrategyError
from ranksmith.types import Document, RerankResult

from ._common import validate_documents_max_chars, validate_top_k
from ._confidence_gain import (
    AnswerGenerator,
    ConfidenceEstimator,
    _call_answer_query,
    _call_answer_with_context,
    _confidence_gain,
    _score_base_answerability,
    _score_context_answerability,
    _validate_confidence_score,
    _validate_estimator_tasks,
)

CBDRAlgorithm = Literal["cbdr"]


@dataclass(frozen=True)
class CBDRStrategy:
    base_estimator: ConfidenceEstimator
    context_estimator: ConfidenceEstimator
    answer_generator: AnswerGenerator
    skip_threshold: float = 0.8
    max_document_chars: int = 4000
    algorithm: CBDRAlgorithm = "cbdr"

    def __post_init__(self) -> None:
        if self.algorithm != "cbdr":
            raise ValueError('algorithm must be "cbdr"')
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")
        _validate_probability_config(self.skip_threshold, "skip_threshold")
        _validate_estimator_tasks(
            base_estimator=self.base_estimator,
            context_estimator=self.context_estimator,
        )

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del model_client
        validate_top_k(top_k)
        if query.strip() == "":
            raise RerankInputError("query must not be empty")
        if not documents or top_k == 0:
            return []

        base_answer = _call_answer_query(self.answer_generator, query)
        base_result = _score_base_answerability(
            estimator=self.base_estimator,
            query=query,
            answer=base_answer,
        )
        base_score = _validate_confidence_score(base_result.score, "base")

        if base_score >= self.skip_threshold:
            return _original_order_results(
                documents=documents,
                top_k=top_k,
                algorithm=self.algorithm,
                base_score=base_score,
                skip_threshold=self.skip_threshold,
            )

        validate_documents_max_chars(
            documents,
            max_document_chars=self.max_document_chars,
        )
        scored = []
        for original_index, document in enumerate(documents):
            context_answer = _call_answer_with_context(
                self.answer_generator,
                query,
                document.text,
            )
            context_result = _score_context_answerability(
                estimator=self.context_estimator,
                query=query,
                context=document.text,
                answer=context_answer,
            )
            context_score = _validate_confidence_score(
                context_result.score,
                "context",
            )
            scored.append(
                (
                    original_index,
                    context_score,
                    _confidence_gain(
                        base_score=base_score,
                        context_score=context_score,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[2], item[0]))
        if top_k is not None:
            scored = scored[:top_k]

        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "cbdr",
                    "algorithm": self.algorithm,
                    "cbdr_skipped": False,
                    "base_confidence": base_score,
                    "skip_threshold": self.skip_threshold,
                    "context_confidence": context_score,
                    "confidence_gain": gain,
                },
            )
            for rank, (original_index, context_score, gain) in enumerate(
                scored,
                start=1,
            )
        ]


def _validate_probability_config(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


def _original_order_results(
    *,
    documents: Sequence[Document],
    top_k: int | None,
    algorithm: CBDRAlgorithm,
    base_score: float,
    skip_threshold: float,
) -> list[RerankResult]:
    indexed = list(enumerate(documents))
    if top_k is not None:
        indexed = indexed[:top_k]
    return [
        RerankResult(
            document=document,
            rank=rank,
            original_index=original_index,
            metadata={
                "strategy": "cbdr",
                "algorithm": algorithm,
                "cbdr_skipped": True,
                "base_confidence": base_score,
                "skip_threshold": skip_threshold,
                "context_confidence": None,
                "confidence_gain": None,
            },
        )
        for rank, (original_index, document) in enumerate(indexed, start=1)
    ]
```

- [ ] **Step 4: Export CBDR strategy and register facade behavior**

Modify `src/ranksmith/strategies/__init__.py`:

```python
from ranksmith.strategies._cbdr import CBDRStrategy
```

Add `"CBDRStrategy"` to `__all__`.

Modify `src/ranksmith/azure.py`:

```python
from ranksmith.strategies import (
    AcuRankStrategy,
    AsyncAcuRankStrategy,
    AsyncListwiseStrategy,
    AsyncPairwiseStrategy,
    AsyncSetwiseStrategy,
    AsyncTourRankStrategy,
    CBDRStrategy,
    ConfidenceGainStrategy,
    ListwiseStrategy,
    PairwiseStrategy,
    SetwiseStrategy,
    TourRankStrategy,
)
```

Add `CBDRStrategy` to `_is_builtin_sync_strategy(...)`:

```python
def _is_builtin_sync_strategy(strategy: object) -> bool:
    return type(strategy) in {
        AcuRankStrategy,
        CBDRStrategy,
        ConfidenceGainStrategy,
        ListwiseStrategy,
        PairwiseStrategy,
        SetwiseStrategy,
        TourRankStrategy,
    }
```

- [ ] **Step 5: Run CBDR and confidence gain tests**

Run:

```bash
uv run pytest tests/test_cbdr_strategy.py tests/test_confidence_gain_strategy.py -q
```

Expected:

```text
exit code 0 with no test failures
```

- [ ] **Step 6: Commit CBDR strategy**

Run:

```bash
git add src/ranksmith/strategies/_confidence_gain.py src/ranksmith/strategies/_cbdr.py src/ranksmith/strategies/__init__.py src/ranksmith/azure.py tests/test_cbdr_strategy.py
git commit -m "feat: add cbdr strategy"
```

Expected:

```text
commit succeeds
```

---

### Task 3: Artifact-Load E2E Smoke Tests

**Files:**
- Modify: `tests/test_cbdr_strategy.py`

- [ ] **Step 1: Add artifact smoke helpers**

Append these imports near the top of `tests/test_cbdr_strategy.py`:

```python
import sys
from pathlib import Path
from types import ModuleType

from ranksmith.confidence import StructuralConfidenceEstimator
from ranksmith.confidence._scorer import ARTIFACT_SCHEMA_VERSION
```

Add these helper classes and functions:

```python
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
        "ranksmith.confidence._structural.FrozenAutoEncoder.from_pretrained",
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
```

- [ ] **Step 2: Add artifact-load skip path smoke**

Append this test to `tests/test_cbdr_strategy.py`:

```python
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
    assert [result.metadata["context_confidence"] for result in results] == [None, None]
    assert [result.metadata["confidence_gain"] for result in results] == [None, None]
```

- [ ] **Step 3: Add artifact-load rerank path smoke**

Append this test to `tests/test_cbdr_strategy.py`:

```python
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
```

- [ ] **Step 4: Run CBDR tests**

Run:

```bash
uv run pytest tests/test_cbdr_strategy.py -q
```

Expected:

```text
exit code 0 with no test failures
```

- [ ] **Step 5: Run targeted regression tests**

Run:

```bash
uv run pytest tests/test_cbdr_strategy.py tests/test_confidence_gain_strategy.py tests/test_ranksmith.py -q
```

Expected:

```text
exit code 0 with no test failures
```

- [ ] **Step 6: Commit E2E smoke tests**

Run:

```bash
git add tests/test_cbdr_strategy.py
git commit -m "test: add cbdr artifact smoke tests"
```

Expected:

```text
commit succeeds
```

---

### Task 4: Documentation, Spec Checklist, and Final Verification

**Files:**
- Modify: `docs/wiki/02_architecture.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/specs/spec_cbdr_strategy.md`

- [ ] **Step 1: Update architecture wiki**

Modify `docs/wiki/02_architecture.md`:

Add `CBDRStrategy` to the v1 public strategy list after `ConfidenceGainStrategy`.

Add `cbdr` to the v1 algorithm list after `confidence_gain`.

Update the confidence section with:

```markdown
`CBDRStrategy`는 `Conf(Q)`가 `skip_threshold` 이상이면 context reranking을 skip하고 original order를 보존한다.
`Conf(Q)`가 threshold보다 낮으면 `Conf(Q+C)-Conf(Q)` confidence gain으로 문서를 정렬한다.
true pre-retrieval skip, retriever integration, async CBDR은 구현하지 않는다.
```

- [ ] **Step 2: Update README method table and usage**

Modify `README.md` method table by adding this row after `confidence_gain`:

```markdown
| `cbdr` | `CBDRStrategy` | You have trained answerability confidence scorers and want to skip context reranking when `Conf(Q)` is already high, otherwise rerank by confidence gain. | Requires scorer artifacts and an answer generator hook. Skip path uses 1 answer generation call and 1 confidence score; rerank path uses `N+1` answer generations and `N+1` confidence scores. |
```

Add a short usage block near the existing Confidence Gain section:

```markdown
`CBDRStrategy` is a sync reranking-side router. It does not integrate with a
retriever or stop upstream retrieval calls; it only skips context reranking once
documents have already been passed to `rerank(...)`.

```python
from ranksmith import AzureOpenAIReranker
from ranksmith.strategies import CBDRStrategy

strategy = CBDRStrategy(
    base_estimator=query_estimator,
    context_estimator=query_context_estimator,
    answer_generator=answer_generator,
    skip_threshold=0.8,
)

reranker = AzureOpenAIReranker(
    model_client=model_client,
    strategy=strategy,
)

results = reranker.rerank(query, documents)
```

When `Conf(Q) >= skip_threshold`, results preserve original document order and
include `metadata["cbdr_skipped"] == True`. When `Conf(Q) < skip_threshold`, all
documents are scored before `top_k` slicing.
```

Ensure Markdown fences are balanced.

- [ ] **Step 3: Mirror README.ko**

Modify `README.ko.md` with the same section structure and table position.

Use this table row:

```markdown
| `cbdr` | `CBDRStrategy` | answerability confidence scorer를 학습했고 `Conf(Q)`가 충분히 높으면 context reranking을 건너뛰고, 낮으면 confidence gain으로 정렬하고 싶을 때 | scorer artifact와 answer generator hook이 필요함. skip path는 answer generation 1회와 confidence scoring 1회, rerank path는 각각 `N+1`회를 수행함 |
```

Use this usage text:

```markdown
`CBDRStrategy`는 sync reranking-side router입니다. retriever와 통합하거나 upstream
retrieval 호출 자체를 멈추지는 않습니다. 이미 `rerank(...)`에 documents가 전달된
뒤 context reranking을 건너뛸지 결정합니다.

```python
from ranksmith import AzureOpenAIReranker
from ranksmith.strategies import CBDRStrategy

strategy = CBDRStrategy(
    base_estimator=query_estimator,
    context_estimator=query_context_estimator,
    answer_generator=answer_generator,
    skip_threshold=0.8,
)

reranker = AzureOpenAIReranker(
    model_client=model_client,
    strategy=strategy,
)

results = reranker.rerank(query, documents)
```

`Conf(Q) >= skip_threshold`이면 original document order를 보존하고
`metadata["cbdr_skipped"] == True`를 남깁니다. `Conf(Q) < skip_threshold`이면
모든 문서를 scoring한 뒤 `top_k`를 적용합니다.
```

Ensure README.md and README.ko.md keep the same section and table structure.

- [ ] **Step 4: Update spec checklist**

Modify `docs/specs/spec_cbdr_strategy.md`:

Set status:

```markdown
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`
```

Mark all implementation, verification, and docs checklist items complete:

```markdown
- [x] `src/ranksmith/strategies/_confidence_gain.py`: 공통 protocol/helper 정리
- [x] `src/ranksmith/strategies/_cbdr.py`: `CBDRStrategy` 구현
- [x] `src/ranksmith/strategies/__init__.py`: strategy export 추가
- [x] `src/ranksmith/azure.py`: built-in sync strategy 처리 추가
- [x] `tests/test_cbdr_strategy.py`: skip path 정상 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: rerank path 정상 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: 엣지/실패 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: Azure facade smoke 추가
- [x] `tests/test_cbdr_strategy.py`: artifact load 기반 skip path E2E smoke 추가
- [x] `tests/test_cbdr_strategy.py`: artifact load 기반 rerank path E2E smoke 추가
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인
- [x] `docs/wiki/02_architecture.md`: CBDR strategy 위치 추가
- [x] `README.md` / `README.ko.md`: benchmark 없는 usage 문서 추가
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
```

- [ ] **Step 5: Run final verification**

Run:

```bash
./scripts/verify.sh
```

Expected:

```text
pytest passes
ruff passes
format check passes
mypy passes
sdist and wheel build
```

- [ ] **Step 6: Commit docs and final spec**

Run:

```bash
git add docs/wiki/02_architecture.md README.md README.ko.md docs/specs/spec_cbdr_strategy.md
git commit -m "docs: document cbdr strategy"
```

Expected:

```text
commit succeeds
```

---

## Self-Review

- Spec coverage:
  - sync `CBDRStrategy`: Task 2.
  - skip decision: Task 2 skip tests and implementation.
  - low-confidence confidence gain reranking: Task 2 rerank tests and implementation.
  - shared helper refactor: Task 1.
  - `AzureOpenAIReranker` built-in behavior: Task 2 facade import and tests.
  - submodule-only export: Task 2 export test.
  - docs and README: Task 4.
  - no upstream retrieval skip, no `should_retrieve()`, no async: preserved by not adding those APIs.
  - no scorer training or artifact creation: preserved by using existing `from_artifact()` only in smoke tests.
  - no benchmark numbers: Task 4 explicitly avoids benchmark metrics.
- Placeholder scan:
  - No placeholder implementation steps remain.
  - Every new test and code path has concrete snippets.
- Type consistency:
  - `CBDRStrategy.rerank(...)` matches `RerankStrategy`.
  - `AnswerGenerator` and `ConfidenceEstimator` reuse existing strategy contracts.
  - Metadata keys match `docs/specs/spec_cbdr_strategy.md`.
