# Confidence Runtime Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trained confidence artifacts directly usable at runtime through `from_artifact(...)` and memory-bounded `score_batch(...)`, stopping just before CBDR reranking logic.

**Architecture:** Extend `StructuralConfidenceEstimator` in `ranksmith.confidence` without adding a new Strategy or Algorithm. Reuse the existing scorer loader, metadata validation, encoder wrapper, and single-item `score(...)`; batch scoring is chunked, order-preserving, fail-fast, and optionally item-parallel within each chunk.

**Tech Stack:** Python 3.10+, dataclasses, `ThreadPoolExecutor`, existing `ranksmith.confidence` optional dependencies, pytest, ruff, mypy.

---

## File Structure

- Modify: `src/ranksmith/confidence/_structural.py`
  - Add `from_artifact(...)`.
  - Add `score_batch(...)`.
  - Add private helpers for batch validation, chunking, and chunk scoring.
  - Import `Path`, `Sequence`, `ThreadPoolExecutor`, and `load_lightgbm_scorer`.
- Modify: `tests/test_confidence_estimator.py`
  - Add unit tests for `from_artifact(...)`.
  - Add unit tests for `score_batch(...)`, validation, ordering, parallel workers, and failure propagation.
- Modify: `tests/test_confidence_training_artifact.py`
  - Add training artifact -> `from_artifact(...)` -> batch scoring smoke test.
- Modify: `docs/specs/spec_confidence_runtime_readiness.md`
  - Mark implementation checklist items as completed during execution.
  - Mark status `Completed` only after `./scripts/verify.sh` passes.

Do not modify reranking Strategy, Algorithm, provider, CBDR, README benchmark tables, or uploaded reference PDFs in this plan.

Current worktree note: unrelated `confidence_training` refactor changes and untracked reference PDFs may already exist. Do not revert or stage them unless the user explicitly asks.

---

### Task 1: Artifact-Based Estimator Creation

**Files:**
- Modify: `tests/test_confidence_estimator.py`
- Modify: `src/ranksmith/confidence/_structural.py`
- Modify: `docs/specs/spec_confidence_runtime_readiness.md`

- [ ] **Step 1: Write failing tests for `from_artifact(...)`**

Append these tests to `tests/test_confidence_estimator.py`:

```python
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
        return FakeEncoder(max_length=128)

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
```

Also add the missing import near the top of `tests/test_confidence_estimator.py`:

```python
from pathlib import Path
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_from_artifact_uses_scorer_metadata_defaults tests/test_confidence_estimator.py::test_from_artifact_rejects_override_metadata_mismatch -q
```

Expected:

```text
FAILED ... AttributeError: type object 'StructuralConfidenceEstimator' has no attribute 'from_artifact'
```

- [ ] **Step 3: Implement `from_artifact(...)`**

Modify imports in `src/ranksmith/confidence/_structural.py`:

```python
from collections.abc import Sequence
from pathlib import Path
```

Keep `Sequence` if Task 2 is implemented in the same edit; otherwise add only `Path` now.

Change scorer imports:

```python
from ranksmith.confidence._scorer import (
    ARTIFACT_SCHEMA_VERSION,
    load_lightgbm_scorer,
    validate_scorer_metadata,
)
```

Add this classmethod below `from_pretrained(...)` or immediately before it:

```python
    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        encoder_name: str | None = None,
        encoder_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        task_type: TaskType | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        device: str = "cpu",
        max_length: int | None = None,
        allow_truncation: bool = False,
    ) -> StructuralConfidenceEstimator:
        scorer = load_lightgbm_scorer(
            artifact_path,
            metadata_path=metadata_path,
        )
        metadata = scorer.metadata
        return cls.from_pretrained(
            scorer=scorer,
            task_type=task_type or metadata.task_type,
            encoder_name=encoder_name or metadata.encoder_name,
            encoder_revision=(
                metadata.encoder_revision
                if encoder_revision is None
                else encoder_revision
            ),
            tokenizer_name=tokenizer_name or metadata.tokenizer_name,
            tokenizer_revision=(
                metadata.tokenizer_revision
                if tokenizer_revision is None
                else tokenizer_revision
            ),
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            max_length=metadata.max_length if max_length is None else max_length,
            allow_truncation=allow_truncation,
        )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_from_artifact_uses_scorer_metadata_defaults tests/test_confidence_estimator.py::test_from_artifact_rejects_override_metadata_mismatch -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Update spec checklist**

In `docs/specs/spec_confidence_runtime_readiness.md`, mark these items:

```markdown
- [x] `src/ranksmith/confidence/_structural.py`: `from_artifact(...)` 구현
- [x] `src/ranksmith/confidence/_structural.py`: metadata default resolution 구현
- [x] `src/ranksmith/confidence/_structural.py`: override mismatch fast fail 테스트 보강
- [x] `tests/test_confidence_estimator.py`: `from_artifact(...)` 정상/실패 테스트 추가
```

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/ranksmith/confidence/_structural.py tests/test_confidence_estimator.py docs/specs/spec_confidence_runtime_readiness.md
git commit -m "feat: load confidence estimator from artifact"
```

Expected:

```text
[codex/structural-confidence-module abc1234] feat: load confidence estimator from artifact
```

---

### Task 2: Sequential Batch Scoring With Bounded Chunking

**Files:**
- Modify: `tests/test_confidence_estimator.py`
- Modify: `src/ranksmith/confidence/_structural.py`
- Modify: `docs/specs/spec_confidence_runtime_readiness.md`

- [ ] **Step 1: Write failing tests for sequential `score_batch(...)`**

Append these tests to `tests/test_confidence_estimator.py`:

```python
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
    ],
)
def test_score_batch_rejects_invalid_options(kwargs: dict[str, int]) -> None:
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_score_batch_preserves_input_order_with_chunking tests/test_confidence_estimator.py::test_score_batch_rejects_invalid_options tests/test_confidence_estimator.py::test_score_batch_rejects_empty_items tests/test_confidence_estimator.py::test_score_batch_rejects_too_many_items -q
```

Expected:

```text
FAILED ... AttributeError: 'StructuralConfidenceEstimator' object has no attribute 'score_batch'
```

- [ ] **Step 3: Implement sequential batch helpers**

Modify imports in `src/ranksmith/confidence/_structural.py`:

```python
from collections.abc import Sequence
```

Also import `ConfidenceInputError`:

```python
from ranksmith.confidence._errors import (
    ConfidenceArtifactError,
    ConfidenceInputError,
)
```

Add `score_batch(...)` to `StructuralConfidenceEstimator`:

```python
    def score_batch(
        self,
        items: Sequence[StructuralConfidenceInput],
        *,
        batch_size: int = 8,
        max_workers: int = 1,
        max_batch_items: int | None = None,
    ) -> list[StructuralConfidenceResult]:
        _validate_batch_options(
            items,
            batch_size=batch_size,
            max_workers=max_workers,
            max_batch_items=max_batch_items,
        )
        results: list[StructuralConfidenceResult] = []
        for chunk in _chunked(items, batch_size):
            results.extend(self._score_chunk(chunk, max_workers=max_workers))
        return results

    def _score_chunk(
        self,
        items: Sequence[StructuralConfidenceInput],
        *,
        max_workers: int,
    ) -> list[StructuralConfidenceResult]:
        if max_workers == 1:
            return [self.score(item) for item in items]
        return _score_chunk_parallel(self, items, max_workers=max_workers)
```

Add private helpers below `_result_metadata(...)` or before it:

```python
def _validate_batch_options(
    items: Sequence[StructuralConfidenceInput],
    *,
    batch_size: int,
    max_workers: int,
    max_batch_items: int | None,
) -> None:
    if not items:
        raise ConfidenceInputError("items must not be empty.")
    if batch_size < 1:
        raise ConfidenceInputError("batch_size must be >= 1.")
    if max_workers < 1:
        raise ConfidenceInputError("max_workers must be >= 1.")
    if max_batch_items is not None and max_batch_items < 1:
        raise ConfidenceInputError("max_batch_items must be >= 1.")
    if max_batch_items is not None and len(items) > max_batch_items:
        raise ConfidenceInputError("items exceeds max_batch_items.")


def _chunked(
    items: Sequence[StructuralConfidenceInput],
    size: int,
) -> list[Sequence[StructuralConfidenceInput]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
```

Add a temporary parallel placeholder so Task 2 can type-check if `max_workers > 1` is not called yet:

```python
def _score_chunk_parallel(
    estimator: StructuralConfidenceEstimator,
    items: Sequence[StructuralConfidenceInput],
    *,
    max_workers: int,
) -> list[StructuralConfidenceResult]:
    del max_workers
    return [estimator.score(item) for item in items]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_score_batch_preserves_input_order_with_chunking tests/test_confidence_estimator.py::test_score_batch_rejects_invalid_options tests/test_confidence_estimator.py::test_score_batch_rejects_empty_items tests/test_confidence_estimator.py::test_score_batch_rejects_too_many_items -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Update spec checklist**

In `docs/specs/spec_confidence_runtime_readiness.md`, mark:

```markdown
- [x] `src/ranksmith/confidence/_structural.py`: `score_batch(...)` 구현
- [x] `src/ranksmith/confidence/_structural.py`: batch option validation helper 구현
- [x] `src/ranksmith/confidence/_structural.py`: chunk helper 구현
- [x] `tests/test_confidence_estimator.py`: `score_batch(...)` 정상/실패 테스트 추가
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/ranksmith/confidence/_structural.py tests/test_confidence_estimator.py docs/specs/spec_confidence_runtime_readiness.md
git commit -m "feat: add confidence batch scoring"
```

Expected:

```text
[codex/structural-confidence-module abc1234] feat: add confidence batch scoring
```

---

### Task 3: Bounded Parallel Batch Execution

**Files:**
- Modify: `tests/test_confidence_estimator.py`
- Modify: `src/ranksmith/confidence/_structural.py`
- Modify: `docs/specs/spec_confidence_runtime_readiness.md`

- [ ] **Step 1: Write failing tests for `max_workers > 1`**

Append these tests to `tests/test_confidence_estimator.py`:

```python
class IndexedScorer(FakeScorer):
    def __init__(self, *, task_type: TaskType = "answer_confidence") -> None:
        super().__init__(score=0.0, task_type=task_type)

    def predict_confidence(self, features: Sequence[float]) -> float:
        assert len(features) == 70
        return 0.5


def test_score_batch_parallel_preserves_input_order() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=IndexedScorer(task_type="judgment_confidence"),
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
    assert [result.score for result in results] == [0.5] * 6
    assert [result.task_type for result in results] == ["judgment_confidence"] * 6


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
```

- [ ] **Step 2: Run tests and verify current placeholder is insufficient**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_score_batch_parallel_preserves_input_order tests/test_confidence_estimator.py::test_score_batch_parallel_propagates_worker_error -q
```

Expected:

```text
2 passed
```

If the placeholder passes these tests, add one more test to prove bounded parallel branch is used:

```python
def test_score_batch_parallel_uses_executor_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            called.append(max_workers)

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def map(self, fn: object, values: object) -> list[object]:
            return [fn(value) for value in values]  # type: ignore[misc]

    monkeypatch.setattr(
        "ranksmith.confidence._structural.ThreadPoolExecutor",
        FakeExecutor,
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
```

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_score_batch_parallel_uses_executor_branch -q
```

Expected:

```text
FAILED ... AttributeError or AssertionError
```

- [ ] **Step 3: Implement bounded worker pool**

Modify imports in `src/ranksmith/confidence/_structural.py`:

```python
from concurrent.futures import ThreadPoolExecutor
```

Replace `_score_chunk_parallel(...)` with:

```python
def _score_chunk_parallel(
    estimator: StructuralConfidenceEstimator,
    items: Sequence[StructuralConfidenceInput],
    *,
    max_workers: int,
) -> list[StructuralConfidenceResult]:
    indexed_items = list(enumerate(items))

    def score_indexed(
        value: tuple[int, StructuralConfidenceInput],
    ) -> tuple[int, StructuralConfidenceResult]:
        index, item = value
        return index, estimator.score(item)

    results: list[StructuralConfidenceResult | None] = [None] * len(indexed_items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, result in executor.map(score_indexed, indexed_items):
            results[index] = result

    return [_require_result(result) for result in results]
```

Add helper:

```python
def _require_result(
    result: StructuralConfidenceResult | None,
) -> StructuralConfidenceResult:
    if result is None:
        raise ConfidenceArtifactError("parallel confidence result is missing.")
    return result
```

- [ ] **Step 4: Run parallel tests**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py::test_score_batch_parallel_preserves_input_order tests/test_confidence_estimator.py::test_score_batch_parallel_propagates_worker_error tests/test_confidence_estimator.py::test_score_batch_parallel_uses_executor_branch -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Update spec checklist**

In `docs/specs/spec_confidence_runtime_readiness.md`, mark:

```markdown
- [x] `src/ranksmith/confidence/_structural.py`: `max_workers` 정책 구현
```

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/ranksmith/confidence/_structural.py tests/test_confidence_estimator.py docs/specs/spec_confidence_runtime_readiness.md
git commit -m "feat: parallelize confidence batch scoring"
```

Expected:

```text
[codex/structural-confidence-module abc1234] feat: parallelize confidence batch scoring
```

---

### Task 4: Training Artifact Runtime Smoke And Final Verification

**Files:**
- Modify: `tests/test_confidence_training_artifact.py`
- Modify: `tests/test_confidence_estimator.py`
- Modify: `docs/specs/spec_confidence_runtime_readiness.md`

- [ ] **Step 1: Write training artifact -> runtime batch smoke test**

Modify imports in `tests/test_confidence_training_artifact.py`:

```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    StructuralConfidenceEstimator,
    load_lightgbm_scorer,
)
```

Append helper:

```python
def _write_judgment_dataset(path: Path, count: int = 40) -> None:
    rows = []
    for i in range(count):
        label = i % 2
        rows.append(
            {
                "id": f"j{i}",
                "query": f"query {i}",
                "document": f"{'positive' if label else 'negative'} document {i}",
                "judgment": "direct evidence" if label else "no evidence",
                "label": label,
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
```

Append test:

```python
def test_train_confidence_scorer_artifact_scores_judgment_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "judgment_dataset.jsonl"
    export_path = tmp_path / "judgment_artifact.joblib"
    output_dir = tmp_path / "judgment_run"
    _write_judgment_dataset(dataset_path)

    monkeypatch.setattr(
        "ranksmith.confidence_training._pipeline.FrozenAutoEncoder.from_pretrained",
        _fake_from_pretrained,
    )
    monkeypatch.setattr(
        "ranksmith.confidence._structural.FrozenAutoEncoder.from_pretrained",
        _fake_from_pretrained,
    )

    train_confidence_scorer(
        ConfidenceTrainingConfig(
            task_type="judgment_confidence",
            dataset_path=dataset_path,
            output_dir=output_dir,
            export_path=export_path,
            max_length=34,
            seed=7,
        )
    )

    estimator = StructuralConfidenceEstimator.from_artifact(export_path)
    results = estimator.score_batch(
        [
            JudgmentConfidenceInput(
                query="query",
                document="positive document",
                judgment="direct evidence",
            ),
            JudgmentConfidenceInput(
                query="query",
                document="negative document",
                judgment="no evidence",
            ),
        ],
        batch_size=1,
        max_workers=1,
        max_batch_items=2,
    )

    assert len(results) == 2
    assert [result.task_type for result in results] == [
        "judgment_confidence",
        "judgment_confidence",
    ]
    assert all(0.0 <= result.score <= 1.0 for result in results)
```

- [ ] **Step 2: Run smoke test**

Run:

```bash
uv run pytest tests/test_confidence_training_artifact.py::test_train_confidence_scorer_artifact_scores_judgment_batch -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Add metadata non-exposure assertion for batch results**

Append to `tests/test_confidence_estimator.py`:

```python
def test_score_batch_metadata_excludes_sensitive_and_heavy_values() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    results = estimator.score_batch(
        [AnswerConfidenceInput(context="context", answer="answer")]
    )

    forbidden = {
        "hf_token",
        "token",
        "cache_dir",
        "model",
        "tokenizer",
        "features",
        "feature_vector",
        "hidden_states",
        "local_path",
    }
    assert forbidden.isdisjoint(results[0].metadata)
```

- [ ] **Step 4: Run focused confidence tests**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py tests/test_confidence_training_artifact.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Update spec checklist and status**

In `docs/specs/spec_confidence_runtime_readiness.md`, mark:

```markdown
- [x] `src/ranksmith/confidence/_structural.py`: memory-safe result metadata 유지
- [x] `tests/test_confidence_training_artifact.py`: training artifact -> runtime batch smoke test 추가
```

Do not mark verification commands or `Completed` until Step 6 passes.

- [ ] **Step 6: Run full verification**

Run:

```bash
uv run pytest tests/test_confidence_*.py tests/test_confidence_training_*.py -q
uv run ruff check src/ranksmith/confidence tests/test_confidence_*.py
uv run mypy src/ranksmith/confidence tests/test_confidence_*.py
./scripts/verify.sh
```

Expected:

```text
pytest: all selected tests passed
ruff: All checks passed!
mypy: Success: no issues found
verify.sh: tests, lint, format, type check, and build pass
```

- [ ] **Step 7: Mark spec completed**

In `docs/specs/spec_confidence_runtime_readiness.md`, update:

```markdown
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`
```

Mark verification checklist items:

```markdown
- [x] `uv run pytest tests/test_confidence_*.py tests/test_confidence_training_*.py -q`
- [x] `uv run ruff check src/ranksmith/confidence tests/test_confidence_*.py`
- [x] `uv run mypy src/ranksmith/confidence tests/test_confidence_*.py`
- [x] `./scripts/verify.sh`
- [x] 본 문서 상태를 `Completed`로 변경
```

- [ ] **Step 8: Commit Task 4**

Run:

```bash
git add src/ranksmith/confidence/_structural.py tests/test_confidence_estimator.py tests/test_confidence_training_artifact.py docs/specs/spec_confidence_runtime_readiness.md
git commit -m "test: verify confidence runtime readiness"
```

Expected:

```text
[codex/structural-confidence-module abc1234] test: verify confidence runtime readiness
```

---

## Self-Review

Spec coverage:
- `from_artifact(...)`: Task 1.
- Metadata default resolution and mismatch fast fail: Task 1.
- `score_batch(...)`: Task 2.
- `batch_size`, `max_workers`, `max_batch_items`: Tasks 2 and 3.
- Chunking and order preservation: Task 2.
- Bounded item-level parallelism: Task 3.
- Fail-fast batch policy: Tasks 2 and 3.
- Memory-safe metadata: Task 4.
- Training artifact -> runtime smoke: Task 4.
- CBDR-ready signal contract: implemented by preserving input order and returning `StructuralConfidenceResult`; candidate identity remains outside this module.

Scope kept out:
- Closed model judgment generation.
- CBDR reranking algorithm.
- Score fusion.
- Async API.
- Provider parallelism.
- Feature cache.
- Benchmark claims.

Placeholder scan:
- No unresolved placeholder patterns remain in this plan.
- Every task has exact files, test code, implementation code, commands, and expected results.

Type consistency:
- `from_artifact(...)` returns `StructuralConfidenceEstimator`.
- `score_batch(...)` returns `list[StructuralConfidenceResult]`.
- Batch input type is `Sequence[StructuralConfidenceInput]`.
- Errors use existing `ConfidenceInputError` and `ConfidenceArtifactError`.
