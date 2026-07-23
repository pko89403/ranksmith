# Confidence Gain Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Inc(Q, C) = Conf(Q+C) - Conf(Q)` support by extending confidence tasks and adding a sync `ConfidenceGainStrategy`.

**Architecture:** Reuse the existing `StructuralConfidenceEstimator.from_artifact()`, `score()`, and `score_batch()` runtime. Add two answerability task types, extend generation/training validation, then add a small Strategy that consumes two estimators plus an answer generator hook and sorts by confidence gain.

**Tech Stack:** Python 3.10+, dataclasses, existing `ranksmith.confidence`, existing `ranksmith.confidence_generation`, existing `ranksmith.confidence_training`, existing Strategy protocol, pytest, ruff, mypy.

---

## File Map

- Modify `src/ranksmith/confidence/_types.py`
  - Add `QueryAnswerabilityConfidenceInput` and `QueryContextAnswerabilityConfidenceInput`.
  - Extend `TaskType` and `StructuralConfidenceInput`.
- Modify `src/ranksmith/confidence/_templates.py`
  - Add templates for query-only and query+context answerability confidence.
- Modify `src/ranksmith/confidence/_scorer.py`
  - Accept the new task types in scorer metadata.
- Modify `src/ranksmith/confidence/__init__.py`
  - Export the new input types from the confidence submodule.
- Modify `src/ranksmith/confidence_training/_types.py`
  - Extend training config and canonical sample fields.
- Modify `src/ranksmith/confidence_training/_dataset.py`
  - Validate and parse the new canonical JSONL schemas.
- Modify `src/ranksmith/confidence_training/_features.py`
  - Convert new canonical samples into new runtime input dataclasses.
- Modify `src/ranksmith/confidence_training/_artifact.py`
  - Ensure metadata export accepts new task types through existing `ScorerMetadata`.
- Modify `src/ranksmith/confidence_generation/_types.py`
  - Add configs and raw sample dataclasses for query-only and query+context answerability generation.
- Modify `src/ranksmith/confidence_generation/_io.py`
  - Load the new raw JSONL schemas and validate resume output rows.
- Modify `src/ranksmith/confidence_generation/_prompts.py`
  - Add prompts for base answer and context-conditioned answer.
- Modify `src/ranksmith/confidence_generation/_pipeline.py`
  - Add `generate_query_answerability_confidence_dataset()` and `generate_query_context_answerability_confidence_dataset()`.
- Modify `src/ranksmith/confidence_generation/__init__.py`
  - Export the new generation configs/functions.
- Create `src/ranksmith/strategies/_confidence_gain.py`
  - Implement `AnswerGenerator`, `ConfidenceGainResult`, and `ConfidenceGainStrategy`.
- Modify `src/ranksmith/strategies/__init__.py`
  - Export `ConfidenceGainStrategy`.
- Modify docs:
  - `docs/wiki/02_architecture.md`
  - `docs/wiki/04_references_index.md`
  - `README.md`
  - `README.ko.md`
  - `docs/specs/spec_confidence_gain_reranking.md`
- Tests:
  - `tests/test_confidence_answerability_tasks.py`
  - `tests/test_confidence_training_answerability.py`
  - `tests/test_confidence_generation_answerability.py`
  - `tests/test_confidence_gain_strategy.py`

---

### Task 1: Confidence Runtime Answerability Tasks

**Files:**
- Modify: `src/ranksmith/confidence/_types.py`
- Modify: `src/ranksmith/confidence/_templates.py`
- Modify: `src/ranksmith/confidence/_scorer.py`
- Modify: `src/ranksmith/confidence/__init__.py`
- Test: `tests/test_confidence_answerability_tasks.py`

- [ ] **Step 1: Write failing runtime task tests**

Create `tests/test_confidence_answerability_tasks.py`:

```python
from __future__ import annotations

import importlib
from collections.abc import Sequence

import pytest

from ranksmith.confidence import (
    ConfidenceInputError,
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceEstimator,
)


class FakeEncoder:
    encoder_name = "bert-base-uncased"
    encoder_revision = None
    tokenizer_name = "bert-base-uncased"
    tokenizer_revision = None
    max_length = 256

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        self.texts.append(text)
        return [[0.1, 0.2], [0.3, 0.4]], [1, 1]


class FakeScorer:
    def __init__(self, task_type: str) -> None:
        self.metadata = ScorerMetadata(
            artifact_schema_version="structural-artifact-v1",
            scorer_type="fake",
            task_type=task_type,  # type: ignore[arg-type]
            encoder_name="bert-base-uncased",
            encoder_revision=None,
            tokenizer_name="bert-base-uncased",
            tokenizer_revision=None,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
            feature_dtype="float64",
            max_length=256,
            granularity="token",
            local_window_size=5,
            local_stride=2,
            score_output="probability",
            positive_class_index=1,
        )

    def predict_confidence(self, features: Sequence[float]) -> float:
        return 0.75


def test_confidence_submodule_exports_answerability_inputs() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert hasattr(confidence, "QueryAnswerabilityConfidenceInput")
    assert hasattr(confidence, "QueryContextAnswerabilityConfidenceInput")


def test_query_answerability_template_is_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ranksmith.confidence._structural.extract_structural_features",
        lambda hidden, mask, *, max_length: [0.0] * 70,
    )
    encoder = FakeEncoder()
    estimator = StructuralConfidenceEstimator(
        encoder=encoder,
        scorer=FakeScorer("query_answerability_confidence"),
        task_type="query_answerability_confidence",
    )

    result = estimator.score(
        QueryAnswerabilityConfidenceInput(query="Who?", answer="Nancy Travis")
    )

    assert result.score == 0.75
    assert encoder.texts == ["Query:\nWho?\n\nAnswer:\nNancy Travis"]


def test_query_context_answerability_template_is_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ranksmith.confidence._structural.extract_structural_features",
        lambda hidden, mask, *, max_length: [0.0] * 70,
    )
    encoder = FakeEncoder()
    estimator = StructuralConfidenceEstimator(
        encoder=encoder,
        scorer=FakeScorer("query_context_answerability_confidence"),
        task_type="query_context_answerability_confidence",
    )

    result = estimator.score(
        QueryContextAnswerabilityConfidenceInput(
            query="Who?",
            context="Karen was played by Nancy Travis.",
            answer="Nancy Travis",
        )
    )

    assert result.score == 0.75
    assert encoder.texts == [
        "Query:\nWho?\n\nContext:\nKaren was played by Nancy Travis.\n\nAnswer:\nNancy Travis"
    ]


def test_answerability_task_rejects_wrong_input_type() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer("query_answerability_confidence"),
        task_type="query_answerability_confidence",
    )

    with pytest.raises(ConfidenceInputError):
        estimator.score(
            QueryContextAnswerabilityConfidenceInput(
                query="Who?",
                context="Context",
                answer="Answer",
            )
        )
```

- [ ] **Step 2: Run runtime task tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_answerability_tasks.py -q
```

Expected: FAIL because `QueryAnswerabilityConfidenceInput` is not exported.

- [ ] **Step 3: Implement runtime task types and templates**

Modify `src/ranksmith/confidence/_types.py`:

```python
TaskType: TypeAlias = Literal[
    "answer_confidence",
    "judgment_confidence",
    "query_answerability_confidence",
    "query_context_answerability_confidence",
]


@dataclass(frozen=True)
class QueryAnswerabilityConfidenceInput:
    query: str
    answer: str


@dataclass(frozen=True)
class QueryContextAnswerabilityConfidenceInput:
    query: str
    context: str
    answer: str


StructuralConfidenceInput: TypeAlias = (
    AnswerConfidenceInput
    | JudgmentConfidenceInput
    | QueryAnswerabilityConfidenceInput
    | QueryContextAnswerabilityConfidenceInput
)
```

Modify `src/ranksmith/confidence/_templates.py`:

```python
from ranksmith.confidence._types import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    StructuralConfidenceInput,
    TaskType,
)
```

Add branches in `format_confidence_input(...)`:

```python
    if task_type == "query_answerability_confidence":
        if not isinstance(item, QueryAnswerabilityConfidenceInput):
            raise ConfidenceInputError(
                "query_answerability_confidence requires "
                "QueryAnswerabilityConfidenceInput"
            )
        query = _require_non_empty(item.query, field_name="query")
        answer = _require_non_empty(item.answer, field_name="answer")
        return f"Query:\n{query}\n\nAnswer:\n{answer}"

    if task_type == "query_context_answerability_confidence":
        if not isinstance(item, QueryContextAnswerabilityConfidenceInput):
            raise ConfidenceInputError(
                "query_context_answerability_confidence requires "
                "QueryContextAnswerabilityConfidenceInput"
            )
        query = _require_non_empty(item.query, field_name="query")
        context = _require_non_empty(item.context, field_name="context")
        answer = _require_non_empty(item.answer, field_name="answer")
        return f"Query:\n{query}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
```

Modify `_task_type(...)` in `src/ranksmith/confidence/_scorer.py` to accept the two new literals.

Modify `src/ranksmith/confidence/__init__.py` to import/export both new dataclasses.

- [ ] **Step 4: Run runtime task tests and targeted existing tests**

Run:

```bash
uv run pytest tests/test_confidence_answerability_tasks.py tests/test_confidence_estimator.py tests/test_confidence_scorer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/ranksmith/confidence tests/test_confidence_answerability_tasks.py
git commit -m "feat: add answerability confidence tasks"
```

---

### Task 2: Training Dataset And Feature Support

**Files:**
- Modify: `src/ranksmith/confidence_training/_types.py`
- Modify: `src/ranksmith/confidence_training/_dataset.py`
- Modify: `src/ranksmith/confidence_training/_features.py`
- Test: `tests/test_confidence_training_answerability.py`

- [ ] **Step 1: Write failing training tests**

Create `tests/test_confidence_training_answerability.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksmith.confidence import (
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
)
from ranksmith.confidence_training import ConfidenceTrainingConfig
from ranksmith.confidence_training._dataset import load_canonical_dataset
from ranksmith.confidence_training._errors import ConfidenceDatasetError
from ranksmith.confidence_training._features import sample_to_confidence_input


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_training_config_accepts_query_answerability_task(tmp_path: Path) -> None:
    config = ConfidenceTrainingConfig(
        task_type="query_answerability_confidence",
        dataset_path=tmp_path / "data.jsonl",
        output_dir=tmp_path / "run",
        export_path=tmp_path / "scorer.joblib",
    )

    assert config.task_type == "query_answerability_confidence"


def test_loads_query_answerability_canonical_dataset(tmp_path: Path) -> None:
    path = tmp_path / "query.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "q1::base",
                "task_type": "query_answerability_confidence",
                "query": "Who?",
                "answer": "Nancy Travis",
                "gold_answer": "Nancy Travis",
                "label": 1,
                "metadata": {"input_metadata": {}},
            }
        ],
    )

    samples = load_canonical_dataset(
        path,
        task_type="query_answerability_confidence",
    )

    assert samples[0].query == "Who?"
    assert samples[0].answer == "Nancy Travis"
    assert isinstance(sample_to_confidence_input(samples[0]), QueryAnswerabilityConfidenceInput)


def test_loads_query_context_answerability_canonical_dataset(tmp_path: Path) -> None:
    path = tmp_path / "query_context.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "q1::doc1",
                "task_type": "query_context_answerability_confidence",
                "query": "Who?",
                "context": "Karen was played by Nancy Travis.",
                "answer": "Nancy Travis",
                "gold_answer": "Nancy Travis",
                "label": 1,
                "metadata": {"input_metadata": {}},
            }
        ],
    )

    samples = load_canonical_dataset(
        path,
        task_type="query_context_answerability_confidence",
    )

    assert samples[0].context == "Karen was played by Nancy Travis."
    assert isinstance(
        sample_to_confidence_input(samples[0]),
        QueryContextAnswerabilityConfidenceInput,
    )


def test_query_context_dataset_rejects_missing_context(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "q1::doc1",
                "task_type": "query_context_answerability_confidence",
                "query": "Who?",
                "answer": "Nancy Travis",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ConfidenceDatasetError, match="missing required field: context"):
        load_canonical_dataset(
            path,
            task_type="query_context_answerability_confidence",
        )
```

- [ ] **Step 2: Run training tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_training_answerability.py -q
```

Expected: FAIL because `ConfidenceTrainingConfig` rejects the new task type or dataset schemas are missing.

- [ ] **Step 3: Implement training schema support**

Modify `ConfidenceTrainingConfig.__post_init__` in `src/ranksmith/confidence_training/_types.py`:

```python
if self.task_type not in {
    "answer_confidence",
    "judgment_confidence",
    "query_answerability_confidence",
    "query_context_answerability_confidence",
}:
    raise ConfidenceTrainingConfigError("unsupported task_type")
```

Extend `CanonicalConfidenceSample` only if `query`, `context`, and `answer` fields already cannot represent the new tasks. Prefer reusing existing fields.

Modify `src/ranksmith/confidence_training/_dataset.py`:

```python
_QUERY_ANSWERABILITY_REQUIRED = ("id", "query", "answer", "label")
_QUERY_CONTEXT_ANSWERABILITY_REQUIRED = ("id", "query", "context", "answer", "label")
_QUERY_ANSWERABILITY_ALLOWED = {
    *_QUERY_ANSWERABILITY_REQUIRED,
    "task_type",
    "gold_answer",
    "source",
    "group_id",
    "metadata",
}
_QUERY_CONTEXT_ANSWERABILITY_ALLOWED = {
    *_QUERY_CONTEXT_ANSWERABILITY_REQUIRED,
    "task_type",
    "gold_answer",
    "source",
    "group_id",
    "metadata",
}
```

Add `_TASK_SCHEMAS` entries that construct `CanonicalConfidenceSample` with the matching `task_type`, `query`, `context`, `answer`, `gold_answer`, `source`, `group_id`, and `metadata`.

Modify `src/ranksmith/confidence_training/_features.py` to map:

```python
if sample.task_type == "query_answerability_confidence":
    return QueryAnswerabilityConfidenceInput(
        query=_required(sample.query, "query"),
        answer=_required(sample.answer, "answer"),
    )
if sample.task_type == "query_context_answerability_confidence":
    return QueryContextAnswerabilityConfidenceInput(
        query=_required(sample.query, "query"),
        context=_required(sample.context, "context"),
        answer=_required(sample.answer, "answer"),
    )
```

- [ ] **Step 4: Run training tests and existing confidence training tests**

Run:

```bash
uv run pytest tests/test_confidence_training_answerability.py tests/test_confidence_training_*.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/ranksmith/confidence_training tests/test_confidence_training_answerability.py
git commit -m "feat: support answerability confidence training"
```

---

### Task 3: Generation Pipeline Answerability Datasets

**Files:**
- Modify: `src/ranksmith/confidence_generation/_types.py`
- Modify: `src/ranksmith/confidence_generation/_io.py`
- Modify: `src/ranksmith/confidence_generation/_prompts.py`
- Modify: `src/ranksmith/confidence_generation/_pipeline.py`
- Modify: `src/ranksmith/confidence_generation/__init__.py`
- Test: `tests/test_confidence_generation_answerability.py`

- [ ] **Step 1: Write failing generation tests**

Create `tests/test_confidence_generation_answerability.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from ranksmith.confidence_generation import (
    QueryAnswerabilityGenerationConfig,
    QueryContextAnswerabilityGenerationConfig,
    generate_query_answerability_confidence_dataset,
    generate_query_context_answerability_confidence_dataset,
)
from ranksmith.model import ModelResponse


class RecordingProvider:
    def __init__(self, answer: str = "Nancy Travis") -> None:
        self.answer = answer
        self.requests = []

    def complete(self, request):  # noqa: ANN001
        self.requests.append(request)
        return ModelResponse(content=json.dumps({"answer": self.answer}))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generates_query_answerability_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "q1::base",
                "query": "Who played Karen?",
                "gold_answer": "Nancy Travis",
                "metadata": {"split": "tiny"},
            }
        ],
    )
    provider = RecordingProvider()

    result = generate_query_answerability_confidence_dataset(
        QueryAnswerabilityGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
        )
    )

    rows = _read_jsonl(output_path)
    assert result.generated_count == 1
    assert rows[0]["task_type"] == "query_answerability_confidence"
    assert rows[0]["query"] == "Who played Karen?"
    assert rows[0]["answer"] == "Nancy Travis"
    assert rows[0]["label"] == 1


def test_generates_query_context_answerability_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "q1::doc1",
                "query": "Who played Karen?",
                "context": "Karen was played by Nancy Travis.",
                "gold_answer": "Nancy Travis",
                "group_id": "q1",
            }
        ],
    )
    provider = RecordingProvider()

    result = generate_query_context_answerability_confidence_dataset(
        QueryContextAnswerabilityGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
        )
    )

    rows = _read_jsonl(output_path)
    assert result.generated_count == 1
    assert rows[0]["task_type"] == "query_context_answerability_confidence"
    assert rows[0]["context"] == "Karen was played by Nancy Travis."
    assert rows[0]["answer"] == "Nancy Travis"
    assert rows[0]["label"] == 1
```

- [ ] **Step 2: Run generation tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_answerability.py -q
```

Expected: FAIL because new generation configs/functions are not exported.

- [ ] **Step 3: Implement generation types, IO, prompts, and pipelines**

Add config/sample dataclasses to `src/ranksmith/confidence_generation/_types.py`:

```python
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
```

Reuse existing `_validate_common_config`, `no_answer_value`, and positive integer validation.

Add raw sample dataclasses:

```python
@dataclass(frozen=True)
class QueryAnswerabilityGenerationSample:
    id: str
    query: str
    gold_answer: str | list[str]
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryContextAnswerabilityGenerationSample:
    id: str
    query: str
    context: str
    gold_answer: str | list[str]
    source: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

In `_io.py`, add loaders mirroring existing answer sample loading:
- query-only requires `id`, `query`, `gold_answer`.
- query+context requires `id`, `query`, `context`, `gold_answer`.
- unexpected fields fail.
- whitespace-only text fails.
- context length uses raw string length and `max_context_chars`.

In `_prompts.py`, add:
- `QUERY_ANSWERABILITY_SYSTEM_PROMPT`
- `QUERY_CONTEXT_ANSWERABILITY_SYSTEM_PROMPT`
- `build_query_answerability_prompt(sample, no_answer_value=...)`
- `build_query_context_answerability_prompt(sample, no_answer_value=...)`

Both prompts require JSON response `{"answer": "..."}`.

In `_pipeline.py`, add:
- `generate_query_answerability_confidence_dataset(config)`
- `generate_query_context_answerability_confidence_dataset(config)`

Build canonical rows with task types:
- `query_answerability_confidence`
- `query_context_answerability_confidence`

Use existing `parse_answer_output()` and `normalized_exact_match()`.

Update `__init__.py` exports.

- [ ] **Step 4: Run generation tests and existing generation tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_answerability.py tests/test_confidence_generation_*.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/ranksmith/confidence_generation tests/test_confidence_generation_answerability.py
git commit -m "feat: generate answerability confidence datasets"
```

---

### Task 4: Confidence Gain Strategy

**Files:**
- Create: `src/ranksmith/strategies/_confidence_gain.py`
- Modify: `src/ranksmith/strategies/__init__.py`
- Test: `tests/test_confidence_gain_strategy.py`

- [ ] **Step 1: Write failing strategy tests**

Create `tests/test_confidence_gain_strategy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ranksmith.errors import DocumentTooLongError, RerankInputError, RerankProviderError
from ranksmith.strategies import ConfidenceGainStrategy
from ranksmith.types import Document


@dataclass
class FakeConfidenceResult:
    score: float
    task_type: str
    feature_schema_version: str = "structural-v1"
    metadata: dict[str, object] | None = None


class BaseEstimator:
    task_type = "query_answerability_confidence"

    def __init__(self, score: float) -> None:
        self.score_value = score
        self.items = []

    def score(self, item):  # noqa: ANN001
        self.items.append(item)
        return FakeConfidenceResult(score=self.score_value, task_type=self.task_type)


class ContextEstimator:
    task_type = "query_context_answerability_confidence"

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.items = []

    def score(self, item):  # noqa: ANN001
        self.items.append(item)
        return FakeConfidenceResult(
            score=self.scores.pop(0),
            task_type=self.task_type,
        )


class AnswerGenerator:
    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.context_calls: list[tuple[str, str]] = []

    def answer_query(self, query: str) -> str:
        self.query_calls.append(query)
        return "base answer"

    def answer_with_context(self, query: str, context: str) -> str:
        self.context_calls.append((query, context))
        return f"answer from {context}"


def test_confidence_gain_strategy_sorts_by_gain_and_preserves_ties() -> None:
    generator = AnswerGenerator()
    strategy = ConfidenceGainStrategy(
        base_estimator=BaseEstimator(0.4),
        context_estimator=ContextEstimator([0.9, 0.5, 0.5]),
        answer_generator=generator,
    )
    documents = [Document("doc-a"), Document("doc-b"), Document("doc-c")]

    results = strategy.rerank(query="Who?", documents=documents, model_client=object())

    assert [result.original_index for result in results] == [0, 1, 2]
    assert [result.metadata["confidence_gain"] for result in results] == [
        pytest.approx(0.5),
        pytest.approx(0.1),
        pytest.approx(0.1),
    ]
    assert len(generator.query_calls) == 1
    assert len(generator.context_calls) == 3


def test_confidence_gain_strategy_applies_top_k() -> None:
    strategy = ConfidenceGainStrategy(
        base_estimator=BaseEstimator(0.2),
        context_estimator=ContextEstimator([0.4, 0.9]),
        answer_generator=AnswerGenerator(),
    )

    results = strategy.rerank(
        query="Who?",
        documents=[Document("a"), Document("b")],
        model_client=object(),
        top_k=1,
    )

    assert [result.original_index for result in results] == [1]


def test_confidence_gain_strategy_rejects_wrong_estimator_task() -> None:
    bad_base = ContextEstimator([0.1])

    with pytest.raises(RerankInputError, match="base_estimator"):
        ConfidenceGainStrategy(
            base_estimator=bad_base,
            context_estimator=ContextEstimator([0.2]),
            answer_generator=AnswerGenerator(),
        )


def test_confidence_gain_strategy_wraps_empty_answer() -> None:
    class EmptyAnswerGenerator(AnswerGenerator):
        def answer_query(self, query: str) -> str:
            return " "

    strategy = ConfidenceGainStrategy(
        base_estimator=BaseEstimator(0.2),
        context_estimator=ContextEstimator([0.4]),
        answer_generator=EmptyAnswerGenerator(),
    )

    with pytest.raises(RerankProviderError, match="answer_query returned empty"):
        strategy.rerank(query="Who?", documents=[Document("a")], model_client=object())


def test_confidence_gain_strategy_rejects_long_document() -> None:
    strategy = ConfidenceGainStrategy(
        base_estimator=BaseEstimator(0.2),
        context_estimator=ContextEstimator([0.4]),
        answer_generator=AnswerGenerator(),
        max_document_chars=3,
    )

    with pytest.raises(DocumentTooLongError):
        strategy.rerank(query="Who?", documents=[Document("abcd")], model_client=object())
```

- [ ] **Step 2: Run strategy tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_gain_strategy.py -q
```

Expected: FAIL because `ConfidenceGainStrategy` is not exported.

- [ ] **Step 3: Implement strategy**

Create `src/ranksmith/strategies/_confidence_gain.py`:

```python
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ranksmith.confidence import (
    QueryAnswerabilityConfidenceInput,
    QueryContextAnswerabilityConfidenceInput,
    StructuralConfidenceResult,
)
from ranksmith.errors import (
    DocumentTooLongError,
    RerankInputError,
    RerankProviderError,
    RerankStrategyError,
)
from ranksmith.types import Document, RerankResult

from ._common import validate_top_k


class AnswerGenerator(Protocol):
    def answer_query(self, query: str) -> str: ...

    def answer_with_context(self, query: str, context: str) -> str: ...


class ConfidenceEstimator(Protocol):
    task_type: str

    def score(self, item: object) -> StructuralConfidenceResult: ...


@dataclass(frozen=True)
class ConfidenceGainResult:
    base_score: float
    context_score: float
    gain: float
    base_result: StructuralConfidenceResult
    context_result: StructuralConfidenceResult


@dataclass(frozen=True)
class ConfidenceGainStrategy:
    base_estimator: ConfidenceEstimator
    context_estimator: ConfidenceEstimator
    answer_generator: AnswerGenerator
    max_document_chars: int = 4000
    algorithm: str = "confidence_gain"

    def __post_init__(self) -> None:
        if self.algorithm != "confidence_gain":
            raise ValueError('algorithm must be "confidence_gain"')
        if self.max_document_chars < 1:
            raise ValueError("max_document_chars must be greater than 0")
        if self.base_estimator.task_type != "query_answerability_confidence":
            raise RerankInputError(
                "base_estimator must use query_answerability_confidence"
            )
        if (
            self.context_estimator.task_type
            != "query_context_answerability_confidence"
        ):
            raise RerankInputError(
                "context_estimator must use "
                "query_context_answerability_confidence"
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
        _require_non_empty(query, "query")
        self._validate_documents(documents)
        if not documents:
            return []

        base_answer = _call_answer_query(self.answer_generator, query)
        base_result = self.base_estimator.score(
            QueryAnswerabilityConfidenceInput(query=query, answer=base_answer)
        )
        base_score = _probability(base_result.score, "base confidence")

        scored: list[tuple[int, ConfidenceGainResult]] = []
        for original_index, document in enumerate(documents):
            context_answer = _call_answer_with_context(
                self.answer_generator,
                query,
                document.text,
            )
            context_result = self.context_estimator.score(
                QueryContextAnswerabilityConfidenceInput(
                    query=query,
                    context=document.text,
                    answer=context_answer,
                )
            )
            context_score = _probability(
                context_result.score,
                f"context confidence at index {original_index}",
            )
            gain = context_score - base_score
            if not math.isfinite(gain) or gain < -1 or gain > 1:
                raise RerankStrategyError("confidence gain must be finite in [-1, 1]")
            scored.append(
                (
                    original_index,
                    ConfidenceGainResult(
                        base_score=base_score,
                        context_score=context_score,
                        gain=gain,
                        base_result=base_result,
                        context_result=context_result,
                    ),
                )
            )

        ordered = sorted(scored, key=lambda item: (-item[1].gain, item[0]))
        if top_k is not None:
            ordered = ordered[:top_k]
        return [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={
                    "strategy": "confidence_gain",
                    "algorithm": self.algorithm,
                    "base_confidence": gain_result.base_score,
                    "context_confidence": gain_result.context_score,
                    "confidence_gain": gain_result.gain,
                },
            )
            for rank, (original_index, gain_result) in enumerate(ordered, start=1)
        ]

    def _validate_documents(self, documents: Sequence[Document]) -> None:
        for index, document in enumerate(documents):
            if len(document.text) > self.max_document_chars:
                raise DocumentTooLongError(
                    f"Document at index {index} has {len(document.text)} "
                    f"characters, exceeding max_document_chars={self.max_document_chars}."
                )


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankInputError(f"{name} must not be empty")
    return value


def _call_answer_query(generator: AnswerGenerator, query: str) -> str:
    try:
        answer = generator.answer_query(query)
    except RerankProviderError:
        raise
    except Exception as exc:
        raise RerankProviderError(str(exc)) from exc
    return _answer_text(answer, "answer_query")


def _call_answer_with_context(
    generator: AnswerGenerator,
    query: str,
    context: str,
) -> str:
    try:
        answer = generator.answer_with_context(query, context)
    except RerankProviderError:
        raise
    except Exception as exc:
        raise RerankProviderError(str(exc)) from exc
    return _answer_text(answer, "answer_with_context")


def _answer_text(value: object, caller: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankProviderError(f"{caller} returned empty answer.")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RerankStrategyError(f"{name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or score < 0 or score > 1:
        raise RerankStrategyError(f"{name} must be finite in [0, 1]")
    return score
```

Modify `src/ranksmith/strategies/__init__.py` to export:

```python
from ranksmith.strategies._confidence_gain import (
    AnswerGenerator,
    ConfidenceGainResult,
    ConfidenceGainStrategy,
)
```

- [ ] **Step 4: Run strategy tests and existing strategy tests**

Run:

```bash
uv run pytest tests/test_confidence_gain_strategy.py tests/test_*strategy*.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/ranksmith/strategies tests/test_confidence_gain_strategy.py
git commit -m "feat: add confidence gain strategy"
```

---

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/wiki/02_architecture.md`
- Modify: `docs/wiki/04_references_index.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/specs/spec_confidence_gain_reranking.md`

- [ ] **Step 1: Update architecture docs**

In `docs/wiki/02_architecture.md`, update the Confidence section:

```markdown
`ConfidenceGainStrategy`는 `ranksmith.confidence`의 answerability confidence scorer를 사용해 `Conf(Q+C)-Conf(Q)`를 계산하고 문서를 정렬하는 Strategy다.

현재 범위:
- query-only answerability confidence
- query+context answerability confidence
- sync confidence gain reranking

제외:
- CBDR retrieval skip
- async confidence gain reranking
- reranker fine-tuning
```

- [ ] **Step 2: Update README and Korean README with matching structure**

Add a short usage section to both `README.md` and `README.ko.md`.

English example:

```python
from ranksmith.confidence import StructuralConfidenceEstimator
from ranksmith.strategies import ConfidenceGainStrategy

base_estimator = StructuralConfidenceEstimator.from_artifact(
    "query-answerability.joblib"
)
context_estimator = StructuralConfidenceEstimator.from_artifact(
    "query-context-answerability.joblib"
)

strategy = ConfidenceGainStrategy(
    base_estimator=base_estimator,
    context_estimator=context_estimator,
    answer_generator=my_answer_generator,
)
```

Korean README must mirror the same section order, table shape, and no benchmark claims.

- [ ] **Step 3: Mark spec checklist implementation items complete**

In `docs/specs/spec_confidence_gain_reranking.md`, mark completed task checklist items as `[x]` only after the corresponding implementation and tests pass.

- [ ] **Step 4: Run targeted checks**

Run:

```bash
uv run pytest tests/test_confidence_answerability_tasks.py tests/test_confidence_training_answerability.py tests/test_confidence_generation_answerability.py tests/test_confidence_gain_strategy.py -q
uv run ruff check src/ranksmith tests
uv run mypy src/ranksmith tests
```

Expected:
- pytest passes.
- ruff passes.
- mypy passes.

- [ ] **Step 5: Run full verification**

Run:

```bash
./scripts/verify.sh
```

Expected: all tests, lint, format, type check, and build pass.

- [ ] **Step 6: Commit docs and verification updates**

```bash
git add docs README.md README.ko.md
git commit -m "docs: document confidence gain reranking"
```

---

## Self-Review

### Spec Coverage
- Runtime query-only/contextual confidence tasks: Task 1.
- Training schema/task expansion: Task 2.
- Generation dataset expansion: Task 3.
- Confidence gain utility/Strategy: Task 4.
- Docs and verification: Task 5.
- CBDR retrieval skip: intentionally excluded by spec.
- Async strategy: intentionally excluded by spec.
- Benchmark claims: intentionally excluded by spec.

### Placeholder Scan
- No placeholder markers or unspecified implementation steps remain.
- Each task has explicit files, tests, commands, and expected outcomes.

### Type Consistency
- New task names are consistently:
  - `query_answerability_confidence`
  - `query_context_answerability_confidence`
- New runtime inputs are consistently:
  - `QueryAnswerabilityConfidenceInput`
  - `QueryContextAnswerabilityConfidenceInput`
- Strategy name is consistently `ConfidenceGainStrategy`.
