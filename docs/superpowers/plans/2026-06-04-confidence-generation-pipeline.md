# Confidence Generation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ranksmith.confidence_generation`, a sync closed-model generation utility that converts raw answer/relevance JSONL into confidence-training canonical JSONL.

**Architecture:** Keep generation separate from `confidence_training`: generation owns provider calls, parsing, label derivation, JSONL output, and resume; training still only consumes canonical JSONL. Use small private modules for types/errors, parsing, labeling, IO, prompts, and pipelines, with public exports only from `ranksmith.confidence_generation`.

**Tech Stack:** Python 3.10+, dataclasses, standard-library JSONL IO, existing `ModelProvider` / `ModelRequest` / `ModelResponse`, existing `RerankProviderError` and `RerankUsage`.

---

## File Map

- Create `src/ranksmith/confidence_generation/_errors.py`
  - Generation-specific error hierarchy.
- Create `src/ranksmith/confidence_generation/_types.py`
  - Config/result/raw sample dataclasses and validation.
- Create `src/ranksmith/confidence_generation/_labeling.py`
  - Normalized exact answer matching and relevance truth conversion.
- Create `src/ranksmith/confidence_generation/_parsing.py`
  - Strict JSON object parsing for closed model outputs.
- Create `src/ranksmith/confidence_generation/_io.py`
  - Raw JSONL loading, output writer policy, completed-id loading.
- Create `src/ranksmith/confidence_generation/_prompts.py`
  - Answer/relevance prompt builders.
- Create `src/ranksmith/confidence_generation/_pipeline.py`
  - `generate_answer_confidence_dataset(...)` and `generate_judgment_confidence_dataset(...)`.
- Create `src/ranksmith/confidence_generation/__init__.py`
  - Public submodule exports only.
- Create tests:
  - `tests/test_confidence_generation_api.py`
  - `tests/test_confidence_generation_labeling.py`
  - `tests/test_confidence_generation_parsing.py`
  - `tests/test_confidence_generation_io.py`
  - `tests/test_confidence_generation_pipeline.py`
- Modify docs:
  - `docs/wiki/02_architecture.md`
  - `README.md`
  - `README.ko.md`
  - `docs/specs/spec_confidence_generation_pipeline.md`

---

### Task 1: Public API, Types, And Errors

**Files:**
- Create: `src/ranksmith/confidence_generation/_errors.py`
- Create: `src/ranksmith/confidence_generation/_types.py`
- Create: `src/ranksmith/confidence_generation/__init__.py`
- Test: `tests/test_confidence_generation_api.py`

- [ ] **Step 1: Write failing API/type tests**

Create `tests/test_confidence_generation_api.py`:

```python
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ranksmith.model import ModelResponse


class FakeProvider:
    def complete(self, request):  # noqa: ANN001
        return ModelResponse(content='{"answer":"ok"}')


def test_confidence_generation_public_submodule_exports_are_available() -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    assert hasattr(generation, "AnswerGenerationConfig")
    assert hasattr(generation, "RelevanceGenerationConfig")
    assert hasattr(generation, "ConfidenceGenerationResult")
    assert hasattr(generation, "ConfidenceGenerationError")
    assert hasattr(generation, "ConfidenceGenerationInputError")
    assert hasattr(generation, "ConfidenceGenerationParseError")
    assert hasattr(generation, "generate_answer_confidence_dataset")
    assert hasattr(generation, "generate_judgment_confidence_dataset")


def test_confidence_generation_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    assert not hasattr(ranksmith, "AnswerGenerationConfig")
    assert not hasattr(ranksmith, "RelevanceGenerationConfig")
    assert not hasattr(ranksmith, "ConfidenceGenerationError")


def test_answer_generation_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            overwrite=True,
            resume=True,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_items=0,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_context_chars=0,
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.AnswerGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            no_answer_value=" ",
        )


def test_relevance_generation_config_rejects_invalid_options(tmp_path: Path) -> None:
    generation = importlib.import_module("ranksmith.confidence_generation")

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.RelevanceGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            truth_positive_operator="eq",  # type: ignore[arg-type]
        )

    with pytest.raises(generation.ConfidenceGenerationInputError):
        generation.RelevanceGenerationConfig(
            input_path=tmp_path / "in.jsonl",
            output_path=tmp_path / "out.jsonl",
            provider=FakeProvider(),
            max_document_chars=0,
        )
```

- [ ] **Step 2: Run API tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_api.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ranksmith.confidence_generation'`.

- [ ] **Step 3: Implement errors, config/result types, and exports**

Create `src/ranksmith/confidence_generation/_errors.py`:

```python
from __future__ import annotations


class ConfidenceGenerationError(Exception):
    """Base error for confidence generation."""


class ConfidenceGenerationInputError(ConfidenceGenerationError):
    """Raised when confidence generation input or config is invalid."""


class ConfidenceGenerationParseError(ConfidenceGenerationError):
    """Raised when closed model output cannot be parsed."""
```

Create `src/ranksmith/confidence_generation/_types.py`:

```python
from __future__ import annotations

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
            overwrite=self.overwrite,
            resume=self.resume,
            max_items=self.max_items,
            source=self.source,
        )
        _validate_positive_int("max_context_chars", self.max_context_chars)
        if not isinstance(self.no_answer_value, str) or not self.no_answer_value.strip():
            raise ConfidenceGenerationInputError(
                "no_answer_value must be a non-empty string"
            )


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
    overwrite: bool,
    resume: bool,
    max_items: int | None,
    source: str | None,
) -> None:
    if overwrite and resume:
        raise ConfidenceGenerationInputError(
            "overwrite and resume cannot both be true"
        )
    if max_items is not None:
        _validate_positive_int("max_items", max_items)
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ConfidenceGenerationInputError("source must be a non-empty string")


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfidenceGenerationInputError(f"{name} must be an int")
    if value < 1:
        raise ConfidenceGenerationInputError(f"{name} must be >= 1")
```

Create `src/ranksmith/confidence_generation/__init__.py`:

```python
from ranksmith.confidence_generation._errors import (
    ConfidenceGenerationError,
    ConfidenceGenerationInputError,
    ConfidenceGenerationParseError,
)
from ranksmith.confidence_generation._pipeline import (
    generate_answer_confidence_dataset,
    generate_judgment_confidence_dataset,
)
from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
)

__all__ = [
    "AnswerGenerationConfig",
    "ConfidenceGenerationError",
    "ConfidenceGenerationInputError",
    "ConfidenceGenerationParseError",
    "ConfidenceGenerationResult",
    "RelevanceGenerationConfig",
    "generate_answer_confidence_dataset",
    "generate_judgment_confidence_dataset",
]
```

Create temporary stubs in `src/ranksmith/confidence_generation/_pipeline.py` so imports resolve:

```python
from __future__ import annotations

from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
    ConfidenceGenerationResult,
    RelevanceGenerationConfig,
)


def generate_answer_confidence_dataset(
    config: AnswerGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError


def generate_judgment_confidence_dataset(
    config: RelevanceGenerationConfig,
) -> ConfidenceGenerationResult:
    raise NotImplementedError
```

- [ ] **Step 4: Run API tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ranksmith/confidence_generation tests/test_confidence_generation_api.py
git commit -m "feat: add confidence generation API shell"
```

---

### Task 2: Labeling And Strict Output Parsing

**Files:**
- Create: `src/ranksmith/confidence_generation/_labeling.py`
- Create: `src/ranksmith/confidence_generation/_parsing.py`
- Test: `tests/test_confidence_generation_labeling.py`
- Test: `tests/test_confidence_generation_parsing.py`

- [ ] **Step 1: Write failing labeling tests**

Create `tests/test_confidence_generation_labeling.py`:

```python
from __future__ import annotations

import pytest

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation._labeling import (
    normalized_exact_match,
    relevance_truth,
)


def test_normalized_exact_match_uses_simple_normalization() -> None:
    assert normalized_exact_match(" Nancy   Travis ", "nancy travis")
    assert normalized_exact_match("Nancy Travis", ["Other", "nancy travis"])
    assert not normalized_exact_match("Nancy T.", "Nancy Travis")


def test_no_answer_value_is_always_mismatch() -> None:
    assert not normalized_exact_match(
        "__NO_ANSWER__",
        "__NO_ANSWER__",
        no_answer_value="__NO_ANSWER__",
    )


def test_relevance_truth_defaults_to_gt_zero() -> None:
    assert relevance_truth(1, threshold=0.0, operator="gt") == "relevant"
    assert relevance_truth(0, threshold=0.0, operator="gt") == "not_relevant"
    assert relevance_truth(True, threshold=100.0, operator="gt") == "relevant"
    assert relevance_truth(False, threshold=-1.0, operator="gte") == "not_relevant"


def test_relevance_truth_supports_gte_threshold() -> None:
    assert relevance_truth(2, threshold=2.0, operator="gte") == "relevant"
    assert relevance_truth(1, threshold=2.0, operator="gte") == "not_relevant"


def test_relevance_truth_rejects_invalid_values() -> None:
    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth("1", threshold=0.0, operator="gt")  # type: ignore[arg-type]

    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(1, threshold=0.0, operator="eq")  # type: ignore[arg-type]
```

- [ ] **Step 2: Write failing parsing tests**

Create `tests/test_confidence_generation_parsing.py`:

```python
from __future__ import annotations

import pytest

from ranksmith.confidence_generation._errors import ConfidenceGenerationParseError
from ranksmith.confidence_generation._parsing import (
    parse_answer_output,
    parse_relevance_output,
)


def test_parse_answer_output_accepts_exact_shape() -> None:
    assert parse_answer_output('{"answer":"Nancy Travis"}') == "Nancy Travis"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"answer":""}',
        '{"answer":"x","rationale":"extra"}',
    ],
)
def test_parse_answer_output_rejects_invalid_shape(content: str) -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_answer_output(content)


def test_parse_relevance_output_accepts_supported_values() -> None:
    assert parse_relevance_output('{"judgment":"relevant"}') == "relevant"
    assert parse_relevance_output('{"judgment":"not_relevant"}') == "not_relevant"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"judgment":"maybe"}',
        '{"judgment":"relevant","confidence":0.9}',
    ],
)
def test_parse_relevance_output_rejects_invalid_shape(content: str) -> None:
    with pytest.raises(ConfidenceGenerationParseError):
        parse_relevance_output(content)
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_labeling.py tests/test_confidence_generation_parsing.py -q
```

Expected: FAIL because modules/functions are missing.

- [ ] **Step 4: Implement labeling**

Create `src/ranksmith/confidence_generation/_labeling.py`:

```python
from __future__ import annotations

import re
from typing import Literal

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError

JudgmentValue = Literal["relevant", "not_relevant"]


def normalized_exact_match(
    answer: str,
    gold_answer: str | list[str],
    *,
    no_answer_value: str = "__NO_ANSWER__",
) -> bool:
    if answer == no_answer_value:
        return False
    normalized_answer = _normalize(answer)
    candidates = [gold_answer] if isinstance(gold_answer, str) else gold_answer
    return any(normalized_answer == _normalize(candidate) for candidate in candidates)


def relevance_truth(
    value: int | float | bool,
    *,
    threshold: float,
    operator: str,
) -> JudgmentValue:
    if isinstance(value, bool):
        return "relevant" if value else "not_relevant"
    if not isinstance(value, (int, float)):
        raise ConfidenceGenerationInputError("relevance_label must be numeric or bool")
    if operator == "gt":
        return "relevant" if value > threshold else "not_relevant"
    if operator == "gte":
        return "relevant" if value >= threshold else "not_relevant"
    raise ConfidenceGenerationInputError('operator must be "gt" or "gte"')


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
```

- [ ] **Step 5: Implement strict parsing**

Create `src/ranksmith/confidence_generation/_parsing.py`:

```python
from __future__ import annotations

import json
from collections.abc import Mapping

from ranksmith.confidence_generation._errors import ConfidenceGenerationParseError
from ranksmith.confidence_generation._labeling import JudgmentValue


def parse_answer_output(content: str) -> str:
    data = _parse_json_object(content)
    _require_exact_keys(data, {"answer"})
    answer = data["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ConfidenceGenerationParseError("answer must be a non-empty string")
    return answer


def parse_relevance_output(content: str) -> JudgmentValue:
    data = _parse_json_object(content)
    _require_exact_keys(data, {"judgment"})
    judgment = data["judgment"]
    if judgment == "relevant":
        return "relevant"
    if judgment == "not_relevant":
        return "not_relevant"
    raise ConfidenceGenerationParseError(
        'judgment must be "relevant" or "not_relevant"'
    )


def _parse_json_object(content: str) -> Mapping[str, object]:
    if content == "":
        raise ConfidenceGenerationParseError("model output must not be empty")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConfidenceGenerationParseError(
            "model output must be valid JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise ConfidenceGenerationParseError("model output must be a JSON object")
    return value


def _require_exact_keys(data: Mapping[str, object], expected: set[str]) -> None:
    keys = set(data)
    if keys != expected:
        missing = sorted(expected - keys)
        if missing:
            raise ConfidenceGenerationParseError(
                f"model output missing required field: {missing[0]}"
            )
        unexpected = sorted(keys - expected)
        raise ConfidenceGenerationParseError(
            f"model output has unexpected field: {unexpected[0]}"
        )
```

- [ ] **Step 6: Run labeling/parsing tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_labeling.py tests/test_confidence_generation_parsing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ranksmith/confidence_generation tests/test_confidence_generation_labeling.py tests/test_confidence_generation_parsing.py
git commit -m "feat: add confidence generation parsing"
```

---

### Task 3: JSONL IO, Raw Sample Validation, And Resume IDs

**Files:**
- Create: `src/ranksmith/confidence_generation/_io.py`
- Test: `tests/test_confidence_generation_io.py`

- [ ] **Step 1: Write failing IO tests**

Create `tests/test_confidence_generation_io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation._io import (
    load_answer_generation_samples,
    load_completed_ids,
    load_relevance_generation_samples,
    open_output_path,
    write_jsonl_row,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_answer_generation_samples_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a1",
                "query": "q",
                "context": "ctx",
                "gold_answer": ["gold"],
                "metadata": {"dataset": "d"},
            }
        ],
    )

    samples = load_answer_generation_samples(path, max_context_chars=10)

    assert samples[0].id == "a1"
    assert samples[0].metadata["dataset"] == "d"


def test_load_relevance_generation_samples_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "relevance.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "relevance_label": 1,
            }
        ],
    )

    samples = load_relevance_generation_samples(path, max_document_chars=10)

    assert samples[0].id == "j1"
    assert samples[0].relevance_label == 1


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a1", "query": "q", "context": "ctx"},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": "g", "x": 1},
        {"id": "a1", "query": " ", "context": "ctx", "gold_answer": "g"},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": []},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": [""]},
        {"id": "a1", "query": "q", "context": "ctx", "gold_answer": "g", "metadata": []},
    ],
)
def test_load_answer_generation_samples_rejects_invalid_rows(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ConfidenceGenerationInputError):
        load_answer_generation_samples(path, max_context_chars=4000)


def test_load_answer_generation_samples_rejects_too_long_context(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(path, [{"id": "a1", "query": "q", "context": "12345", "gold_answer": "g"}])

    with pytest.raises(ConfidenceGenerationInputError):
        load_answer_generation_samples(path, max_context_chars=4)


def test_load_relevance_generation_samples_rejects_too_long_document(tmp_path: Path) -> None:
    path = tmp_path / "rel.jsonl"
    _write_jsonl(path, [{"id": "j1", "query": "q", "document": "12345", "relevance_label": 1}])

    with pytest.raises(ConfidenceGenerationInputError):
        load_relevance_generation_samples(path, max_document_chars=4)


def test_duplicate_ids_fail(tmp_path: Path) -> None:
    path = tmp_path / "answer.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
        ],
    )

    with pytest.raises(ConfidenceGenerationInputError, match="duplicate id"):
        load_answer_generation_samples(path, max_context_chars=4000)


def test_output_policy_rejects_existing_file_without_overwrite_or_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfidenceGenerationInputError):
        open_output_path(path, overwrite=False, resume=False)


def test_output_policy_supports_overwrite_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"id":"old"}\n', encoding="utf-8")

    with open_output_path(path, overwrite=True, resume=False) as handle:
        write_jsonl_row(handle, {"id": "new"})
    assert path.read_text(encoding="utf-8") == '{"id": "new"}\n'

    with open_output_path(path, overwrite=False, resume=True) as handle:
        write_jsonl_row(handle, {"id": "next"})
    assert path.read_text(encoding="utf-8").endswith('{"id": "next"}\n')


def test_load_completed_ids_rejects_duplicates_and_task_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 1}])

    assert load_completed_ids(path, task_type="answer_confidence") == {"a1"}

    with pytest.raises(ConfidenceGenerationInputError):
        load_completed_ids(path, task_type="judgment_confidence")

    _write_jsonl(path, [{"id": "a1", "context": "c", "answer": "a", "label": 1}, {"id": "a1", "context": "c", "answer": "a", "label": 0}])
    with pytest.raises(ConfidenceGenerationInputError):
        load_completed_ids(path, task_type="answer_confidence")
```

- [ ] **Step 2: Run IO tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_io.py -q
```

Expected: FAIL because `_io.py` is missing.

- [ ] **Step 3: Implement IO helpers**

Create `src/ranksmith/confidence_generation/_io.py` with these functions:

```python
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, TextIO
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from ranksmith.confidence_generation._errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation._types import (
    AnswerGenerationSample,
    RelevanceGenerationSample,
)

TaskType = Literal["answer_confidence", "judgment_confidence"]

_ANSWER_REQUIRED = ("id", "query", "context", "gold_answer")
_ANSWER_ALLOWED = {*_ANSWER_REQUIRED, "source", "group_id", "metadata"}
_RELEVANCE_REQUIRED = ("id", "query", "document", "relevance_label")
_RELEVANCE_ALLOWED = {*_RELEVANCE_REQUIRED, "source", "group_id", "metadata"}


def load_answer_generation_samples(
    path: str | Path,
    *,
    max_context_chars: int,
) -> list[AnswerGenerationSample]:
    samples: list[AnswerGenerationSample] = []
    seen: set[str] = set()
    for line_number, row in _read_jsonl_objects(Path(path)):
        try:
            _validate_keys(row, required=_ANSWER_REQUIRED, allowed=_ANSWER_ALLOWED)
            sample = AnswerGenerationSample(
                id=_required_text(row, "id"),
                query=_required_text(row, "query"),
                context=_bounded_text(row, "context", max_context_chars),
                gold_answer=_gold_answer(row["gold_answer"]),
                source=_optional_text(row.get("source"), "source"),
                group_id=_optional_text(row.get("group_id"), "group_id"),
                metadata=_metadata(row.get("metadata")),
            )
        except ConfidenceGenerationInputError as exc:
            raise ConfidenceGenerationInputError(f"line {line_number}: {exc}") from exc
        _check_duplicate(sample.id, seen)
        samples.append(sample)
    return samples


def load_relevance_generation_samples(
    path: str | Path,
    *,
    max_document_chars: int,
) -> list[RelevanceGenerationSample]:
    samples: list[RelevanceGenerationSample] = []
    seen: set[str] = set()
    for line_number, row in _read_jsonl_objects(Path(path)):
        try:
            _validate_keys(row, required=_RELEVANCE_REQUIRED, allowed=_RELEVANCE_ALLOWED)
            sample = RelevanceGenerationSample(
                id=_required_text(row, "id"),
                query=_required_text(row, "query"),
                document=_bounded_text(row, "document", max_document_chars),
                relevance_label=_relevance_label(row["relevance_label"]),
                source=_optional_text(row.get("source"), "source"),
                group_id=_optional_text(row.get("group_id"), "group_id"),
                metadata=_metadata(row.get("metadata")),
            )
        except ConfidenceGenerationInputError as exc:
            raise ConfidenceGenerationInputError(f"line {line_number}: {exc}") from exc
        _check_duplicate(sample.id, seen)
        samples.append(sample)
    return samples


@contextmanager
def open_output_path(
    path: str | Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Iterator[TextIO]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and not resume:
        raise ConfidenceGenerationInputError(
            "output_path already exists; use overwrite or resume"
        )
    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        yield handle


def write_jsonl_row(handle: TextIO, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    handle.flush()


def load_completed_ids(path: str | Path, *, task_type: TaskType) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for line_number, row in _read_jsonl_objects(output_path):
        row_id = _required_text(row, "id")
        if row_id in ids:
            raise ConfidenceGenerationInputError(f"duplicate id in output: {row_id}")
        if task_type == "answer_confidence":
            if "context" not in row or "answer" not in row:
                raise ConfidenceGenerationInputError(
                    f"line {line_number}: output task mismatch"
                )
        elif "query" not in row or "document" not in row or "judgment" not in row:
            raise ConfidenceGenerationInputError(
                f"line {line_number}: output task mismatch"
            )
        ids.add(row_id)
    return ids


def _read_jsonl_objects(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfidenceGenerationInputError("jsonl file could not be read") from exc
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfidenceGenerationInputError(
                f"line {line_number}: must be valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ConfidenceGenerationInputError(
                f"line {line_number}: must be a JSON object"
            )
        rows.append((line_number, value))
    return rows


def _validate_keys(
    row: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    allowed: set[str],
) -> None:
    for field in required:
        if field not in row:
            raise ConfidenceGenerationInputError(f"missing required field: {field}")
    unexpected = sorted(key for key in row if key not in allowed)
    if unexpected:
        raise ConfidenceGenerationInputError(
            f"unexpected field for task: {unexpected[0]}"
        )


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ConfidenceGenerationInputError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ConfidenceGenerationInputError(f"{name} must not be empty")
    return stripped


def _bounded_text(row: Mapping[str, Any], name: str, max_chars: int) -> str:
    value = _required_text(row, name)
    if len(value) > max_chars:
        raise ConfidenceGenerationInputError(f"{name} exceeds max chars")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfidenceGenerationInputError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ConfidenceGenerationInputError(f"{name} must not be empty")
    return stripped


def _gold_answer(value: object) -> str | list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ConfidenceGenerationInputError("gold_answer must not be empty")
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if any(not item.strip() for item in value):
            raise ConfidenceGenerationInputError(
                "gold_answer must not contain empty strings"
            )
        return list(value)
    raise ConfidenceGenerationInputError(
        "gold_answer must be a string or non-empty list of strings"
    )


def _relevance_label(value: object) -> int | float | bool:
    if isinstance(value, bool | int | float):
        return value
    raise ConfidenceGenerationInputError("relevance_label must be numeric or bool")


def _metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfidenceGenerationInputError("metadata must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ConfidenceGenerationInputError("metadata keys must be strings")
    try:
        json.dumps(dict(value))
    except (TypeError, ValueError) as exc:
        raise ConfidenceGenerationInputError(
            "metadata must be JSON-serializable"
        ) from exc
    return dict(value)


def _check_duplicate(sample_id: str, seen: set[str]) -> None:
    if sample_id in seen:
        raise ConfidenceGenerationInputError(f"duplicate id: {sample_id}")
    seen.add(sample_id)
```

- [ ] **Step 4: Run IO tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_io.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ranksmith/confidence_generation/_io.py tests/test_confidence_generation_io.py
git commit -m "feat: add confidence generation jsonl IO"
```

---

### Task 4: Prompt Builders And Provider Call Wrapper

**Files:**
- Create: `src/ranksmith/confidence_generation/_prompts.py`
- Modify: `src/ranksmith/confidence_generation/_pipeline.py`
- Test: `tests/test_confidence_generation_pipeline.py`

- [ ] **Step 1: Write failing prompt/call tests**

Create the first part of `tests/test_confidence_generation_pipeline.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranksmith.confidence_generation import (
    AnswerGenerationConfig,
    RelevanceGenerationConfig,
)
from ranksmith.confidence_generation._pipeline import _call_provider
from ranksmith.confidence_generation._prompts import (
    build_answer_prompt,
    build_relevance_prompt,
)
from ranksmith.confidence_generation._types import (
    AnswerGenerationSample,
    RelevanceGenerationSample,
)
from ranksmith.errors import RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse
from ranksmith.types import RerankUsage


class RecordingProvider:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=self.outputs.pop(0),
            usage=RerankUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )


class BrokenProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("boom")


def test_prompt_builders_include_contract_text() -> None:
    answer = build_answer_prompt(
        AnswerGenerationSample(
            id="a1",
            query="q",
            context="ctx",
            gold_answer="gold",
        ),
        no_answer_value="__NO_ANSWER__",
    )
    assert "Question:\nq" in answer
    assert '{"answer":"__NO_ANSWER__"}' in answer

    relevance = build_relevance_prompt(
        RelevanceGenerationSample(
            id="j1",
            query="q",
            document="doc",
            relevance_label=1,
        )
    )
    assert "Document:\ndoc" in relevance
    assert '"not_relevant"' in relevance


def test_call_provider_uses_json_request_and_emits_usage() -> None:
    provider = RecordingProvider(['{"answer":"ok"}'])
    seen: list[RerankUsage] = []

    content = _call_provider(
        provider,
        system="system",
        user="user",
        on_usage=seen.append,
    )

    assert content == '{"answer":"ok"}'
    assert provider.requests[0].response_format == "json_object"
    assert provider.requests[0].temperature == 0
    assert seen == [RerankUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)]


def test_call_provider_wraps_unexpected_errors() -> None:
    with pytest.raises(RerankProviderError, match="boom"):
        _call_provider(
            BrokenProvider(),
            system="system",
            user="user",
            on_usage=None,
        )
```

- [ ] **Step 2: Run prompt/call tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py -q
```

Expected: FAIL because prompts and `_call_provider` are missing.

- [ ] **Step 3: Implement prompt builders**

Create `src/ranksmith/confidence_generation/_prompts.py`:

```python
from __future__ import annotations

from ranksmith.confidence_generation._types import (
    AnswerGenerationSample,
    RelevanceGenerationSample,
)

ANSWER_SYSTEM_PROMPT = (
    'You answer questions using only the provided context. Return only JSON with an "answer" string.'
)

RELEVANCE_SYSTEM_PROMPT = (
    'You judge document relevance. Return only JSON with a "judgment" value of "relevant" or "not_relevant".'
)


def build_answer_prompt(
    sample: AnswerGenerationSample,
    *,
    no_answer_value: str,
) -> str:
    return (
        f"Question:\n{sample.query}\n\n"
        f"Context:\n{sample.context}\n\n"
        "Return JSON exactly like this shape:\n"
        '{"answer":"..."}\n\n'
        "Use only the context. If the context does not contain the answer, "
        f'return {{"answer":"{no_answer_value}"}}.'
    )


def build_relevance_prompt(sample: RelevanceGenerationSample) -> str:
    return (
        f"Query:\n{sample.query}\n\n"
        f"Document:\n{sample.document}\n\n"
        "Return JSON exactly like this shape:\n"
        '{"judgment":"relevant"}\n\n'
        'Use "relevant" if the document contains information useful for answering the query.\n'
        'Use "not_relevant" otherwise.'
    )
```

- [ ] **Step 4: Implement provider call wrapper**

Replace `src/ranksmith/confidence_generation/_pipeline.py` stub with:

```python
from __future__ import annotations

from ranksmith.confidence_generation._types import (
    AnswerGenerationConfig,
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
    raise NotImplementedError


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
```

- [ ] **Step 5: Run prompt/call tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py -q
```

Expected: PASS for the three prompt/call tests.

- [ ] **Step 6: Commit**

```bash
git add src/ranksmith/confidence_generation tests/test_confidence_generation_pipeline.py
git commit -m "feat: add confidence generation provider calls"
```

---

### Task 5: Answer-Oriented Generation Pipeline

**Files:**
- Modify: `src/ranksmith/confidence_generation/_pipeline.py`
- Test: `tests/test_confidence_generation_pipeline.py`

- [ ] **Step 1: Add failing answer pipeline tests**

Append to `tests/test_confidence_generation_pipeline.py`:

```python
def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_generate_answer_confidence_dataset_writes_canonical_rows(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "a1",
                "query": "Who?",
                "context": "Nancy Travis played Karen.",
                "gold_answer": ["nancy travis"],
                "metadata": {"dataset": "unit"},
            },
            {
                "id": "a2",
                "query": "Who?",
                "context": "No answer here.",
                "gold_answer": "Nancy Travis",
                "source": "row-source",
            },
        ],
    )
    provider = RecordingProvider(
        ['{"answer":" Nancy   Travis "}', '{"answer":"__NO_ANSWER__"}']
    )

    result = generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            source="config-source",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 2
    assert result.generated_count == 2
    assert result.positive_count == 1
    assert result.negative_count == 1
    assert rows[0]["label"] == 1
    assert rows[0]["source"] == "config-source"
    assert rows[0]["metadata"]["input_metadata"] == {"dataset": "unit"}
    assert rows[0]["metadata"]["generation"]["match_policy"] == "normalized_exact"
    assert rows[1]["label"] == 0
    assert rows[1]["source"] == "row-source"


def test_generate_answer_confidence_dataset_respects_resume_and_max_items(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "a1", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a2", "query": "q", "context": "c", "gold_answer": "g"},
            {"id": "a3", "query": "q", "context": "c", "gold_answer": "g"},
        ],
    )
    _write_jsonl(output_path, [{"id": "a1", "context": "c", "answer": "g", "label": 1}])
    provider = RecordingProvider(['{"answer":"g"}'])

    result = generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            resume=True,
            max_items=1,
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 3
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert [row["id"] for row in rows] == ["a1", "a2"]


def test_generate_answer_confidence_dataset_can_omit_raw_output(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_answer_confidence_dataset

    input_path = tmp_path / "answer_input.jsonl"
    output_path = tmp_path / "answer_output.jsonl"
    _write_jsonl(input_path, [{"id": "a1", "query": "q", "context": "c", "gold_answer": "g"}])

    generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=RecordingProvider(['{"answer":"g"}']),
            include_raw_model_output=False,
        )
    )

    row = _read_jsonl(output_path)[0]
    assert "raw_model_output" not in row["metadata"]["generation"]
```

- [ ] **Step 2: Run answer pipeline tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py -q
```

Expected: FAIL with `NotImplementedError` from `generate_answer_confidence_dataset`.

- [ ] **Step 3: Implement answer pipeline**

Add to `src/ranksmith/confidence_generation/_pipeline.py`:

```python
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
from ranksmith.confidence_generation._types import AnswerGenerationSample
```

Replace `generate_answer_confidence_dataset(...)`:

```python
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
            row = _answer_canonical_row(
                sample,
                answer=answer,
                label=label,
                raw_output=raw_output,
                config=config,
            )
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
```

Add helper:

```python
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
```

- [ ] **Step 4: Run answer pipeline tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py tests/test_confidence_generation_io.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ranksmith/confidence_generation/_pipeline.py tests/test_confidence_generation_pipeline.py
git commit -m "feat: generate answer confidence datasets"
```

---

### Task 6: Relevance-Oriented Generation Pipeline

**Files:**
- Modify: `src/ranksmith/confidence_generation/_pipeline.py`
- Test: `tests/test_confidence_generation_pipeline.py`

- [ ] **Step 1: Add failing relevance pipeline tests**

Append to `tests/test_confidence_generation_pipeline.py`:

```python
def test_generate_judgment_confidence_dataset_writes_canonical_rows(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "j1",
                "query": "q",
                "document": "doc",
                "relevance_label": 1,
                "metadata": {"dataset": "unit"},
            },
            {
                "id": "j2",
                "query": "q",
                "document": "doc",
                "relevance_label": 0,
            },
        ],
    )
    provider = RecordingProvider(
        ['{"judgment":"relevant"}', '{"judgment":"relevant"}']
    )

    result = generate_judgment_confidence_dataset(
        RelevanceGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            source="config-source",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.input_count == 2
    assert result.generated_count == 2
    assert result.positive_count == 1
    assert result.negative_count == 1
    assert rows[0]["label"] == 1
    assert rows[0]["source"] == "config-source"
    assert rows[0]["metadata"]["input_metadata"] == {"dataset": "unit"}
    assert rows[0]["metadata"]["generation"]["truth_judgment"] == "relevant"
    assert rows[1]["label"] == 0
    assert rows[1]["metadata"]["generation"]["truth_judgment"] == "not_relevant"


def test_generate_judgment_confidence_dataset_respects_threshold_and_resume(
    tmp_path: Path,
) -> None:
    from ranksmith.confidence_generation import generate_judgment_confidence_dataset

    input_path = tmp_path / "rel_input.jsonl"
    output_path = tmp_path / "rel_output.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "j1", "query": "q", "document": "d", "relevance_label": 2},
            {"id": "j2", "query": "q", "document": "d", "relevance_label": 2},
            {"id": "j3", "query": "q", "document": "d", "relevance_label": 1},
        ],
    )
    _write_jsonl(
        output_path,
        [{"id": "j1", "query": "q", "document": "d", "judgment": "relevant", "label": 1}],
    )
    provider = RecordingProvider(['{"judgment":"relevant"}'])

    result = generate_judgment_confidence_dataset(
        RelevanceGenerationConfig(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            resume=True,
            max_items=1,
            truth_positive_threshold=2.0,
            truth_positive_operator="gte",
        )
    )

    rows = _read_jsonl(output_path)
    assert result.skipped_count == 1
    assert result.generated_count == 1
    assert [row["id"] for row in rows] == ["j1", "j2"]
    assert rows[1]["metadata"]["generation"]["truth_positive_threshold"] == 2.0
```

- [ ] **Step 2: Run relevance pipeline tests and verify failure**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py -q
```

Expected: FAIL with `NotImplementedError` from `generate_judgment_confidence_dataset`.

- [ ] **Step 3: Implement relevance pipeline**

Add imports to `src/ranksmith/confidence_generation/_pipeline.py`:

```python
from ranksmith.confidence_generation._io import load_relevance_generation_samples
from ranksmith.confidence_generation._labeling import relevance_truth
from ranksmith.confidence_generation._parsing import parse_relevance_output
from ranksmith.confidence_generation._prompts import (
    RELEVANCE_SYSTEM_PROMPT,
    build_relevance_prompt,
)
from ranksmith.confidence_generation._types import RelevanceGenerationSample
```

Replace `generate_judgment_confidence_dataset(...)`:

```python
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
            row = _judgment_canonical_row(
                sample,
                judgment=judgment,
                truth=truth,
                label=label,
                raw_output=raw_output,
                config=config,
            )
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
```

Add helper:

```python
def _judgment_canonical_row(
    sample: RelevanceGenerationSample,
    *,
    judgment: str,
    truth: str,
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
```

- [ ] **Step 4: Run pipeline tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Run all generation tests**

Run:

```bash
uv run pytest tests/test_confidence_generation_*.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ranksmith/confidence_generation tests/test_confidence_generation_pipeline.py
git commit -m "feat: generate judgment confidence datasets"
```

---

### Task 7: Documentation, Architecture, And Final Verification

**Files:**
- Modify: `docs/wiki/02_architecture.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/specs/spec_confidence_generation_pipeline.md`
- Test: existing test suite

- [ ] **Step 1: Update architecture wiki**

Add to `docs/wiki/02_architecture.md` under the Confidence section:

```markdown
`ranksmith.confidence_generation`은 closed model output을 생성해 confidence training canonical JSONL로 저장하는 utility layer다.

현재 범위:
- answer-oriented raw JSONL -> `answer_confidence` canonical JSONL
- relevance-oriented raw JSONL -> `judgment_confidence` canonical JSONL
- sync closed model call
- resume 가능한 JSONL output

제외:
- async generation
- dataset adapter
- CLI
- runtime reranking Strategy 또는 Algorithm
```

- [ ] **Step 2: Update README/README.ko minimally**

In `README.md`, add only a short paragraph near Structural Confidence:

```markdown
`ranksmith.confidence_generation` can create supervised canonical JSONL for
confidence training by calling a closed model over raw answer or relevance
examples. It is a data-generation utility, not a reranking Strategy.
```

In `README.ko.md`, add the matched Korean paragraph in the same section position:

```markdown
`ranksmith.confidence_generation`은 raw answer/relevance 예시에 대해 closed
model을 호출해 confidence training용 supervised canonical JSONL을 생성할 수
있습니다. 이 모듈은 reranking Strategy가 아니라 데이터 생성 utility입니다.
```

- [ ] **Step 3: Mark spec checklist implemented**

In `docs/specs/spec_confidence_generation_pipeline.md`:
- Change status to `[ ] Draft | [ ] In Progress | [x] Completed`.
- Mark implementation/test/docs checklist items completed only after the commands below pass.

- [ ] **Step 4: Run targeted verification**

Run:

```bash
uv run pytest tests/test_confidence_generation_*.py -q
uv run ruff check src/ranksmith/confidence_generation tests/test_confidence_generation_*.py
uv run mypy src/ranksmith/confidence_generation tests/test_confidence_generation_*.py
```

Expected:
- pytest PASS
- ruff `All checks passed!`
- mypy `Success: no issues found`

- [ ] **Step 5: Run full verification**

Run:

```bash
./scripts/verify.sh
```

Expected:
- pytest all pass
- ruff/format pass
- mypy pass
- build pass

If build fails with PyPI TLS `UnknownIssuer`, run and record:

```bash
UV_NATIVE_TLS=true ./scripts/verify.sh
```

- [ ] **Step 6: Commit**

```bash
git add docs/wiki/02_architecture.md README.md README.ko.md docs/specs/spec_confidence_generation_pipeline.md
git commit -m "docs: document confidence generation pipeline"
```

---

## Self-Review Checklist

- Spec coverage:
  - Answer-oriented generation: Task 5.
  - Relevance-oriented generation: Task 6.
  - Raw input validation and unexpected fields: Task 3.
  - Strict closed model JSON parsing and unexpected output fields: Task 2.
  - Normalized exact answer matching and no-answer sentinel: Task 2.
  - Relevance threshold/operator truth conversion: Task 2 and Task 6.
  - Output overwrite/resume/max_items behavior: Task 3, Task 5, Task 6.
  - Metadata/source/raw-output policies: Task 5 and Task 6.
  - Public submodule only, no root export: Task 1.
  - Docs and final verification: Task 7.
- Filler scan:
  - No task uses unresolved filler language.
  - Each implementation task has concrete files and commands.
- Scope guard:
  - No async, CLI, adapters, runtime reranking Strategy, or training execution.
