# Structural Confidence Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the official `ranksmith.confidence` inference module for single-item structural confidence scoring from a frozen HuggingFace encoder and a pre-trained scorer artifact.

**Architecture:** Implement `ranksmith.confidence` as a submodule, not a reranking Strategy and not a root export. The module is split into focused files for public types, errors, lazy dependency loading, input templates, structural features, HuggingFace encoding, scorer loading, and estimator orchestration.

**Tech Stack:** Python 3.10+, dataclasses, Protocol, optional `numpy`, `scipy`, `torch`, `transformers`, `joblib`, `lightgbm`, pytest, mypy strict mode.

---

## Scope Lock

Implement only Phase 1 from `docs/specs/spec_structural_confidence.md`.

Build:
- `ranksmith.confidence` submodule
- `AnswerConfidenceInput`
- `JudgmentConfidenceInput`
- `StructuralConfidenceEstimator.score()` single sync inference
- `structural-v1` 70-dimensional structural feature extraction
- scorer protocol
- joblib wrapper loader
- LightGBM Booster + metadata JSON loader
- CPU-only frozen HuggingFace encoder wrapper
- confidence-specific errors
- optional dependency lazy import
- minimal README / README.ko usage docs

Do not build:
- training pipeline
- dataset or label generation
- feature cache
- calibration or evaluation pipeline
- semantic feature or Struct+Sent fusion
- batch inference
- async API
- reranking Strategy
- non-CPU device support
- root import export
- artifact save/export helper
- benchmark numbers or performance claims

---

## File Structure

- Create `src/ranksmith/confidence/__init__.py`
  - Public submodule exports only.
- Create `src/ranksmith/confidence/_errors.py`
  - `ConfidenceError`, `ConfidenceDependencyError`, `ConfidenceInputError`, `ConfidenceArtifactError`.
- Create `src/ranksmith/confidence/_types.py`
  - Frozen dataclasses and protocols.
- Create `src/ranksmith/confidence/_dependencies.py`
  - Lazy optional imports.
- Create `src/ranksmith/confidence/_templates.py`
  - Input validation and exact template formatting.
- Create `src/ranksmith/confidence/_features.py`
  - `structural-v1` feature extraction.
- Create `src/ranksmith/confidence/_encoder.py`
  - HuggingFace encoder wrapper.
- Create `src/ranksmith/confidence/_scorer.py`
  - scorer protocol helpers and LightGBM/joblib loaders.
- Create `src/ranksmith/confidence/_structural.py`
  - `StructuralConfidenceEstimator`.
- Modify `pyproject.toml`
  - Add `confidence` optional extra.
- Modify `docs/wiki/02_architecture.md`
  - Add confidence utility layer.
- Create `docs/wiki/references/structural_confidence.md`
  - Trust reference summary.
- Modify `docs/wiki/04_references_index.md`
  - Mark Trust reference as summarized.
- Modify `README.md` and `README.ko.md`
  - Add minimal usage and optional extra only.

Tests:
- Create `tests/test_confidence_types.py`
- Create `tests/test_confidence_templates.py`
- Create `tests/test_confidence_features.py`
- Create `tests/test_confidence_dependencies.py`
- Create `tests/test_confidence_encoder.py`
- Create `tests/test_confidence_scorer.py`
- Create `tests/test_confidence_estimator.py`
- Create `tests/test_confidence_api_scope.py`
- Create `tests/test_confidence_hf_token.py`
- Create `tests/test_confidence_hf_options.py`
- Create `tests/test_confidence_numeric_stability.py`

---

### Task 1: Public Types And Errors

**Files:**
- Create: `src/ranksmith/confidence/_errors.py`
- Create: `src/ranksmith/confidence/_types.py`
- Create: `src/ranksmith/confidence/__init__.py`
- Test: `tests/test_confidence_types.py`
- Test: `tests/test_confidence_api_scope.py`

- [x] **Step 1: Write failing public type tests**

Create `tests/test_confidence_types.py`:

```python
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceArtifactError,
    ConfidenceDependencyError,
    ConfidenceError,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceResult,
)


def test_answer_confidence_input_is_frozen() -> None:
    item = AnswerConfidenceInput(context="context", answer="answer")

    with pytest.raises(FrozenInstanceError):
        item.answer = "changed"  # type: ignore[misc]


def test_judgment_confidence_input_is_frozen() -> None:
    item = JudgmentConfidenceInput(
        query="query",
        document="document",
        judgment="direct evidence",
    )

    with pytest.raises(FrozenInstanceError):
        item.judgment = "changed"  # type: ignore[misc]


def test_structural_confidence_result_copies_metadata() -> None:
    metadata = {"encoder_name": "bert-base-uncased"}

    result = StructuralConfidenceResult(
        score=0.7,
        task_type="answer_confidence",
        feature_schema_version="structural-v1",
        metadata=metadata,
    )
    metadata["encoder_name"] = "changed"

    assert result.metadata == {"encoder_name": "bert-base-uncased"}


def test_scorer_metadata_preserves_extra_fields() -> None:
    metadata = ScorerMetadata(
        artifact_schema_version="structural-artifact-v1",
        scorer_type="lightgbm",
        task_type="answer_confidence",
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        input_template_version="structural-template-v1",
        feature_schema_version="structural-v1",
        feature_dim=70,
        feature_dtype="float64",
        max_length=256,
        granularity="two_scale",
        local_window_size=5,
        local_stride=2,
        score_output="probability",
        positive_class_index=1,
        extra={"trained_on": "fixture"},
    )

    assert metadata.extra == {"trained_on": "fixture"}


def test_confidence_errors_share_base_class() -> None:
    assert issubclass(ConfidenceDependencyError, ConfidenceError)
    assert issubclass(ConfidenceInputError, ConfidenceError)
    assert issubclass(ConfidenceArtifactError, ConfidenceError)
```

- [x] **Step 2: Write failing API scope test**

Create `tests/test_confidence_api_scope.py`:

```python
from __future__ import annotations

import importlib


def test_confidence_public_submodule_exports_are_available() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert hasattr(confidence, "AnswerConfidenceInput")
    assert hasattr(confidence, "JudgmentConfidenceInput")
    assert hasattr(confidence, "StructuralConfidenceEstimator")
    assert hasattr(confidence, "load_lightgbm_scorer")
    assert hasattr(confidence, "ConfidenceError")


def test_confidence_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    assert not hasattr(ranksmith, "AnswerConfidenceInput")
    assert not hasattr(ranksmith, "JudgmentConfidenceInput")
    assert not hasattr(ranksmith, "StructuralConfidenceEstimator")
    assert not hasattr(ranksmith, "ConfidenceError")


def test_no_batch_or_async_api_in_phase_one() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert not hasattr(confidence.StructuralConfidenceEstimator, "score_batch")
    assert not hasattr(confidence, "AsyncStructuralConfidenceEstimator")
```

- [x] **Step 3: Run failing tests**

Run:

```bash
uv run pytest tests/test_confidence_types.py tests/test_confidence_api_scope.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ranksmith.confidence'`.

- [x] **Step 4: Implement errors**

Create `src/ranksmith/confidence/_errors.py`:

```python
from __future__ import annotations


class ConfidenceError(Exception):
    """Base error for confidence estimation."""


class ConfidenceDependencyError(ConfidenceError):
    """Raised when an optional confidence dependency is unavailable."""


class ConfidenceInputError(ConfidenceError):
    """Raised when confidence input or estimator configuration is invalid."""


class ConfidenceArtifactError(ConfidenceError):
    """Raised when a confidence scorer artifact is invalid or incompatible."""
```

- [x] **Step 5: Implement types**

Create `src/ranksmith/confidence/_types.py`:

```python
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


StructuralConfidenceInput: TypeAlias = (
    AnswerConfidenceInput | JudgmentConfidenceInput
)


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
```

- [x] **Step 6: Implement submodule exports**

Create `src/ranksmith/confidence/__init__.py`:

```python
from ranksmith.confidence._errors import (
    ConfidenceArtifactError,
    ConfidenceDependencyError,
    ConfidenceError,
    ConfidenceInputError,
)
from ranksmith.confidence._scorer import load_lightgbm_scorer
from ranksmith.confidence._structural import StructuralConfidenceEstimator
from ranksmith.confidence._types import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceInput,
    StructuralConfidenceResult,
    StructuralConfidenceScorer,
)

__all__ = [
    "AnswerConfidenceInput",
    "ConfidenceArtifactError",
    "ConfidenceDependencyError",
    "ConfidenceError",
    "ConfidenceInputError",
    "JudgmentConfidenceInput",
    "ScorerMetadata",
    "StructuralConfidenceEstimator",
    "StructuralConfidenceInput",
    "StructuralConfidenceResult",
    "StructuralConfidenceScorer",
    "load_lightgbm_scorer",
]
```

Create temporary stubs so imports resolve; later tasks replace them.

Create `src/ranksmith/confidence/_scorer.py`:

```python
from __future__ import annotations

from pathlib import Path

from ranksmith.confidence._errors import ConfidenceDependencyError


def load_lightgbm_scorer(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> object:
    del path, metadata_path
    raise ConfidenceDependencyError("lightgbm scorer loading is not implemented yet")
```

Create `src/ranksmith/confidence/_structural.py`:

```python
from __future__ import annotations


class StructuralConfidenceEstimator:
    """Structural confidence estimator placeholder replaced in later tasks."""
```

- [x] **Step 7: Run tests**

Run:

```bash
uv run pytest tests/test_confidence_types.py tests/test_confidence_api_scope.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add src/ranksmith/confidence tests/test_confidence_types.py tests/test_confidence_api_scope.py
git commit -m "feat: add confidence public types"
```

---

### Task 2: Optional Dependencies And Templates

**Files:**
- Create: `src/ranksmith/confidence/_dependencies.py`
- Create: `src/ranksmith/confidence/_templates.py`
- Test: `tests/test_confidence_dependencies.py`
- Test: `tests/test_confidence_templates.py`

- [x] **Step 1: Write failing dependency tests**

Create `tests/test_confidence_dependencies.py`:

```python
from __future__ import annotations

import builtins
import importlib

import pytest

from ranksmith.confidence import ConfidenceDependencyError
from ranksmith.confidence._dependencies import import_optional_dependency


def test_import_ranksmith_confidence_without_optional_dependencies() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert confidence.AnswerConfidenceInput is not None


def test_missing_optional_dependency_raises_confidence_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "definitely_missing_package":
            raise ImportError("missing package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ConfidenceDependencyError) as error:
        import_optional_dependency(
            "definitely_missing_package",
            extra="confidence",
        )

    assert "pip install ranksmith[confidence]" in str(error.value)
```

- [x] **Step 2: Write failing template tests**

Create `tests/test_confidence_templates.py`:

```python
from __future__ import annotations

import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceInputError,
    JudgmentConfidenceInput,
)
from ranksmith.confidence._templates import (
    format_confidence_input,
)


def test_formats_answer_confidence_template() -> None:
    text = format_confidence_input(
        "answer_confidence",
        AnswerConfidenceInput(context="passage", answer="answer"),
    )

    assert text == "Context:\npassage\n\nAnswer:\nanswer"


def test_formats_judgment_confidence_template() -> None:
    text = format_confidence_input(
        "judgment_confidence",
        JudgmentConfidenceInput(
            query="query",
            document="document",
            judgment="direct evidence",
        ),
    )

    assert text == "Query:\nquery\n\nDocument:\ndocument\n\nJudgment:\ndirect evidence"


def test_rejects_mismatched_input_type() -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(
            "answer_confidence",
            JudgmentConfidenceInput(
                query="query",
                document="document",
                judgment="direct evidence",
            ),
        )


def test_rejects_whitespace_required_field() -> None:
    with pytest.raises(ConfidenceInputError):
        format_confidence_input(
            "answer_confidence",
            AnswerConfidenceInput(context="  ", answer="answer"),
        )
```

- [x] **Step 3: Run failing tests**

Run:

```bash
uv run pytest tests/test_confidence_dependencies.py tests/test_confidence_templates.py -q
```

Expected: FAIL with missing `_dependencies` and `_templates`.

- [x] **Step 4: Implement lazy dependency helper**

Create `src/ranksmith/confidence/_dependencies.py`:

```python
from __future__ import annotations

import importlib
from types import ModuleType

from ranksmith.confidence._errors import ConfidenceDependencyError


def import_optional_dependency(name: str, *, extra: str = "confidence") -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfidenceDependencyError(
            f"Optional dependency {name!r} is required. "
            f"Install it with `pip install ranksmith[{extra}]`."
        ) from exc
```

- [x] **Step 5: Implement templates**

Create `src/ranksmith/confidence/_templates.py`:

```python
from __future__ import annotations

from ranksmith.confidence._errors import ConfidenceInputError
from ranksmith.confidence._types import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    StructuralConfidenceInput,
    TaskType,
)

INPUT_TEMPLATE_VERSION = "structural-template-v1"


def _require_non_empty(value: str, *, field_name: str) -> str:
    if value.strip() == "":
        raise ConfidenceInputError(f"{field_name} must not be empty")
    return value


def format_confidence_input(
    task_type: TaskType,
    item: StructuralConfidenceInput,
) -> str:
    if task_type == "answer_confidence":
        if not isinstance(item, AnswerConfidenceInput):
            raise ConfidenceInputError(
                "answer_confidence requires AnswerConfidenceInput"
            )
        context = _require_non_empty(item.context, field_name="context")
        answer = _require_non_empty(item.answer, field_name="answer")
        return f"Context:\n{context}\n\nAnswer:\n{answer}"

    if task_type == "judgment_confidence":
        if not isinstance(item, JudgmentConfidenceInput):
            raise ConfidenceInputError(
                "judgment_confidence requires JudgmentConfidenceInput"
            )
        query = _require_non_empty(item.query, field_name="query")
        document = _require_non_empty(item.document, field_name="document")
        judgment = _require_non_empty(item.judgment, field_name="judgment")
        return f"Query:\n{query}\n\nDocument:\n{document}\n\nJudgment:\n{judgment}"

    raise ConfidenceInputError(f"unsupported task_type: {task_type!r}")
```

- [x] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_confidence_dependencies.py tests/test_confidence_templates.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/ranksmith/confidence/_dependencies.py src/ranksmith/confidence/_templates.py tests/test_confidence_dependencies.py tests/test_confidence_templates.py
git commit -m "feat: add confidence dependencies and templates"
```

---

### Task 3: Structural Feature Extraction

**Files:**
- Create: `src/ranksmith/confidence/_features.py`
- Test: `tests/test_confidence_features.py`
- Test: `tests/test_confidence_numeric_stability.py`

- [x] **Step 1: Write failing feature tests**

Create `tests/test_confidence_features.py`:

```python
from __future__ import annotations

import numpy as np

from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    extract_structural_features,
)


def test_extract_structural_features_returns_70_finite_values() -> None:
    hidden_states = np.arange(6 * 4, dtype=np.float64).reshape(6, 4)
    attention_mask = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert FEATURE_SCHEMA_VERSION == "structural-v1"
    assert FEATURE_DIM == 70
    assert FEATURE_DTYPE == "float64"
    assert len(features) == 70
    assert all(np.isfinite(features))


def test_feature_order_has_expected_family_lengths() -> None:
    hidden_states = np.eye(8, 4, dtype=np.float64)
    attention_mask = np.ones(8, dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    spectral = features[:48]
    local = features[48:54]
    shape = features[54:]

    assert len(spectral) == 48
    assert len(local) == 6
    assert len(shape) == 16
```

- [x] **Step 2: Write failing numeric stability tests**

Create `tests/test_confidence_numeric_stability.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from ranksmith.confidence import ConfidenceArtifactError, ConfidenceInputError
from ranksmith.confidence._features import extract_structural_features


def test_rejects_nan_hidden_states() -> None:
    hidden_states = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float64)
    attention_mask = np.array([1, 1], dtype=np.int64)

    with pytest.raises(ConfidenceArtifactError):
        extract_structural_features(hidden_states, attention_mask, max_length=64)


def test_rejects_zero_non_padding_tokens() -> None:
    hidden_states = np.zeros((2, 4), dtype=np.float64)
    attention_mask = np.array([0, 0], dtype=np.int64)

    with pytest.raises(ConfidenceInputError):
        extract_structural_features(hidden_states, attention_mask, max_length=64)


def test_single_token_uses_zero_fallbacks() -> None:
    hidden_states = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    attention_mask = np.array([1], dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert len(features) == 70
    assert all(np.isfinite(features))


def test_degree_zero_graph_stays_finite() -> None:
    hidden_states = np.zeros((4, 3), dtype=np.float64)
    attention_mask = np.ones(4, dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert len(features) == 70
    assert all(np.isfinite(features))


def test_rejects_too_small_max_length() -> None:
    hidden_states = np.zeros((4, 3), dtype=np.float64)
    attention_mask = np.ones(4, dtype=np.int64)

    with pytest.raises(ConfidenceInputError):
        extract_structural_features(hidden_states, attention_mask, max_length=33)
```

- [x] **Step 3: Run failing tests**

Run:

```bash
uv run pytest tests/test_confidence_features.py tests/test_confidence_numeric_stability.py -q
```

Expected: FAIL with missing `_features`.

- [x] **Step 4: Implement structural features**

Create `src/ranksmith/confidence/_features.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._errors import (
    ConfidenceArtifactError,
    ConfidenceInputError,
)

FEATURE_SCHEMA_VERSION = "structural-v1"
FEATURE_DIM = 70
FEATURE_DTYPE = "float64"
GRANULARITY = "two_scale"
LOCAL_WINDOW_SIZE = 5
LOCAL_STRIDE = 2
MIN_MAX_LENGTH = 34
EIGENVALUE_TOLERANCE = 1e-12


def extract_structural_features(
    hidden_states: object,
    attention_mask: object,
    *,
    max_length: int,
) -> list[float]:
    np = import_optional_dependency("numpy")
    scipy_spatial_distance = import_optional_dependency("scipy.spatial.distance")

    if max_length < MIN_MAX_LENGTH:
        raise ConfidenceInputError("max_length must be at least 34")

    hidden = np.asarray(hidden_states, dtype=np.float64)
    mask = np.asarray(attention_mask, dtype=bool)
    if hidden.ndim != 2:
        raise ConfidenceArtifactError("hidden_states must be a 2D array")
    if mask.ndim != 1 or mask.shape[0] != hidden.shape[0]:
        raise ConfidenceArtifactError("attention_mask must match hidden_states rows")
    _ensure_finite(hidden, "hidden_states")

    trajectory = hidden[mask]
    if trajectory.shape[0] == 0:
        raise ConfidenceInputError("at least one non-padding token is required")

    global_descriptor = _descriptor(trajectory, max_length=max_length)
    local_descriptor = _local_descriptor(trajectory, max_length=max_length)
    combined = (np.asarray(global_descriptor) + np.asarray(local_descriptor)) / 2.0
    result = combined.astype(np.float64).tolist()
    del scipy_spatial_distance
    _validate_feature_vector(result)
    return result


def _descriptor(trajectory: object, *, max_length: int) -> list[float]:
    return (
        _frequency_domain_smoothness(trajectory, max_length=max_length)
        + _graph_spectral_diffusion(trajectory)
        + _local_variation(trajectory)
        + _shape_coherence(trajectory)
    )


def _frequency_domain_smoothness(trajectory: object, *, max_length: int) -> list[float]:
    np = import_optional_dependency("numpy")
    token_count, hidden_dim = trajectory.shape
    padded = np.zeros((max_length, hidden_dim), dtype=np.float64)
    padded[: min(token_count, max_length)] = trajectory[:max_length]
    fft_values = np.fft.rfft(padded, axis=0) / float(max_length)
    features: list[float] = []
    for frequency in range(1, 17):
        power = np.abs(fft_values[frequency]) ** 2
        features.append(float(np.mean(power)))
        features.append(float(np.max(power)))
    return features


def _graph_spectral_diffusion(trajectory: object) -> list[float]:
    np = import_optional_dependency("numpy")
    norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
    normalized = np.divide(
        trajectory,
        norms,
        out=np.zeros_like(trajectory, dtype=np.float64),
        where=norms > 0,
    )
    similarity = normalized @ normalized.T
    similarity = np.maximum(similarity, 0.0)
    np.fill_diagonal(similarity, 0.0)

    degree = similarity.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * similarity * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(similarity.shape[0], dtype=np.float64) - normalized_adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)
    checked: list[float] = []
    for value in eigenvalues[:16]:
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ConfidenceArtifactError("laplacian eigenvalue is not finite")
        if scalar < 0:
            if abs(scalar) <= EIGENVALUE_TOLERANCE:
                scalar = 0.0
            else:
                raise ConfidenceArtifactError("laplacian eigenvalue is negative")
        checked.append(scalar)
    while len(checked) < 16:
        checked.append(0.0)
    return checked


def _local_variation(trajectory: object) -> list[float]:
    np = import_optional_dependency("numpy")
    if trajectory.shape[0] <= 1:
        displacements = np.asarray([], dtype=np.float64)
    else:
        displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_path_length = float(np.sum(displacements)) if displacements.size else 0.0
    mean_displacement = float(np.mean(displacements)) if displacements.size else 0.0
    displacement_variance = float(np.var(displacements)) if displacements.size else 0.0
    start_end_distance = (
        float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        if trajectory.shape[0] > 1
        else 0.0
    )
    embedding_variance = float(np.mean(np.var(trajectory, axis=0)))
    centroid_norm = float(np.linalg.norm(np.mean(trajectory, axis=0)))
    return [
        total_path_length,
        mean_displacement,
        displacement_variance,
        start_end_distance,
        embedding_variance,
        centroid_norm,
    ]


def _shape_coherence(trajectory: object) -> list[float]:
    np = import_optional_dependency("numpy")
    if trajectory.shape[0] <= 1:
        return [0.0] * 16
    distances = []
    for left in range(trajectory.shape[0]):
        for right in range(left + 1, trajectory.shape[0]):
            distances.append(float(np.linalg.norm(trajectory[left] - trajectory[right])))
    values = np.asarray(distances, dtype=np.float64)
    max_distance = float(np.max(values)) if values.size else 0.0
    if max_distance > 0:
        values = values / max_distance
    histogram, _ = np.histogram(values, bins=16, range=(0.0, 1.0), density=False)
    total = int(histogram.sum())
    if total == 0:
        return [0.0] * 16
    return (histogram.astype(np.float64) / float(total)).tolist()


def _local_descriptor(trajectory: object, *, max_length: int) -> list[float]:
    np = import_optional_dependency("numpy")
    token_count = trajectory.shape[0]
    if token_count < LOCAL_WINDOW_SIZE:
        return _descriptor(trajectory, max_length=max_length)
    descriptors = []
    for start in range(0, token_count - LOCAL_WINDOW_SIZE + 1, LOCAL_STRIDE):
        window = trajectory[start : start + LOCAL_WINDOW_SIZE]
        descriptors.append(_descriptor(window, max_length=max_length))
    return np.mean(np.asarray(descriptors, dtype=np.float64), axis=0).tolist()


def _ensure_finite(values: object, name: str) -> None:
    np = import_optional_dependency("numpy")
    if not np.all(np.isfinite(values)):
        raise ConfidenceArtifactError(f"{name} contains NaN or Inf")


def _validate_feature_vector(features: Sequence[float]) -> None:
    np = import_optional_dependency("numpy")
    if len(features) != FEATURE_DIM:
        raise ConfidenceArtifactError("structural feature vector must have length 70")
    if not np.all(np.isfinite(np.asarray(features, dtype=np.float64))):
        raise ConfidenceArtifactError("structural feature vector contains NaN or Inf")
```

- [x] **Step 5: Run feature tests**

Run:

```bash
uv run pytest tests/test_confidence_features.py tests/test_confidence_numeric_stability.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/ranksmith/confidence/_features.py tests/test_confidence_features.py tests/test_confidence_numeric_stability.py
git commit -m "feat: add structural confidence features"
```

---

### Task 4: Frozen HuggingFace Encoder

**Files:**
- Create: `src/ranksmith/confidence/_encoder.py`
- Test: `tests/test_confidence_encoder.py`
- Test: `tests/test_confidence_hf_token.py`
- Test: `tests/test_confidence_hf_options.py`

- [x] **Step 1: Write failing encoder tests with fakes**

Create `tests/test_confidence_encoder.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ranksmith.confidence import ConfidenceInputError
from ranksmith.confidence._encoder import FrozenAutoEncoder


@dataclass
class FakeTensor:
    value: list[list[float]]

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> list[list[float]]:
        return self.value


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorch:
    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()


class FakeTokenizer:
    def __init__(self, length: int) -> None:
        self.length = length
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"text": text, **kwargs})
        return {
            "input_ids": [[1] * self.length],
            "attention_mask": [[1] * self.length],
        }


class FakeOutput:
    last_hidden_state = FakeTensor([[1.0, 2.0], [3.0, 4.0]])


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.calls: list[dict[str, object]] = []

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: object) -> FakeOutput:
        self.calls.append(kwargs)
        return FakeOutput()


def test_encoder_sets_eval_and_encodes_to_numpy() -> None:
    tokenizer = FakeTokenizer(length=2)
    model = FakeModel()
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
        max_length=256,
        allow_truncation=False,
    )

    hidden, mask = encoder.encode("hello")

    assert model.eval_called is True
    assert hidden == [[1.0, 2.0], [3.0, 4.0]]
    assert mask == [1, 1]


def test_encoder_preflight_rejects_long_input() -> None:
    tokenizer = FakeTokenizer(length=3)
    model = FakeModel()
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
        max_length=2,
        allow_truncation=False,
    )

    with pytest.raises(ConfidenceInputError):
        encoder.encode("too long")

    assert model.calls == []
```

- [x] **Step 2: Write failing HF option tests**

Create `tests/test_confidence_hf_token.py`:

```python
from __future__ import annotations

from ranksmith.confidence._encoder import build_hf_from_pretrained_kwargs


def test_hf_token_is_forwarded_but_not_returned_as_metadata() -> None:
    kwargs = build_hf_from_pretrained_kwargs(
        revision="main",
        hf_token="secret-token",
        local_files_only=False,
        cache_dir=None,
    )

    assert kwargs["token"] == "secret-token"
    assert "secret-token" not in repr({k: v for k, v in kwargs.items() if k != "token"})
```

Create `tests/test_confidence_hf_options.py`:

```python
from __future__ import annotations

import os

import pytest

from ranksmith.confidence import ConfidenceInputError
from ranksmith.confidence._encoder import (
    HF_LIVE_TEST_ENV,
    build_hf_from_pretrained_kwargs,
    validate_device,
)


def test_local_files_only_and_cache_dir_are_forwarded() -> None:
    kwargs = build_hf_from_pretrained_kwargs(
        revision=None,
        hf_token=None,
        local_files_only=True,
        cache_dir="/tmp/hf-cache",
    )

    assert kwargs["local_files_only"] is True
    assert kwargs["cache_dir"] == "/tmp/hf-cache"


def test_only_cpu_device_is_supported() -> None:
    validate_device("cpu")

    with pytest.raises(ConfidenceInputError):
        validate_device("mps")


def test_hf_live_tests_are_opt_in() -> None:
    assert HF_LIVE_TEST_ENV == "RANKSMITH_RUN_HF_TESTS"
    assert os.environ.get(HF_LIVE_TEST_ENV) is None
```

- [x] **Step 3: Run failing encoder tests**

Run:

```bash
uv run pytest tests/test_confidence_encoder.py tests/test_confidence_hf_token.py tests/test_confidence_hf_options.py -q
```

Expected: FAIL with missing `_encoder`.

- [x] **Step 4: Implement encoder wrapper**

Create `src/ranksmith/confidence/_encoder.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._errors import (
    ConfidenceDependencyError,
    ConfidenceInputError,
)

HF_LIVE_TEST_ENV = "RANKSMITH_RUN_HF_TESTS"


def validate_device(device: str) -> None:
    if device != "cpu":
        raise ConfidenceInputError('Phase 1 supports only device="cpu"')


def build_hf_from_pretrained_kwargs(
    *,
    revision: str | None,
    hf_token: str | None,
    local_files_only: bool,
    cache_dir: str | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if hf_token is not None:
        kwargs["token"] = hf_token
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return kwargs


@dataclass
class FrozenAutoEncoder:
    encoder_name: str
    encoder_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None
    tokenizer: Any
    model: Any
    torch_module: Any
    max_length: int
    allow_truncation: bool

    def __post_init__(self) -> None:
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_name: str,
        encoder_revision: str | None,
        tokenizer_name: str | None,
        tokenizer_revision: str | None,
        hf_token: str | None,
        local_files_only: bool,
        cache_dir: str | None,
        device: str,
        max_length: int,
        allow_truncation: bool,
    ) -> "FrozenAutoEncoder":
        validate_device(device)
        transformers = import_optional_dependency("transformers")
        torch = import_optional_dependency("torch")
        resolved_tokenizer_name = tokenizer_name or encoder_name
        tokenizer_kwargs = build_hf_from_pretrained_kwargs(
            revision=tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        model_kwargs = build_hf_from_pretrained_kwargs(
            revision=encoder_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                resolved_tokenizer_name,
                **tokenizer_kwargs,
            )
            model = transformers.AutoModel.from_pretrained(
                encoder_name,
                **model_kwargs,
            )
        except Exception as exc:
            raise ConfidenceDependencyError(
                "failed to load HuggingFace tokenizer or model"
            ) from exc
        return cls(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=resolved_tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            max_length=max_length,
            allow_truncation=allow_truncation,
        )

    def encode(self, text: str) -> tuple[object, object]:
        preflight_tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        input_ids = preflight_tokens["input_ids"]
        token_count = len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
        if not self.allow_truncation and token_count > self.max_length:
            raise ConfidenceInputError("input exceeds max_length")

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=self.allow_truncation,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        with self.torch_module.no_grad():
            output = self.model(**encoded)

        hidden = output.last_hidden_state.detach().cpu().numpy()
        mask = encoded["attention_mask"]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        hidden_rows = hidden[0].tolist() if hasattr(hidden, "tolist") else hidden
        mask_values = mask[0].tolist() if hasattr(mask, "tolist") else mask[0]
        return hidden_rows, mask_values
```

- [x] **Step 5: Run encoder tests**

Run:

```bash
uv run pytest tests/test_confidence_encoder.py tests/test_confidence_hf_token.py tests/test_confidence_hf_options.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/ranksmith/confidence/_encoder.py tests/test_confidence_encoder.py tests/test_confidence_hf_token.py tests/test_confidence_hf_options.py
git commit -m "feat: add frozen confidence encoder"
```

---

### Task 5: Scorer Metadata And Loaders

**Files:**
- Modify: `src/ranksmith/confidence/_scorer.py`
- Test: `tests/test_confidence_scorer.py`

- [x] **Step 1: Write failing scorer tests**

Create `tests/test_confidence_scorer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ranksmith.confidence import ConfidenceArtifactError, ScorerMetadata
from ranksmith.confidence._scorer import (
    LightGBMScorer,
    metadata_from_dict,
    validate_scorer_metadata,
)


def metadata_dict(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "artifact_schema_version": "structural-artifact-v1",
        "scorer_type": "lightgbm",
        "task_type": "answer_confidence",
        "encoder_name": "bert-base-uncased",
        "encoder_revision": None,
        "tokenizer_name": "bert-base-uncased",
        "tokenizer_revision": None,
        "input_template_version": "structural-template-v1",
        "feature_schema_version": "structural-v1",
        "feature_dim": 70,
        "feature_dtype": "float64",
        "max_length": 256,
        "granularity": "two_scale",
        "local_window_size": 5,
        "local_stride": 2,
        "score_output": "probability",
        "positive_class_index": 1,
        "extra_note": "preserved",
    }
    data.update(overrides)
    return data


def test_metadata_from_dict_preserves_unknown_fields() -> None:
    metadata = metadata_from_dict(metadata_dict())

    assert metadata.extra == {"extra_note": "preserved"}


def test_metadata_requires_supported_artifact_schema_version() -> None:
    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict(metadata_dict(artifact_schema_version="other"))


def test_validate_scorer_metadata_accepts_matching_metadata() -> None:
    metadata = metadata_from_dict(metadata_dict())

    validate_scorer_metadata(
        metadata,
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        task_type="answer_confidence",
        max_length=256,
        input_template_version="structural-template-v1",
        feature_schema_version="structural-v1",
        feature_dim=70,
    )


def test_validate_scorer_metadata_rejects_mismatch() -> None:
    metadata = metadata_from_dict(metadata_dict(encoder_name="other"))

    with pytest.raises(ConfidenceArtifactError):
        validate_scorer_metadata(
            metadata,
            encoder_name="bert-base-uncased",
            encoder_revision=None,
            tokenizer_name="bert-base-uncased",
            tokenizer_revision=None,
            task_type="answer_confidence",
            max_length=256,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
        )


class FakeProbaModel:
    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        assert len(rows) == 1
        return [[0.2, 0.8]]


class BadShapeProbaModel:
    def predict_proba(self, rows: list[list[float]]) -> list[float]:
        del rows
        return [0.2, 0.8]


class FakePredictModel:
    def predict(self, rows: list[list[float]]) -> list[float]:
        assert len(rows) == 1
        return [0.6]


def test_lightgbm_scorer_uses_predict_proba_positive_class() -> None:
    scorer = LightGBMScorer(
        model=FakeProbaModel(),
        metadata=metadata_from_dict(metadata_dict()),
    )

    assert scorer.predict_confidence([0.0] * 70) == 0.8


def test_lightgbm_scorer_accepts_probability_predict() -> None:
    scorer = LightGBMScorer(
        model=FakePredictModel(),
        metadata=metadata_from_dict(metadata_dict()),
    )

    assert scorer.predict_confidence([0.0] * 70) == 0.6


def test_lightgbm_scorer_accepts_numpy_outputs() -> None:
    import numpy as np

    class NumpyProbaModel:
        def predict_proba(self, rows: list[list[float]]) -> object:
            assert len(rows) == 1
            return np.asarray([[0.1, 0.9]], dtype=np.float64)

    scorer = LightGBMScorer(
        model=NumpyProbaModel(),
        metadata=metadata_from_dict(metadata_dict()),
    )

    assert scorer.predict_confidence([0.0] * 70) == 0.9


def test_lightgbm_scorer_rejects_bad_predict_proba_shape() -> None:
    scorer = LightGBMScorer(
        model=BadShapeProbaModel(),
        metadata=metadata_from_dict(metadata_dict()),
    )

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 70)


def test_booster_metadata_json_must_be_json_serializable(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict()), encoding="utf-8")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert data["artifact_schema_version"] == "structural-artifact-v1"
```

- [x] **Step 2: Run failing scorer tests**

Run:

```bash
uv run pytest tests/test_confidence_scorer.py -q
```

Expected: FAIL because `_scorer.py` still contains stubs.

- [x] **Step 3: Implement scorer metadata and model wrapper**

Replace `src/ranksmith/confidence/_scorer.py` with:

```python
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._errors import ConfidenceArtifactError
from ranksmith.confidence._types import ScorerMetadata, TaskType

ARTIFACT_SCHEMA_VERSION = "structural-artifact-v1"


def metadata_from_dict(data: Mapping[str, Any]) -> ScorerMetadata:
    _ensure_json_serializable(data)
    known = {field.name for field in fields(ScorerMetadata)}
    constructor_names = known - {"extra"}
    missing = [name for name in constructor_names if name not in data]
    if missing:
        raise ConfidenceArtifactError(f"missing scorer metadata fields: {missing}")
    extra = {key: value for key, value in data.items() if key not in constructor_names}
    metadata = ScorerMetadata(
        artifact_schema_version=str(data["artifact_schema_version"]),
        scorer_type=str(data["scorer_type"]),
        task_type=_task_type(data["task_type"]),
        encoder_name=str(data["encoder_name"]),
        encoder_revision=_optional_str(data["encoder_revision"]),
        tokenizer_name=str(data["tokenizer_name"]),
        tokenizer_revision=_optional_str(data["tokenizer_revision"]),
        input_template_version=str(data["input_template_version"]),
        feature_schema_version=str(data["feature_schema_version"]),
        feature_dim=int(data["feature_dim"]),
        feature_dtype=str(data["feature_dtype"]),
        max_length=int(data["max_length"]),
        granularity=str(data["granularity"]),
        local_window_size=int(data["local_window_size"]),
        local_stride=int(data["local_stride"]),
        score_output="probability",
        positive_class_index=int(data["positive_class_index"]),
        extra=extra,
    )
    if metadata.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ConfidenceArtifactError("unsupported artifact_schema_version")
    if metadata.score_output != "probability":
        raise ConfidenceArtifactError("score_output must be probability")
    return metadata


def validate_scorer_metadata(
    metadata: ScorerMetadata,
    *,
    encoder_name: str,
    encoder_revision: str | None,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    task_type: TaskType,
    max_length: int,
    input_template_version: str,
    feature_schema_version: str,
    feature_dim: int,
) -> None:
    expected = {
        "encoder_name": encoder_name,
        "encoder_revision": encoder_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "task_type": task_type,
        "max_length": max_length,
        "input_template_version": input_template_version,
        "feature_schema_version": feature_schema_version,
        "feature_dim": feature_dim,
        "feature_dtype": "float64",
        "granularity": "two_scale",
        "local_window_size": 5,
        "local_stride": 2,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    for name, value in expected.items():
        if getattr(metadata, name) != value:
            raise ConfidenceArtifactError(f"scorer metadata mismatch: {name}")


class LightGBMScorer:
    def __init__(self, *, model: object, metadata: ScorerMetadata) -> None:
        self.model = model
        self.metadata = metadata

    def predict_confidence(self, features: Sequence[float]) -> float:
        if len(features) != self.metadata.feature_dim:
            raise ConfidenceArtifactError("feature length does not match metadata")
        rows = [list(features)]
        if hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(rows)
            score = _extract_predict_proba_score(
                raw,
                positive_class_index=self.metadata.positive_class_index,
            )
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(rows)
            score = _extract_predict_score(raw)
        else:
            raise ConfidenceArtifactError(
                "scorer model must provide predict_proba() or predict()"
            )
        _validate_probability(score)
        return score


def load_lightgbm_scorer(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> LightGBMScorer:
    if metadata_path is None:
        joblib = import_optional_dependency("joblib")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ConfidenceArtifactError("joblib artifact must be a dict")
        if "model" not in artifact or "metadata" not in artifact:
            raise ConfidenceArtifactError("joblib artifact requires model and metadata")
        return LightGBMScorer(
            model=artifact["model"],
            metadata=metadata_from_dict(artifact["metadata"]),
        )

    lightgbm = import_optional_dependency("lightgbm")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return LightGBMScorer(
        model=lightgbm.Booster(model_file=str(path)),
        metadata=metadata_from_dict(metadata),
    )


def _task_type(value: object) -> TaskType:
    if value in {"answer_confidence", "judgment_confidence"}:
        return value  # type: ignore[return-value]
    raise ConfidenceArtifactError("invalid task_type")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _ensure_json_serializable(data: Mapping[str, Any]) -> None:
    try:
        json.dumps(data)
    except TypeError as exc:
        raise ConfidenceArtifactError("metadata must be JSON-serializable") from exc


def _extract_predict_proba_score(
    raw: object,
    *,
    positive_class_index: int,
) -> float:
    np = import_optional_dependency("numpy")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ConfidenceArtifactError("predict_proba output must have shape (1, n)")
    if positive_class_index < 0 or positive_class_index >= array.shape[1]:
        raise ConfidenceArtifactError("positive_class_index is out of range")
    return float(array[0, positive_class_index])


def _extract_predict_score(raw: object) -> float:
    np = import_optional_dependency("numpy")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise ConfidenceArtifactError("predict output must be scalar or length 1")


def _validate_probability(score: float) -> None:
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ConfidenceArtifactError("confidence score must be a probability")
```

- [x] **Step 4: Run scorer tests**

Run:

```bash
uv run pytest tests/test_confidence_scorer.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ranksmith/confidence/_scorer.py tests/test_confidence_scorer.py
git commit -m "feat: add confidence scorer loaders"
```

---

### Task 6: Structural Confidence Estimator

**Files:**
- Modify: `src/ranksmith/confidence/_structural.py`
- Test: `tests/test_confidence_estimator.py`

- [x] **Step 1: Write failing estimator tests**

Create `tests/test_confidence_estimator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ranksmith.confidence import (
    AnswerConfidenceInput,
    ConfidenceArtifactError,
    ConfidenceInputError,
    JudgmentConfidenceInput,
    ScorerMetadata,
    StructuralConfidenceEstimator,
)


@dataclass
class FakeEncoder:
    encoder_name: str = "bert-base-uncased"
    encoder_revision: str | None = None
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_revision: str | None = None
    max_length: int = 256

    def encode(self, text: str) -> tuple[object, object]:
        assert text
        hidden = np.arange(8 * 4, dtype=np.float64).reshape(8, 4)
        mask = np.ones(8, dtype=np.int64)
        return hidden, mask


class FakeScorer:
    def __init__(self, *, score: float = 0.75) -> None:
        self.metadata = ScorerMetadata(
            artifact_schema_version="structural-artifact-v1",
            scorer_type="fake",
            task_type="answer_confidence",
            encoder_name="bert-base-uncased",
            encoder_revision=None,
            tokenizer_name="bert-base-uncased",
            tokenizer_revision=None,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
            feature_dtype="float64",
            max_length=256,
            granularity="two_scale",
            local_window_size=5,
            local_stride=2,
            score_output="probability",
            positive_class_index=1,
        )
        self.score = score

    def predict_confidence(self, features: list[float]) -> float:
        assert len(features) == 70
        return self.score


def test_estimator_scores_answer_input() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        task_type="answer_confidence",
    )

    result = estimator.score(
        AnswerConfidenceInput(context="context", answer="answer")
    )

    assert result.score == 0.75
    assert result.task_type == "answer_confidence"
    assert result.feature_schema_version == "structural-v1"
    assert result.metadata["encoder_name"] == "bert-base-uncased"
    assert "hf_token" not in result.metadata


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


def test_estimator_rejects_out_of_range_score() -> None:
    estimator = StructuralConfidenceEstimator(
        encoder=FakeEncoder(),
        scorer=FakeScorer(score=2.0),
        task_type="answer_confidence",
    )

    with pytest.raises(ConfidenceArtifactError):
        estimator.score(
            AnswerConfidenceInput(context="context", answer="answer")
        )
```

- [x] **Step 2: Run failing estimator tests**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py -q
```

Expected: FAIL because `_structural.py` is still a placeholder.

- [x] **Step 3: Implement estimator**

Replace `src/ranksmith/confidence/_structural.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ranksmith.confidence._encoder import FrozenAutoEncoder
from ranksmith.confidence._errors import (
    ConfidenceArtifactError,
    ConfidenceInputError,
)
from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    GRANULARITY,
    LOCAL_STRIDE,
    LOCAL_WINDOW_SIZE,
    extract_structural_features,
)
from ranksmith.confidence._scorer import validate_scorer_metadata
from ranksmith.confidence._templates import (
    INPUT_TEMPLATE_VERSION,
    format_confidence_input,
)
from ranksmith.confidence._types import (
    StructuralConfidenceInput,
    StructuralConfidenceResult,
    StructuralConfidenceScorer,
    TaskType,
)


@dataclass(frozen=True)
class StructuralConfidenceEstimator:
    encoder: Any
    scorer: StructuralConfidenceScorer
    task_type: TaskType

    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_name: str = "bert-base-uncased",
        encoder_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        device: str = "cpu",
        scorer: StructuralConfidenceScorer,
        task_type: TaskType,
        max_length: int = 256,
        allow_truncation: bool = False,
    ) -> "StructuralConfidenceEstimator":
        encoder = FrozenAutoEncoder.from_pretrained(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            max_length=max_length,
            allow_truncation=allow_truncation,
        )
        validate_scorer_metadata(
            scorer.metadata,
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name or encoder_name,
            tokenizer_revision=tokenizer_revision,
            task_type=task_type,
            max_length=max_length,
            input_template_version=INPUT_TEMPLATE_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_dim=FEATURE_DIM,
        )
        return cls(encoder=encoder, scorer=scorer, task_type=task_type)

    def score(self, item: StructuralConfidenceInput) -> StructuralConfidenceResult:
        text = format_confidence_input(self.task_type, item)
        hidden_states, mask = self.encoder.encode(text)
        features = extract_structural_features(
            hidden_states,
            mask,
            max_length=self.encoder.max_length,
        )
        score = self.scorer.predict_confidence(features)
        _validate_score(score)
        return StructuralConfidenceResult(
            score=score,
            task_type=self.task_type,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            metadata={
                "encoder_name": self.encoder.encoder_name,
                "encoder_revision": self.encoder.encoder_revision,
                "tokenizer_name": self.encoder.tokenizer_name,
                "tokenizer_revision": self.encoder.tokenizer_revision,
                "max_length": self.encoder.max_length,
                "feature_dim": len(features),
                "feature_dtype": FEATURE_DTYPE,
                "granularity": GRANULARITY,
                "local_window_size": LOCAL_WINDOW_SIZE,
                "local_stride": LOCAL_STRIDE,
                "input_template_version": INPUT_TEMPLATE_VERSION,
                "scorer_type": self.scorer.metadata.scorer_type,
                "artifact_schema_version": (
                    self.scorer.metadata.artifact_schema_version
                ),
            },
        )


def _validate_score(score: float) -> None:
    if not isinstance(score, (int, float)):
        raise ConfidenceArtifactError("confidence score must be numeric")
    if score < 0.0 or score > 1.0:
        raise ConfidenceArtifactError("confidence score must be in [0, 1]")
```

- [x] **Step 4: Run estimator tests**

Run:

```bash
uv run pytest tests/test_confidence_estimator.py -q
```

Expected: PASS.

- [x] **Step 5: Run all confidence tests**

Run:

```bash
uv run pytest tests/test_confidence_*.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/ranksmith/confidence/_structural.py tests/test_confidence_estimator.py
git commit -m "feat: add structural confidence estimator"
```

---

### Task 7: Optional Extra And Docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/wiki/02_architecture.md`
- Create: `docs/wiki/references/structural_confidence.md`
- Modify: `docs/wiki/04_references_index.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Test: `tests/test_confidence_api_scope.py`

- [ ] **Step 1: Add optional extra**

Modify `pyproject.toml` after `[project.urls]`:

```toml
[project.optional-dependencies]
confidence = [
    "torch>=2.0",
    "transformers>=4.40",
    "numpy>=1.24",
    "scipy>=1.10",
    "joblib>=1.3",
    "lightgbm>=4.0",
]
```

- [x] **Step 2: Update architecture wiki**

In `docs/wiki/02_architecture.md`, add a `Confidence` section after `Algorithm`:

```markdown
## Confidence
`ranksmith.confidence`는 reranking Strategy가 아니라 closed model output confidence를 계산하는 utility layer다.

현재 범위:
- frozen HuggingFace encoder 기반 structural feature extraction
- pre-trained scorer artifact 기반 single-item confidence inference
- root import가 아닌 `ranksmith.confidence` submodule export

제외:
- training pipeline
- semantic feature fusion
- batch/async inference
- reranking Strategy
```

- [x] **Step 3: Add Trust reference summary**

Create `docs/wiki/references/structural_confidence.md`:

```markdown
# Reference: Trust in One Round

## Source
- Paper: Trust in One Round: Confidence Estimation for Large Language Models via Structural Signals
- Blog:
- Repo:
- License:

## 적용 영역
- `ranksmith.confidence`
- closed model output confidence estimation

## 핵심 메커니즘
closed model의 hidden state를 직접 보지 않고, `context + answer` 또는 ranksmith용 `query + document + judgment`를 frozen encoder에 넣어 token-level hidden trajectory를 만든다. 이 trajectory를 spectral stability, local variation, shape coherence feature로 요약하고, 학습된 lightweight scorer가 confidence probability를 추정한다.

## ranksmith 매핑
- Strategy: 추가하지 않음
- Algorithm: 추가하지 않음
- Public API 영향: `ranksmith.confidence` submodule 추가
- Error 동작: confidence-specific error로 fast fail
- 추가할 테스트: feature schema, scorer artifact validation, HF token handling, numeric stability

## 현재 설계와 충돌
- ranksmith core는 training-free reranking을 지향한다.
- Trust 방식은 scorer 학습이 필요하므로 Phase 1은 inference-only로 제한한다.
- training pipeline은 별도 스펙으로 분리한다.

## Do Not Copy
- 외부 reference 구현 코드를 복사하지 않는다.
- 논문이 명시하지 않은 feature 세부 계산은 ranksmith `structural-v1` schema로 고정한다.

## 부족한 정보
- Phase 2 training dataset schema
- Phase 2 artifact save/export helper
```

- [x] **Step 4: Update reference index**

In `docs/wiki/04_references_index.md`, move Trust from pending to registered:

```markdown
- [Trust in One Round: Confidence Estimation for Large Language Models via Structural Signals](references/structural_confidence.md): Paper / Black-box confidence, proxy hidden-state trajectory / 요약 완료, Phase 1 inference spec 작성
```

Remove the pending Trust PDF line. Keep the PDF file in `docs/wiki/references/`.

- [x] **Step 5: Update README files without performance claims**

Add this minimal English section to `README.md`:

```markdown
## Structural Confidence

`ranksmith.confidence` provides single-item confidence inference for closed-model outputs using a frozen HuggingFace encoder and a pre-trained scorer artifact.

Install optional dependencies:

```bash
pip install "ranksmith[confidence]"
```

```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    StructuralConfidenceEstimator,
    load_lightgbm_scorer,
)

scorer = load_lightgbm_scorer("structural-confidence.joblib")
estimator = StructuralConfidenceEstimator.from_pretrained(
    scorer=scorer,
    task_type="answer_confidence",
)

result = estimator.score(
    AnswerConfidenceInput(context="...", answer="...")
)
```

This module does not train a scorer and does not add a reranking Strategy.
```

Add the matching Korean section to `README.ko.md`.

- [ ] **Step 6: Run docs/API tests**

Run:

```bash
uv run pytest tests/test_confidence_api_scope.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docs/wiki/02_architecture.md docs/wiki/references/structural_confidence.md docs/wiki/04_references_index.md README.md README.ko.md tests/test_confidence_api_scope.py
git commit -m "docs: document structural confidence module"
```

---

### Task 8: Verification And Spec Closeout

**Files:**
- Modify: `docs/specs/spec_structural_confidence.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_confidence_*.py -q
```

Expected: all confidence tests PASS.

- [ ] **Step 2: Run full verification**

Run:

```bash
./scripts/verify.sh
```

Expected: full verification PASS.

- [ ] **Step 3: Mark spec completed**

Modify `docs/specs/spec_structural_confidence.md`:

```markdown
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`
```

Mark every completed task checklist item in the spec with `[x]`.

- [ ] **Step 4: Commit closeout**

```bash
git add docs/specs/spec_structural_confidence.md
git commit -m "docs: complete structural confidence spec"
```

---

## Self-Review Checklist

- [ ] Spec coverage: every Phase 1 scope item in `docs/specs/spec_structural_confidence.md` maps to at least one task above.
- [ ] No placeholders: no task uses `TBD`, `TODO`, `implement later`, or vague “add tests” language.
- [ ] Type consistency: `AnswerConfidenceInput`, `JudgmentConfidenceInput`, `ScorerMetadata`, `StructuralConfidenceEstimator`, and `load_lightgbm_scorer` names match across tasks.
- [ ] Scope containment: no training, semantic fusion, batch, async, root export, non-CPU, or artifact save helper is implemented.
- [ ] Verification: focused confidence tests and `./scripts/verify.sh` run before marking complete.
