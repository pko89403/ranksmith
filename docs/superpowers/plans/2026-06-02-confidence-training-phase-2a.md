# Confidence Training Phase 2A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal supervised training pipeline that turns task-specific canonical JSONL into a Phase 1-compatible structural confidence scorer artifact.

**Architecture:** Add `ranksmith.confidence_training` as a separate submodule so training-only dependencies do not affect inference-only users. The pipeline validates canonical JSONL, performs deterministic splits, reuses Phase 1 frozen encoder and structural feature extraction, trains LightGBM, calibrates with validation data, reports metrics, and exports a joblib artifact that Phase 1 can load.

**Tech Stack:** Python dataclasses, JSONL, NumPy, LightGBM, scikit-learn, joblib, existing `ranksmith.confidence` helpers, pytest, mypy, ruff.

---

### Task 1: Dependency And API Boundary

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ranksmith/confidence_training/__init__.py`
- Create: `src/ranksmith/confidence_training/_errors.py`
- Create: `src/ranksmith/confidence_training/_types.py`
- Test: `tests/test_confidence_training_api.py`

- [ ] Write failing tests that `import ranksmith.confidence_training` succeeds without root exports and exposes only the approved training API.
- [ ] Add `confidence-train` optional extra with the current `confidence` dependencies plus `scikit-learn`.
- [ ] Implement training error classes and frozen config/result dataclasses.
- [ ] Run `uv run pytest tests/test_confidence_training_api.py -q`.
- [ ] Commit.

### Task 2: Canonical Dataset And Split

**Files:**
- Create: `src/ranksmith/confidence_training/_dataset.py`
- Create: `src/ranksmith/confidence_training/_split.py`
- Test: `tests/test_confidence_training_dataset.py`
- Test: `tests/test_confidence_training_split.py`

- [ ] Write failing tests for valid `answer_confidence` and `judgment_confidence` JSONL loading.
- [ ] Write failing tests for missing fields, invalid labels, duplicate ids, empty text, unsupported task type, and ambiguous task fields.
- [ ] Write failing tests for deterministic 80/10/10 split, `group_id` grouping, split class collapse, and minimum sample violations.
- [ ] Implement canonical sample parsing and split validation with fast-fail errors.
- [ ] Run dataset/split tests.
- [ ] Commit.

### Task 3: Feature Extraction Runner

**Files:**
- Create: `src/ranksmith/confidence_training/_features.py`
- Test: `tests/test_confidence_training_features.py`

- [ ] Write failing tests using a fake encoder that returns deterministic hidden states.
- [ ] Verify feature rows contain `id`, `task_type`, `label`, `features`, `feature_schema_version`, and metadata.
- [ ] Verify feature length is 70 and NaN/Inf features fail.
- [ ] Implement feature extraction runner using Phase 1 templates and structural feature extractor.
- [ ] Run feature tests.
- [ ] Commit.

### Task 4: Training, Calibration, Report

**Files:**
- Create: `src/ranksmith/confidence_training/_train.py`
- Create: `src/ranksmith/confidence_training/_calibration.py`
- Create: `src/ranksmith/confidence_training/_report.py`
- Test: `tests/test_confidence_training_train.py`

- [ ] Write failing tests for LightGBM training on toy feature rows.
- [ ] Write failing tests for validation-only sigmoid calibration and test-only final metrics.
- [ ] Write failing tests for one-class metric failure and too-small calibration data.
- [ ] Implement training, calibration wrapper, and report generation.
- [ ] Run training tests.
- [ ] Commit.

### Task 5: Artifact Export And End-To-End API

**Files:**
- Create: `src/ranksmith/confidence_training/_artifact.py`
- Modify: `src/ranksmith/confidence_training/__init__.py`
- Test: `tests/test_confidence_training_artifact.py`
- Test: `tests/test_confidence_training_metadata.py`

- [ ] Write failing tests that exported joblib artifacts load through Phase 1 `load_lightgbm_scorer()`.
- [ ] Write failing tests that all Phase 1 `ScorerMetadata` fields are present and incompatible metadata fails before export.
- [ ] Implement `train_confidence_scorer(config)` orchestration and artifact export.
- [ ] Smoke test loaded artifact with `StructuralConfidenceEstimator.score()` using a fake scorer/encoder path where network is not required.
- [ ] Run artifact/metadata tests.
- [ ] Commit.

### Task 6: Docs, Ignore Rules, And Verification

**Files:**
- Modify: `.gitignore`
- Modify: `docs/wiki/02_architecture.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/specs/spec_confidence_training_pipeline.md`

- [ ] Add generated training outputs to `.gitignore`.
- [ ] Document `ranksmith.confidence_training` as a training utility layer, not Strategy/Algorithm.
- [ ] Add minimal README/README.ko note for `confidence-train` extra without benchmark claims.
- [ ] Mark completed checklist items in the spec.
- [ ] Run `UV_NATIVE_TLS=true ./scripts/verify.sh`.
- [ ] Commit.

---

### Self-Review

- Spec coverage: Phase 2A scope is covered; adapters, CLI, semantic labels, and reranking Strategy remain excluded.
- Placeholder scan: no deferred implementation placeholders are used in task steps.
- Type consistency: public API remains `ConfidenceTrainingConfig` and `train_confidence_scorer`; root exports are excluded.
