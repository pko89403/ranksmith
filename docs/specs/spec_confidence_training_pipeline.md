# Spec: Confidence Training Pipeline

## 1. 개요 (Overview)
- **작업 목적**: `ranksmith.confidence` Phase 1 inference module에 꽂을 수 있는 compatible scorer artifact를 생성한다.
- **Reference**:
  - `docs/specs/spec_structural_confidence.md`
  - `docs/wiki/references/structural_confidence.md`
- **상태**: `[x] Draft` | `[ ] In Progress` | `[ ] Completed`

Phase 1은 추론 core만 제공한다.
Phase 2 전체 방향은 실제 confidence score를 만들기 위한 **training pipeline**이다.
이번 구현 단위는 범위를 줄인 **Phase 2A: trainable artifact 생성 최소 경로**다.

지원 task는 두 개다.
- `answer_confidence`
- `judgment_confidence`

두 task는 label source와 평가 의미가 다르므로 dataset과 artifact를 분리한다.

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 범위
포함한다:
- task별 canonical JSONL dataset schema.
- canonical JSONL validation/loading.
- train/valid/test split.
- frozen HuggingFace encoder 기반 structural feature extraction.
- `structural-v1` 70차원 feature matrix 생성.
- LightGBM binary classifier training.
- validation-set calibration.
- metric report 생성.
- task별 scorer artifact save/export.
- artifact load smoke test.

포함하지 않는다:
- reranking Strategy 추가.
- online learning.
- closed model 호출 자동 생성.
- answer generation pipeline.
- 외부 benchmark/source adapter 구현.
- qrel 생성 자동화.
- CLI.
- benchmark 수치 README 반영.
- Phase 1 public inference API 변경.

### 입력 (Inputs)

#### Canonical JSONL: `answer_confidence`
각 line은 하나의 supervised sample이다.

필수 필드:
- `id: str`
- `context: str`
- `answer: str`
- `label: int`
  - `1`: answer가 gold/reference answer와 맞음
  - `0`: answer가 틀림

선택 필드:
- `gold_answer: str | list[str]`
- `source: str`
- `group_id: str`
- `metadata: dict[str, object]`

Phase 2A는 `label`을 생성하지 않는다.
`label`은 입력 JSONL에 이미 있어야 하며, `gold_answer`는 provenance/debug용 metadata다.

#### Canonical JSONL: `judgment_confidence`
각 line은 하나의 supervised sample이다.

필수 필드:
- `id: str`
- `query: str`
- `document: str`
- `judgment: str`
- `label: int`
  - `1`: judgment가 relevance label과 일치
  - `0`: judgment가 relevance label과 불일치

선택 필드:
- `relevance_label: int | float | bool`
- `source: str`
- `group_id: str`
- `metadata: dict[str, object]`

Phase 2A는 `judgment`와 `relevance_label`로 label을 생성하지 않는다.
`label`은 입력 JSONL에 이미 있어야 하며, `relevance_label`은 provenance/debug용 metadata다.

### 후속 Adapter 후보

Adapter는 Phase 2A 구현 범위가 아니다.
아래 항목은 후속 Phase 2B 후보로만 둔다.

#### IR adapter 후보
지원 대상:
- BEIR/MTEB style `queries`, `corpus`, `qrels`
- 필요 시 fixed candidate run file

출력:
- `judgment_confidence` canonical JSONL

주의:
- adapter는 judgment 문장을 임의 생성하지 않는다.
- judgment source가 없는 경우 구현을 멈추고 사용자 결정이 필요하다.
- 예: `"direct evidence"`, `"no evidence"` 같은 judgment label text를 쓰려면 mapping table을 명시해야 한다.

#### QA adapter 후보
지원 대상:
- SQuAD-style `context`, `question`, `answers`
- answer prediction과 gold/reference answer가 함께 있는 source

출력:
- `answer_confidence` canonical JSONL

주의:
- answer correctness는 exact match 또는 semantic match 중 하나를 label schema로 고정해야 한다.
- Phase 2B adapter 후보의 기본값은 exact/normalized match다.
- semantic match label은 별도 evaluator spec 없이는 구현하지 않는다.

### 출력 (Outputs)

Training run output directory:
```text
confidence-runs/<run_id>/
  dataset_manifest.json
  split_manifest.json
  features_train.jsonl
  features_valid.jsonl
  features_test.jsonl
  model.joblib
  metadata.json
  report.json
  report.md
```

Exported scorer artifact:
```text
artifacts/
  answer_confidence.joblib
  judgment_confidence.joblib
```

Artifact는 Phase 1 `load_lightgbm_scorer()`가 읽을 수 있어야 한다.

### 제약 사항
- encoder는 학습하지 않는다.
- feature schema는 `structural-v1`만 지원한다.
- task별 artifact를 통합하지 않는다.
- task별 calibration을 분리한다.
- label은 반드시 binary `0` 또는 `1`이다.
- label이 없거나 모호하면 fast fail 한다.
- hidden truncation을 숨기지 않는다.
- `allow_truncation=True`는 명시적으로만 허용한다.
- benchmark/README 성능 claim은 실제 summary artifact 없이는 금지한다.
- raw dataset, raw feature dump, large model artifact는 기본 커밋 대상이 아니다.

## 3. 상세 설계 (Architecture & Design)

### 모듈 구조
```text
src/ranksmith/confidence_training/
  __init__.py
  _types.py          # dataset/config/result dataclass
  _errors.py         # training-specific errors
  _dataset.py        # canonical JSONL validation/loading
  _split.py          # deterministic split
  _features.py       # feature extraction runner
  _train.py          # LightGBM training
  _calibration.py    # probability calibration
  _report.py         # metrics/report generation
  _artifact.py       # scorer artifact export
```

이 module은 Phase 1 `ranksmith.confidence`와 분리한다.
Inference-only 사용자가 training dependency를 설치하지 않아도 되게 하기 위함이다.

### Optional Extra
`pyproject.toml`에 별도 extra를 둔다.

```toml
[project.optional-dependencies]
confidence-train = [
  "joblib>=1.3",
  "lightgbm>=4.0",
  "numpy>=1.24",
  "scikit-learn>=1.4",
  "torch>=2.0",
  "transformers>=4.36",
]
```

`confidence-train`은 `confidence` extra를 자기 참조하지 않고 실제 dependency를 명시한다.
dependency 목록은 현재 구현된 Phase 1 `confidence` extra에 `scikit-learn`만 더한 형태다.

필요 시 `pandas`, `datasets`, `pyarrow`는 별도 검토 후 추가한다.
기본 설계는 JSONL + standard library 중심으로 둔다.

### Data Flow
```text
external source
  -> user-provided canonical JSONL
  -> deterministic split
  -> frozen encoder
  -> structural-v1 feature extraction
  -> LightGBM training
  -> calibration
  -> metrics/report
  -> scorer artifact export
  -> load smoke with ranksmith.confidence
```

### Label Schema

#### `answer_confidence`
Phase 2A는 supervised `label`을 그대로 학습한다.
`gold_answer` 기반 exact/normalized match label 생성은 구현하지 않는다.

후속 adapter/evaluator에서 label을 생성할 때의 후보 기준:
- trim whitespace
- lowercase
- collapse internal whitespace
- optional surrounding punctuation removal

Semantic equivalence label은 Phase 2A에서 제외한다.

#### `judgment_confidence`
Phase 2A는 supervised `label`을 그대로 학습한다.
`judgment` text mapping과 `relevance_label` threshold 기반 label 생성은 구현하지 않는다.

후속 adapter/evaluator에서 label을 생성할 때는 다음 값을 명시해야 한다.
- `judgment` text를 positive/negative로 바꾸는 mapping
- `relevance_label`을 positive/negative로 바꾸는 threshold

mapping에 없는 judgment는 후속 adapter에서도 실패해야 한다.
조용히 negative로 처리하지 않는다.

### Split
기본 split:
- train: 80%
- valid: 10%
- test: 10%
- seed: required

규칙:
- 같은 `id`가 중복되면 실패한다.
- 같은 source group을 split 단위로 묶을 수 있도록 optional `group_id`를 지원한다.
- split manifest에 seed, counts, task_type, source hash를 기록한다.
- 전체 dataset과 train/valid/test 각각에 positive/negative class가 모두 있어야 한다.
- 기본 최소 sample 수는 전체 30개, split별 2개, split별 class당 1개다.
- 이 기준을 만족하지 못하면 training/calibration/report를 진행하지 않고 실패한다.

### Feature Extraction
Phase 1의 다음 기능을 재사용한다.
- `FrozenAutoEncoder`
- `format_confidence_input`
- `extract_structural_features`

Feature row:
```json
{
  "id": "sample-id",
  "task_type": "answer_confidence",
  "label": 1,
  "features": [0.0, ...],
  "feature_schema_version": "structural-v1",
  "metadata": {}
}
```

Feature file은 JSONL로 저장한다.
Parquet은 Phase 2 기본 구현에서 제외한다.

### Training
기본 model:
- LightGBM binary classifier

입력:
- train feature matrix

출력:
- uncalibrated classifier
- calibrated scorer wrapper

기본 hyperparameter는 config에 명시한다.
자동 tuning은 제외한다.
Phase 2A는 early stopping을 사용하지 않는다.
validation split은 calibration과 calibration 전 metric 확인에 사용한다.

### Calibration
기본 calibration:
- validation split 기반 sigmoid calibration
- `scikit-learn`의 calibration utility 또는 equivalent calibrated wrapper 사용

규칙:
- calibration은 task별로 수행한다.
- calibration 전후 metrics를 모두 report에 기록한다.
- calibration data가 너무 작으면 fast fail 한다.
- test split은 최종 report에만 사용한다.
- test split은 training, hyperparameter 선택, calibration에 사용하지 않는다.

### Metrics
필수 report metric:
- sample count
- positive/negative label count
- accuracy
- ROC AUC
- average precision
- Brier score
- log loss
- calibration error summary

주의:
- 이 metrics는 confidence scorer 평가다.
- reranking benchmark로 표현하지 않는다.

### Artifact Metadata
`metadata.json`과 joblib artifact metadata는 Phase 1 `ScorerMetadata`와 호환되어야 한다.

Phase 2A export는 다음 Phase 1 필드를 반드시 생성한다.
- `artifact_schema_version`
- `scorer_type`
- `task_type`
- `encoder_name`
- `encoder_revision`
- `tokenizer_name`
- `tokenizer_revision`
- `input_template_version`
- `feature_schema_version`
- `feature_dim`
- `feature_dtype`
- `max_length`
- `granularity`
- `local_window_size`
- `local_stride`
- `score_output`
- `positive_class_index`

필수 추가 training metadata:
- `label_schema_version`
- `dataset_manifest_hash`
- `split_seed`
- `train_count`
- `valid_count`
- `test_count`
- `calibration_method`
- `training_config_hash`
- `created_at`

Unknown metadata는 Phase 1 loader의 `extra`에 보존된다.

Phase 1 loader가 검증하는 값과 맞지 않으면 export 단계에서 실패한다.
loader 실패를 smoke test에서 뒤늦게 발견하는 방식으로 넘기지 않는다.

### Training Config
`ConfidenceTrainingConfig` 최소 필드:
- `task_type: Literal["answer_confidence", "judgment_confidence"]`
- `dataset_path: str | Path`
- `output_dir: str | Path`
- `export_path: str | Path`
- `encoder_name: str`
- `encoder_revision: str | None`
- `tokenizer_name: str | None`
- `tokenizer_revision: str | None`
- `cache_dir: str | None`
- `local_files_only: bool`
- `max_length: int`
- `allow_truncation: bool`
- `seed: int`
- `train_ratio: float`
- `valid_ratio: float`
- `test_ratio: float`
- `calibration_method: Literal["sigmoid"]`

기본값:
- `encoder_name = "bert-base-uncased"`
- `tokenizer_name = None`
- `max_length = 256`
- `allow_truncation = False`
- `local_files_only = False`
- `train_ratio = 0.8`
- `valid_ratio = 0.1`
- `test_ratio = 0.1`
- `calibration_method = "sigmoid"`

`seed`는 명시 입력을 권장한다.
기본 seed를 제공하더라도 report와 split manifest에 반드시 기록한다.

### Public API Scope
Phase 2 public API는 기본적으로 submodule 아래에 둔다.

```python
from ranksmith.confidence_training import (
    ConfidenceTrainingConfig,
    train_confidence_scorer,
)
```

Root export는 추가하지 않는다.

CLI는 별도 검토한다.
기본 구현은 Python API 우선이다.
CLI를 추가하려면 이 spec 승인 후 별도 scope로 확정한다.

### Phase 2B 후보
Phase 2A 완료 후 별도 spec 또는 이 spec 개정으로 다룬다.
- BEIR/MTEB-style IR adapter
- SQuAD-style QA adapter
- CLI
- semantic match evaluator
- generated artifact store/release workflow

## 4. 재사용 및 모듈화 (Reusability & Modularization)

재사용:
- Phase 1 input dataclass
- Phase 1 template formatter
- Phase 1 encoder wrapper
- Phase 1 structural feature extractor
- Phase 1 scorer loader contract

분리:
- training-only dependency는 `confidence_training`으로 격리한다.
- report generation은 training model과 분리한다.

## 5. 에러 핸들링 (Error Handling)

새 error:
- `ConfidenceTrainingError`
- `ConfidenceDatasetError`
- `ConfidenceLabelError`
- `ConfidenceTrainingConfigError`

주요 실패:
- canonical JSONL 필수 필드 누락
- label이 `0/1`이 아님
- duplicate id
- split 후 class가 한쪽만 존재
- split/sample 최소 기준 미달
- feature vector length mismatch
- NaN/Inf feature
- calibration sample 부족
- artifact metadata 불일치

모두 fast fail 한다.

## 6. 저장 및 보안 정책 (Storage & Security)

기본적으로 git에 포함하지 않는다:
- raw dataset
- generated feature JSONL
- `confidence-runs/`
- exported `.joblib` artifact

포함 가능한 파일:
- 작은 fixture dataset
- test용 toy artifact
- report schema fixture

HuggingFace token:
- public model은 token 없이 동작해야 한다.
- private/gated model은 `HF_TOKEN` 또는 HuggingFace cache/login을 사용한다.
- token 값은 config, report, artifact metadata에 저장하지 않는다.
- metadata에는 token 사용 여부도 기록하지 않는다.

대용량 artifact는 별도 artifact store 또는 release asset 후보로만 둔다.

## 7. 테스트 계획 (Test Plan)

### 성공 케이스
- answer canonical JSONL load 성공
- judgment canonical JSONL load 성공
- deterministic split이 seed 재현
- feature extraction runner가 70차원 feature JSONL 생성
- LightGBM training smoke 성공
- calibration wrapper score가 `[0, 1]` 반환
- exported artifact를 Phase 1 `load_lightgbm_scorer()`로 로드
- loaded artifact로 `StructuralConfidenceEstimator.score()` smoke 성공

### 실패 케이스
- missing required field
- invalid label
- duplicate id
- split class collapse
- split/sample minimum violation
- feature vector NaN
- metadata mismatch
- unsupported task_type
- README benchmark claim 없이 docs 유지

### 검증 명령
```bash
uv run pytest tests/test_confidence_training_*.py -q
uv run mypy src tests/test_confidence_training_*.py
./scripts/verify.sh
```

## 8. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] Phase 1 inference spec 확인
- [x] Trust reference summary 확인
- [x] task scope를 `answer_confidence` + `judgment_confidence`로 확정
- [x] dataset source를 task별 분리로 확정
- [x] canonical JSONL 우선 구현 및 adapter 후속 분리 확정
- [ ] 사용자 스펙 검토 및 최종 승인

### Phase 2: 로직 구현 (Implementation)
- [ ] `pyproject.toml`: `confidence-train` optional extra 추가
- [ ] `.gitignore`: generated training output ignore 규칙 추가
- [ ] `src/ranksmith/confidence_training/__init__.py`: public training submodule export
- [ ] `src/ranksmith/confidence_training/_errors.py`: training-specific error 구현
- [ ] `src/ranksmith/confidence_training/_types.py`: config/result/schema dataclass 구현
- [ ] `src/ranksmith/confidence_training/_dataset.py`: canonical JSONL validation/load 구현
- [ ] `src/ranksmith/confidence_training/_split.py`: deterministic split 구현
- [ ] `src/ranksmith/confidence_training/_features.py`: feature extraction runner 구현
- [ ] `src/ranksmith/confidence_training/_train.py`: LightGBM training 구현
- [ ] `src/ranksmith/confidence_training/_calibration.py`: calibration 구현
- [ ] `src/ranksmith/confidence_training/_report.py`: metrics/report 구현
- [ ] `src/ranksmith/confidence_training/_artifact.py`: artifact export 구현
- [ ] Phase 1 `ScorerMetadata` 필수 필드 export 검증 구현

### Phase 3: 검증 (Verification)
- [ ] `tests/test_confidence_training_dataset.py`: canonical schema tests
- [ ] `tests/test_confidence_training_split.py`: split tests
- [ ] `tests/test_confidence_training_features.py`: feature runner tests
- [ ] `tests/test_confidence_training_train.py`: training/calibration tests
- [ ] `tests/test_confidence_training_artifact.py`: artifact load smoke tests
- [ ] `tests/test_confidence_training_metadata.py`: Phase 1 metadata compatibility tests
- [ ] `./scripts/verify.sh` 실행

### Phase 4: 완료 및 정리
- [ ] `docs/wiki/02_architecture.md`: training utility layer 반영
- [ ] `README.md` / `README.ko.md`: training extra와 no-benchmark-claim caveat 반영
- [ ] 본 문서 최상단의 **상태**를 `Completed`로 변경

## 9. Open Questions
- `answer_confidence`의 semantic match label은 별도 evaluator 없이 제외한다.
- Phase 2B에서 adapter와 CLI를 각각 별도 구현 단위로 둘지 결정한다.
