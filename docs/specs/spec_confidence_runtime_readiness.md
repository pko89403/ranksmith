# Spec: Confidence Runtime Readiness

## 1. 개요 (Overview)
- **작업 목적**: `confidence_training`이 생성한 scorer artifact를 runtime에서 안정적으로 소비해, CBDR 같은 후속 reranking module이 사용할 수 있는 confidence signal을 만든다.
- **Reference**:
  - `docs/specs/spec_structural_confidence.md`
  - `docs/specs/spec_confidence_training_pipeline.md`
  - `docs/wiki/references/structural_confidence.md`
- **상태**: `[x] Draft` | `[ ] In Progress` | `[ ] Completed`

이 스펙은 reranking `Strategy` 또는 `Algorithm`을 구현하지 않는다.

목표는 다음 경로를 완성하는 것이다.

```text
training artifact
  -> runtime estimator 생성
  -> single/batch confidence scoring
  -> bounded memory/parallel execution
  -> CBDR-ready confidence signal
```

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 포함 범위
- `StructuralConfidenceEstimator.from_artifact(...)` 추가.
- `StructuralConfidenceEstimator.score_batch(...)` 추가.
- batch 입력을 chunk 단위로 처리한다.
- `batch_size`, `max_workers`, `max_batch_items` 실행 정책을 지원한다.
- batch 결과는 입력 순서를 보존한다.
- training artifact와 runtime estimator의 metadata compatibility를 검증한다.
- `judgment_confidence` 결과를 CBDR 직전 signal로 사용할 수 있게 결과 contract를 명확히 한다.
- hidden state, feature vector, HuggingFace token, cache path는 결과 metadata에 노출하지 않는다.

### 제외 범위
- closed model judgment 생성.
- CBDR reranking algorithm.
- confidence score fusion formula.
- reranking `Strategy` 추가.
- async API.
- provider 호출 병렬화.
- feature cache.
- benchmark integration.
- README benchmark 수치 또는 성능 claim.
- `on_error="return_error"` 같은 partial failure mode.

### 입력 (Inputs)

#### `from_artifact(...)`
필수:
- `artifact_path: str | Path`

선택:
- `metadata_path: str | Path | None = None`
- `encoder_name: str | None = None`
- `encoder_revision: str | None = None`
- `tokenizer_name: str | None = None`
- `tokenizer_revision: str | None = None`
- `task_type: TaskType | None = None`
- `hf_token: str | None = None`
- `local_files_only: bool = False`
- `cache_dir: str | None = None`
- `device: str = "cpu"`
- `max_length: int | None = None`
- `allow_truncation: bool = False`

기본값 규칙:
- `encoder_name is None`이면 artifact metadata의 `encoder_name`을 사용한다.
- `encoder_revision is None`이면 artifact metadata의 `encoder_revision`을 사용한다.
- `tokenizer_name is None`이면 artifact metadata의 `tokenizer_name`을 사용한다.
- `tokenizer_revision is None`이면 artifact metadata의 `tokenizer_revision`을 사용한다.
- `task_type is None`이면 artifact metadata의 `task_type`을 사용한다.
- `max_length is None`이면 artifact metadata의 `max_length`를 사용한다.

사용자가 값을 override한 경우, 최종 estimator 설정과 artifact metadata가 일치하지 않으면 `ConfidenceArtifactError`로 실패한다.

#### `score_batch(...)`
필수:
- `items: Sequence[StructuralConfidenceInput]`

선택:
- `batch_size: int = 8`
- `max_workers: int = 1`
- `max_batch_items: int | None = None`

입력 제약:
- `items`는 비어 있으면 실패한다.
- `batch_size >= 1`.
- `max_workers >= 1`.
- `max_batch_items`가 지정되면 `len(items) <= max_batch_items`여야 한다.
- 모든 item은 estimator의 `task_type`과 맞아야 한다.
- 기본 failure policy는 fast fail이다.

### 출력 (Outputs)

`score(...)`는 기존 `StructuralConfidenceResult`를 유지한다.

`score_batch(...)`는 입력 순서와 같은 순서의 result list를 반환한다.

```python
list[StructuralConfidenceResult]
```

CBDR 연결은 별도 `Document` 식별자를 confidence module 안에 넣지 않는다.
confidence module은 입력 순서 보존을 보장하고, CBDR 쪽 adapter가 candidate id와 result를 결합한다.

`judgment_confidence`에서 CBDR이 소비하는 최소 signal은 다음 값으로 정의한다.

```text
candidate identity: CBDR/candidate layer 책임
judgment text: caller가 보유
confidence_score: StructuralConfidenceResult.score
task_type: "judgment_confidence"
feature_schema_version: "structural-v1"
metadata: scorer/encoder compatibility metadata
```

## 3. 상세 설계 (Architecture & Design)

### 동작 메커니즘

#### `from_artifact(...)`
1. `load_lightgbm_scorer(artifact_path, metadata_path=metadata_path)`로 scorer를 로드한다.
2. scorer metadata에서 encoder/tokenizer/task/max_length 기본값을 가져온다.
3. 사용자가 명시한 override를 적용한다.
4. `FrozenAutoEncoder.from_pretrained(...)`로 encoder를 로드한다.
5. `StructuralConfidenceEstimator(...)` 생성자에서 기존 metadata validation을 수행한다.
6. metadata가 맞지 않으면 estimator를 반환하지 않고 실패한다.

#### `score_batch(...)`
1. batch 설정을 검증한다.
2. 전체 item 수가 `max_batch_items`를 넘으면 실패한다.
3. 입력을 `batch_size` 단위 chunk로 나눈다.
4. 각 chunk의 item을 score한다.
5. 결과를 입력 순서대로 append한다.
6. chunk 처리 후 hidden state와 feature vector를 보관하지 않는다.
7. 하나라도 실패하면 전체 호출이 실패한다.

### 병렬 처리 정책

기본값은 안정성을 위해 `max_workers=1`이다.

`max_workers > 1`이면 chunk 내부 item scoring을 bounded worker pool로 병렬화한다.

정책:
- 결과 순서는 반드시 입력 순서와 동일해야 한다.
- worker 수는 `max_workers`를 넘지 않는다.
- 병렬 단위는 chunk 내부 item이다.
- estimator, encoder, scorer는 요청별 mutable state를 저장하지 않는 read-only object로 취급한다.
- 병렬 실행에서도 각 item의 hidden state와 feature vector는 해당 worker 안에서만 사용하고 결과에 저장하지 않는다.
- 조용히 단일 worker로 낮추지 않는다.
- provider/closed model 호출 병렬화는 이 스펙 범위가 아니다.

권장 구현:
- 1차 구현은 `max_workers=1`을 완전 지원한다.
- `max_workers > 1`은 `ThreadPoolExecutor` 같은 bounded worker pool로 구현한다.
- worker별 결과는 원래 batch index와 함께 수집한 뒤 정렬 또는 index assignment로 순서를 복원한다.
- worker에서 발생한 첫 번째 confidence error는 전체 batch 실패로 전파한다.

구현 상태:
- `max_workers=1`은 sequential batch로 처리한다.
- `max_workers > 1`은 bounded worker pool로 처리한다.
- `max_workers < 1`은 `ConfidenceInputError`로 실패한다.

### 메모리 관리 정책

- 전체 item을 한 번에 encoder에 넣지 않는다.
- `batch_size` chunk 단위로 처리한다.
- hidden state는 chunk/item 처리 후 결과에 저장하지 않는다.
- structural feature vector는 결과 metadata에 저장하지 않는다.
- `hf_token`, `cache_dir`, local path, model/tokenizer 객체는 result metadata에 저장하지 않는다.
- `max_batch_items`는 caller가 큰 candidate set을 명시적으로 제한하기 위한 guard다.
- token 길이 초과는 기존 `allow_truncation` 정책을 따른다.

### 의사 알고리즘 (Pseudo-algorithm)

```text
from_artifact(path):
  scorer = load_lightgbm_scorer(path)
  metadata = scorer.metadata
  config = resolve(metadata defaults + explicit overrides)
  encoder = FrozenAutoEncoder.from_pretrained(config)
  return StructuralConfidenceEstimator(encoder, scorer, config.task_type)

score_batch(items, batch_size, max_workers, max_batch_items):
  validate_batch_config(...)
  if max_batch_items is not None and len(items) > max_batch_items:
    fail

  results = []
  for chunk in chunks(items, batch_size):
    chunk_results = score_chunk(chunk, max_workers)
    append chunk_results in original order

  return results
```

### 의사 코드 (Pseudo-code)

```python
class StructuralConfidenceEstimator:
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
    ) -> "StructuralConfidenceEstimator":
        scorer = load_lightgbm_scorer(artifact_path, metadata_path=metadata_path)
        metadata = scorer.metadata
        return cls.from_pretrained(
            scorer=scorer,
            task_type=task_type or metadata.task_type,
            encoder_name=encoder_name or metadata.encoder_name,
            encoder_revision=(
                metadata.encoder_revision if encoder_revision is None else encoder_revision
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

    def score_batch(
        self,
        items: Sequence[StructuralConfidenceInput],
        *,
        batch_size: int = 8,
        max_workers: int = 1,
        max_batch_items: int | None = None,
    ) -> list[StructuralConfidenceResult]:
        validate_batch_options(items, batch_size, max_workers, max_batch_items)
        results = []
        for chunk in chunked(items, batch_size):
            results.extend(self._score_chunk(chunk, max_workers=max_workers))
        return results
```

### 통합 지점 (Integration Points)

- `src/ranksmith/confidence/_structural.py`
  - `StructuralConfidenceEstimator.from_artifact(...)`
  - `StructuralConfidenceEstimator.score_batch(...)`
  - batch option validation helpers
  - chunk helper
- `tests/test_confidence_estimator.py`
  - single/batch estimator tests
- `tests/test_confidence_training_artifact.py`
  - training artifact -> runtime estimator smoke tests

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### Shared Components
- `load_lightgbm_scorer()`
  - artifact loading을 재사용한다.
- `ScorerMetadata`
  - runtime default 복원과 compatibility validation에 사용한다.
- `FrozenAutoEncoder`
  - 기존 encoder wrapper를 재사용한다.
- `format_confidence_input()`
  - task별 input validation을 재사용한다.
- `extract_structural_features()`
  - feature schema를 변경하지 않는다.

### Abstraction Plan
- artifact path 기반 생성 로직은 `from_artifact(...)`에만 둔다.
- batch option validation은 private helper로 분리한다.
- chunking은 private helper로 분리한다.
- CBDR candidate identity는 confidence module에 넣지 않는다.
- result type은 기존 `StructuralConfidenceResult`를 우선 사용한다.
  - 별도 signal dataclass는 candidate identity까지 포함해야 할 때만 후속 spec으로 분리한다.

## 5. 에러 핸들링 (Error Handling)

- artifact load 실패: `ConfidenceArtifactError`.
- artifact metadata 누락/불일치: `ConfidenceArtifactError`.
- override로 metadata mismatch 발생: `ConfidenceArtifactError`.
- `items`가 빈 sequence: `ConfidenceInputError`.
- `batch_size < 1`: `ConfidenceInputError`.
- `max_workers < 1`: `ConfidenceInputError`.
- `max_batch_items < 1`: `ConfidenceInputError`.
- `len(items) > max_batch_items`: `ConfidenceInputError`.
- item task type mismatch: 기존 `ConfidenceInputError`.
- token length 초과: 기존 `ConfidenceInputError`.
- scorer score가 probability가 아닌 경우: 기존 `ConfidenceArtifactError`.

실패는 기본적으로 전체 batch를 중단한다.
partial result를 반환하지 않는다.

## 6. 테스트 계획 (Test Plan)

### 성공 케이스 (Happy Paths)
- `from_artifact(...)`가 scorer metadata를 기본값으로 사용해 estimator를 생성한다.
- `from_artifact(...)`가 `metadata_path`를 LightGBM booster artifact에 전달한다.
- `score_batch(...)`가 입력 순서와 같은 순서로 결과를 반환한다.
- `score_batch(...)`가 `batch_size` 단위로 모든 item을 처리한다.
- `judgment_confidence` artifact로 `JudgmentConfidenceInput` batch를 scoring한다.
- training pipeline이 export한 artifact를 `from_artifact(...)`로 로드해 score한다.

### Edge & Failure Cases
- 빈 batch는 실패한다.
- `batch_size=0`은 실패한다.
- `max_workers=0`은 실패한다.
- `max_batch_items=0`은 실패한다.
- `len(items) > max_batch_items`는 실패한다.
- artifact metadata와 override가 다르면 실패한다.
- `answer_confidence` estimator에 `JudgmentConfidenceInput` batch를 넣으면 실패한다.
- `judgment_confidence` estimator에 `AnswerConfidenceInput` batch를 넣으면 실패한다.
- `max_workers > 1`은 bounded parallel scoring으로 입력 순서 보존과 worker error 전파를 검증한다.
- batch 중간 item이 실패하면 partial result 없이 전체 호출이 실패한다.
- result metadata에 token, cache path, feature vector, hidden state가 없어야 한다.

### 검증 명령
```bash
uv run pytest tests/test_confidence_*.py tests/test_confidence_training_*.py -q
uv run ruff check src/ranksmith/confidence tests/test_confidence_*.py
uv run mypy src/ranksmith/confidence tests/test_confidence_*.py
./scripts/verify.sh
```

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] `spec_structural_confidence.md` 확인
- [x] `spec_confidence_training_pipeline.md` 확인
- [x] CBDR 직전 준비 범위 확정
- [ ] 스펙 문서 리뷰 및 최종 승인

### Phase 2: Runtime Artifact API 구현
- [x] `src/ranksmith/confidence/_structural.py`: `from_artifact(...)` 구현
- [x] `src/ranksmith/confidence/_structural.py`: metadata default resolution 구현
- [x] `src/ranksmith/confidence/_structural.py`: override mismatch fast fail 테스트 보강

### Phase 3: Batch Scoring 구현
- [x] `src/ranksmith/confidence/_structural.py`: `score_batch(...)` 구현
- [x] `src/ranksmith/confidence/_structural.py`: batch option validation helper 구현
- [x] `src/ranksmith/confidence/_structural.py`: chunk helper 구현
- [x] `src/ranksmith/confidence/_structural.py`: `max_workers` 정책 구현
- [ ] `src/ranksmith/confidence/_structural.py`: memory-safe result metadata 유지

### Phase 4: 검증
- [x] `tests/test_confidence_estimator.py`: `from_artifact(...)` 정상/실패 테스트 추가
- [x] `tests/test_confidence_estimator.py`: `score_batch(...)` 정상/실패 테스트 추가
- [ ] `tests/test_confidence_training_artifact.py`: training artifact -> runtime batch smoke test 추가
- [ ] `uv run pytest tests/test_confidence_*.py tests/test_confidence_training_*.py -q`
- [ ] `uv run ruff check src/ranksmith/confidence tests/test_confidence_*.py`
- [ ] `uv run mypy src/ranksmith/confidence tests/test_confidence_*.py`
- [ ] `./scripts/verify.sh`

### Phase 5: 완료 및 정리
- [ ] 필요 시 `docs/wiki/02_architecture.md` Confidence 범위 업데이트
- [ ] 본 문서 상태를 `Completed`로 변경
