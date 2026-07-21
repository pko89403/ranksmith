# Spec: CBDR User-Facing Integration Layer

## 1. 개요 (Overview)
- **작업 목적**: scorer artifact와 Azure 설정만으로 `CBDRStrategy`를 쉽게 생성하고, 기존 benchmark runner에서 `--algorithm cbdr`로 실행할 수 있게 한다.
- **Reference**:
  - `docs/specs/spec_cbdr_strategy.md`
  - `docs/specs/spec_confidence_gain_reranking.md`
  - `docs/wiki/references/parametric_post_retrieval_confidence.md`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**:
  - `CBDRStrategy.from_artifacts(...)`: base/context scorer artifact path, optional metadata path, answer generator, threshold, HF 로딩 옵션.
  - `AzureAnswerGenerator`: Azure OpenAI credential/config 또는 기존 환경 변수, no-answer sentinel.
  - `compare_reranking.py`: `--algorithm cbdr`, CBDR artifact flags, HF/runtime flags.
- **출력 (Outputs)**:
  - 기존 `CBDRStrategy.rerank(...)`와 동일한 `list[RerankResult]`.
  - benchmark report에는 CBDR method settings와 provider call upper bound estimate를 기록한다.
- **제약 사항 (Constraints)**:
  - CBDR은 sync만 지원한다.
  - root import는 확장하지 않는다.
  - scorer training, retriever integration, async CBDR은 제외한다.
  - 잘못된 설정과 malformed closed-model output은 fast fail한다.

## 3. 상세 설계 (Architecture & Design)
- `CBDRStrategy.from_artifacts(...)`는 `StructuralConfidenceEstimator.from_artifact(...)`를 두 번 호출해 base/context estimator를 만든다.
- `ranksmith.integrations.AzureAnswerGenerator`는 query-only와 query+context answer 생성을 담당한다.
- `AzureAnswerGenerator`는 generation pipeline과 같은 no-answer sentinel 계약을 prompt에 포함하고, JSON object 응답에서 `answer` 문자열만 파싱한다.
- `compare_reranking.py`는 CBDR일 때 factory와 Azure answer generator를 조립한 뒤 `CBDRStrategy.rerank(...)`를 직접 호출한다.

### Pseudo-algorithm
```text
create CBDR:
  base_estimator = StructuralConfidenceEstimator.from_artifact(base_artifact, task=query_answerability_confidence, ...)
  context_estimator = StructuralConfidenceEstimator.from_artifact(context_artifact, task=query_context_answerability_confidence, ...)
  return CBDRStrategy(base_estimator, context_estimator, answer_generator, threshold)

benchmark cbdr:
  validate live opt-in
  validate artifact flags
  answer_generator = AzureAnswerGenerator.from_env(...)
  strategy = CBDRStrategy.from_artifacts(...)
  results = strategy.rerank(query=query, documents=documents)
```

### Integration Points
- `src/ranksmith/strategies/_cbdr.py`
- `src/ranksmith/integrations/`
- `scripts/compare_reranking.py`
- `tests/test_cbdr_strategy.py`
- `tests/test_azure_answer_generator.py`
- `tests/test_compare_reranking.py`

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- `CBDRStrategy.from_artifacts(...)`는 estimator 조립만 담당하고 reranking logic은 기존 `CBDRStrategy.rerank(...)`를 재사용한다.
- `AzureAnswerGenerator`는 Strategy가 아니라 integration helper다.
- benchmark runner는 CBDR 전용 설정만 추가하고 기존 algorithm selection/report 구조를 유지한다.

## 5. 에러 핸들링 (Error Handling)
- artifact path 누락: `SystemExit` 또는 기존 artifact error.
- 잘못된 confidence task type: 기존 `ConfidenceArtifactError`.
- malformed answer JSON: `RerankParseError`.
- missing/empty answer: `RerankParseError`.
- Azure env 누락: `RerankInputError`.
- invalid no-answer sentinel: `ValueError`.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스**:
  - `CBDRStrategy.from_artifacts(...)`가 base/context artifact를 올바른 task type으로 로드한다.
  - `AzureAnswerGenerator`가 `{"answer": "..."}`를 파싱한다.
  - `AzureAnswerGenerator` prompt가 no-answer sentinel 계약을 포함한다.
  - `compare_reranking.py --algorithm cbdr`가 CBDR strategy를 생성한다.
- **엣지/실패 케이스**:
  - artifact flag 누락 시 실패.
  - `--allow-live` 없으면 CBDR 실행 차단.
  - `--cbdr-max-document-chars`가 strategy factory로 전달됨.
  - malformed/missing/empty answer는 실패.
  - HF token env 이름은 실제 env 값으로 resolve한다.
- **공통 Reranking Smoke/Benchmark**:
  - live provider는 opt-in으로만 실행한다.
  - CBDR provider call estimate는 upper bound로 기록한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] 스펙 문서(본 문서) 상의 의사 코드 설계 검토 및 확정

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/strategies/_cbdr.py`: `from_artifacts(...)` factory 추가
- [x] `src/ranksmith/integrations/`: `AzureAnswerGenerator` 추가
- [x] `scripts/compare_reranking.py`: CBDR algorithm/flags/estimate 연결

### Phase 3: 검증 (Verification)
- [x] `tests/test_cbdr_strategy.py`: factory 테스트 추가
- [x] `tests/test_azure_answer_generator.py`: answer generator 테스트 추가
- [x] `tests/test_compare_reranking.py`: benchmark 연결 테스트 추가
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] `docs/wiki/02_architecture.md` 업데이트
- [x] `README.md`, `README.ko.md` 업데이트
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
