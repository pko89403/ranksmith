# Spec: Public Strategy Protocols

## 1. 개요 (Overview)
- **작업 목적**: 사용자가 커스텀 reranking Strategy와 provider protocol을 공식 public contract로 import해 타입 안전하게 확장할 수 있게 한다.
- **Reference**: 기존 `docs/wiki/01_decisions.md`의 Strategy 모델(D006), pairwise provider contract(D009), 사용자 승인 계획.
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**:
  - 사용자는 `from ranksmith import RerankStrategy, LLMProvider`처럼 protocol을 import할 수 있어야 한다.
  - 사용자는 `from ranksmith.protocols import ...`로도 동일 protocol을 import할 수 있어야 한다.
  - 사용자는 `Provider` / `AsyncProvider` alias도 root import로 사용할 수 있어야 한다.
  - provider JSON ranking을 직접 쓰는 커스텀 Strategy는 public parser helper로 permutation을 검증할 수 있어야 한다.
- **출력 (Outputs)**:
  - 기존 `AzureOpenAIReranker(strategy=...)` 주입 방식은 유지한다.
  - 커스텀 sync/async strategy는 `RerankResult`를 직접 반환한다.
  - 커스텀 Strategy의 예상 밖 예외는 provider 오류가 아니라 strategy 오류로 분류한다.
- **제약 사항 (Constraints)**:
  - 새 algorithm 자체는 추가하지 않는다.
  - algorithm registry나 plugin discovery는 추가하지 않는다.
  - `AzureOpenAIProvider`는 public API로 승격하지 않는다.

## 3. 상세 설계 (Architecture & Design)
- **동작 메커니즘**:
  1. `ranksmith.protocols`에 strategy/provider protocol을 둔다.
  2. 기존 내부 구현은 public protocol 모듈을 import한다.
  3. `ranksmith.__init__`에서 protocol을 re-export한다.
  4. README에 커스텀 Strategy 작성 예제를 추가한다.
- **통합 지점 (Integration Points)**:
- `src/ranksmith/protocols.py`: 신규 public protocol module.
- `src/ranksmith/parsing.py`: public ranking parser helper.
  - `src/ranksmith/_providers.py`: provider protocol 중복 정의 제거 및 public module import.
  - `src/ranksmith/strategies.py`, `src/ranksmith/azure.py`, `src/ranksmith/__init__.py`: public protocol import/export 정리.
  - `examples/custom_strategy.py`: live credential 없이 실행 가능한 custom Strategy 예제.
  - `docs/wiki/08_custom_strategy_extension.md`: 사람/코딩 어시스턴트용 확장 규칙.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- Strategy와 provider protocol을 한 public module에 모아 사용자 확장 지점을 명확히 한다.
- 기존 `RerankResult` 생성 방식은 유지하고 helper는 추가하지 않는다.

## 5. 에러 핸들링 (Error Handling)
- protocol export 변경은 런타임 error 동작을 바꾸지 않는다.
- 기존 provider mismatch, parse error, document length error 정책은 그대로 유지한다.
- 커스텀 Strategy의 예상 밖 예외는 `RerankStrategyError`로 감싼다.
- public `parse_ranking_response()`는 invalid JSON, 누락/중복/범위 밖 ranking을 `RerankParseError`로 fast fail한다.
- `parse_ranking_response(expected_count < 0)`는 `RerankInputError`로 fast fail한다.
- custom Strategy가 내부 provider 실패를 의미 있게 분류해야 할 때는 `RerankProviderError`를 직접 raise한다.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**:
  - public root import와 `ranksmith.protocols` import가 성공한다.
  - `parse_ranking_response()`가 invalid ranking을 fast fail한다.
  - `parse_ranking_response()`가 음수 `expected_count`를 fast fail한다.
  - sync custom strategy가 `AzureOpenAIReranker`에 주입되어 1-based `rank`, 0-based `original_index`를 반환한다.
  - async custom strategy가 `AsyncAzureOpenAIReranker`에 주입되어 동일 계약을 반환한다.
  - custom strategy의 예상 밖 예외가 `RerankStrategyError`로 분류된다.
  - custom provider-backed strategy가 provider 실패를 `RerankProviderError`로 보존할 수 있다.
  - `examples/custom_strategy.py`가 live provider 없이 실행된다.
- **엣지/실패 케이스 (Edge & Failure Cases)**:
  - 기존 provider mismatch, invalid ranking, invalid winner 테스트가 계속 통과해야 한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] 승인된 계획을 spec으로 정리

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/protocols.py`: public protocol module 추가
- [x] `src/ranksmith/parsing.py`: public ranking parser helper 추가
- [x] `src/ranksmith/errors.py`: custom strategy error 추가
- [x] `src/ranksmith/_providers.py`: provider protocol 정의 이동
- [x] `src/ranksmith/strategies.py`, `src/ranksmith/azure.py`, `src/ranksmith/__init__.py`: public protocol 사용 및 export
- [x] `README.md`, `README.ko.md`: 커스텀 Strategy 문서 추가
- [x] `examples/custom_strategy.py`: 실행 가능한 오프라인 예제 추가
- [x] `docs/wiki/08_custom_strategy_extension.md`: 확장 가이드 추가
- [x] `docs/wiki/02_architecture.md`: 공식 Strategy 확장 지점 명시

### Phase 3: 검증 (Verification)
- [x] `tests/test_public_protocols.py`: public import 및 custom strategy 테스트 추가
- [x] `tests/test_examples.py`: custom strategy 예제 실행 테스트 추가
- [x] `./scripts/verify.sh` 통과 확인

### Phase 4: 완료 및 정리
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
