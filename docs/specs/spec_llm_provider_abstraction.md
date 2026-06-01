# Spec: LLM Provider Abstraction

> **Historical Spec**
> 이 문서는 과거 구현 당시의 설계 기록입니다.
> 현재 code/API 기준은 `docs/wiki/02_architecture.md`와
> `docs/wiki/08_custom_strategy_extension.md`를 따릅니다.
> 아래 provider 파일 경로와 stub 명칭은 현재 구조에 맞게 최소 보정했습니다.

## 1. 개요 (Overview)
- **작업 목적**: LLM 호출 계층을 `Strategy -> ModelClient -> ModelProvider`로 분리해, ranksmith 도메인 계약과 vendor SDK 호출을 독립적으로 유지한다.
- **Reference**: 사용자 승인 계획, `docs/wiki/02_architecture.md`, Clean Architecture 의존성 역전 원칙.
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**:
  - `ModelClient`는 query/documents 기반의 `rank`, `compare`, `select` 요청을 받는다.
  - `ModelProvider`는 vendor-neutral `ModelRequest`를 받아 JSON completion을 반환한다.
- **출력 (Outputs)**:
  - `ModelClient`는 기존 Strategy가 기대하던 JSON 문자열 계약을 유지한다.
  - Azure provider는 실제 Azure OpenAI SDK 호출 결과를 `ModelResponse`로 반환한다.
  - OpenAI/Anthropic/Gemini stub provider는 명확한 `RerankProviderError`로 실패한다.
- **제약 사항 (Constraints)**:
  - 기존 `LLMProvider` 계열 public API는 제거한다.
  - deprecated alias는 남기지 않는다.
  - 실제 OpenAI/Anthropic/Gemini SDK dependency는 추가하지 않는다.
  - `AzureOpenAIReranker` / `AsyncAzureOpenAIReranker` 이름은 당장 유지한다.

## 3. 상세 설계 (Architecture & Design)
- **동작 메커니즘**:
  1. Strategy는 `ModelClient` 또는 `AsyncModelClient`만 받는다.
  2. Strategy는 `rank`, `compare`, `select` 중 필요한 도메인 메서드만 호출한다.
  3. `ModelClient`는 prompt를 만들고 `ModelRequest`로 변환한다.
  4. `ModelProvider`는 `ModelRequest`를 vendor SDK 호출로 실행한다.
  5. `ModelClient`는 usage callback, empty response, provider exception 정책을 처리한다.
- **통합 지점 (Integration Points)**:
  - `src/ranksmith/model.py`: public DTO, protocol, model client.
  - `src/ranksmith/providers/_azure.py`: Azure provider.
  - `src/ranksmith/providers/_stubs.py`: OpenAI/Anthropic/Gemini stub provider.
  - `src/ranksmith/_providers.py`: 기존 import 경로 호환 re-export layer.
  - `src/ranksmith/protocols.py`: strategy protocol과 model protocol re-export.
  - `src/ranksmith/strategies/`: provider 인자를 `ModelClient`로 전환.
  - `src/ranksmith/azure.py`: wrapper가 `ModelClient(AzureAOAIProvider(...))`를 조립.
  - `src/ranksmith/__init__.py`: 새 public API export, 기존 provider protocol 제거.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- prompt 생성은 `ModelClient`에 모아 vendor provider가 ranking 도메인을 알지 않게 한다.
- provider별 SDK 호출은 `ModelProvider.complete()` 내부에 격리한다.
- stub provider는 후속 실제 구현 전까지 import 가능한 placeholder로만 둔다.

## 5. 에러 핸들링 (Error Handling)
- vendor SDK 예외는 `RerankProviderError`로 감싼다.
- `ModelResponse.content`가 비어 있으면 `RerankProviderError`로 실패한다.
- provider mismatch는 기존처럼 `RerankInputError`로 실패한다.
- stub provider 호출은 `RerankProviderError("<Provider> provider is not implemented yet.")`로 실패한다.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**:
  - 새 public API import가 가능하다.
  - `ModelClient.rank/compare/select`가 기존 JSON 계약을 유지한다.
  - Azure provider가 `ModelRequest`를 SDK kwargs로 변환한다.
  - `AzureOpenAIReranker` wrapper가 기존 생성자 사용법으로 동작한다.
- **엣지/실패 케이스 (Edge & Failure Cases)**:
  - 기존 `LLMProvider` 계열 import surface가 제거된다.
  - stub provider는 `RerankProviderError`로 fast fail 한다.
  - empty response와 SDK 예외가 `RerankProviderError`로 fast fail 한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] 스펙 문서 작성

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/model.py`: DTO, protocol, client 구현
- [x] `src/ranksmith/providers/_azure.py`, `src/ranksmith/providers/_stubs.py`: Azure provider 및 stub provider 구현
- [x] `src/ranksmith/strategies/`, `src/ranksmith/azure.py`: ModelClient 경로로 전환
- [x] `src/ranksmith/__init__.py`, `src/ranksmith/protocols.py`: public API 정리

### Phase 3: 검증 (Verification)
- [x] `tests/test_model_architecture.py`: 신규 public API 및 provider 테스트 추가
- [x] 기존 strategy/reranker 테스트 갱신
- [x] `./scripts/verify.sh` 통과 확인

### Phase 4: 완료 및 정리
- [x] README / README.ko / wiki 갱신
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
