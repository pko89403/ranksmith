# Spec: Async Reranking Support (비동기 지원)

> **작성 가이드**: 이 문서는 코딩 어시스턴트의 작업 추적용이기도 하지만, **최우선적으로 사람(개발자)이 읽고 이해하기 가장 좋은 형태(가독성)**여야 합니다. 
> 장황한 설명은 피하고, 핵심을 찌르는 간결한 문장, 명확한 목록(List), 구조화된 마크다운 포맷을 활용하세요.

## 1. 개요 (Overview)
- **작업 목적**: 외부 LLM API(Azure OpenAI 등) 호출 시 발생하는 네트워크 I/O 병목을 해소하고, 고성능/대용량 트래픽 처리에 적합한 비동기(Async) 리랭킹 인프라를 구축합니다.
- **Reference**: Python `asyncio`, OpenAI Python SDK AsyncClient
- **상태**: `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**: 기존 동기 인터페이스와 동일하나, `await`를 통해 호출되어야 함.
- **출력 (Outputs)**: 기존 `RerankResult` 반환 구조 유지.
- **제약 사항 (Constraints)**:
  - 기존 동기식 코드(Synchronous)와 비동기식 코드(Asynchronous) 간의 모듈 재사용성을 극대화하여 중복 로직(파싱, 유효성 검증)을 최소화해야 합니다. (장기 지속성 최우선)
  - `openai.AsyncAzureOpenAI` 클라이언트를 사용해야 합니다.
  - 기존 `Fast Fail` 정책 및 에러(`RerankInputError`, `RerankProviderError`)를 비동기 환경에서도 완벽히 동일하게 유지해야 합니다.

## 3. 상세 설계 (Architecture & Design)

- **동작 메커니즘**:
  1. `AsyncAzureOpenAIReranker`가 쿼리와 문서를 입력받습니다.
  2. 비동기 전략인 `AsyncRerankStrategy`(예: `AsyncListwiseStrategy`)로 요청을 위임합니다.
  3. `AsyncLLMProvider`를 통해 `await client.chat.completions.create(...)`로 비동기 API 호출을 수행합니다.
  4. 응답을 파싱하고 예외를 처리한 뒤 순위를 반환합니다.

- **의사 알고리즘 (Pseudo-algorithm)**:
  ```text
  Async Sliding Window:
  1. documents를 크기 window_size로 분할
  2. for each window in reversed(windows):
  3.     ranking = AWAIT AsyncLLMProvider.rank(window)
  4.     update positions based on ranking
  5. return final positions
  ```

- **의사 코드 (Pseudo-code)**:
  ```python
  class AsyncLLMProvider(Protocol):
      async def rank(self, query: str, documents: list[Document]) -> str: ...

  class AsyncAzureOpenAIProvider:
      async def rank(self, query: str, documents: list[Document]) -> str:
          response = await self._client.chat.completions.create(...)
          return response.choices[0].message.content

  class AsyncRerankStrategy(Protocol):
      async def rerank(self, query: str, documents: list[Document], provider: AsyncLLMProvider, ...) -> list[RerankResult]: ...
  ```

- **통합 지점 (Integration Points)**:
  - `src/ranksmith/_providers.py`: `AsyncLLMProvider`, `AsyncAzureOpenAIProvider` 추가
  - `src/ranksmith/strategies.py`: `AsyncRerankStrategy`, `AsyncListwiseStrategy` 추가
  - `src/ranksmith/azure.py`: `AsyncAzureOpenAIReranker` 추가

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **공통 컴포넌트 식별 (Shared Components)**:
  - 프롬프트 렌더링 (`_build_prompt` 등)
  - JSON 응답 파싱 로직 및 Rank GPT 순위 유효성 검사 (Fast Fail 로직)
  - 문서 정규화 (`_normalize_documents`)
- **추상화 방안 (Abstraction Plan)**:
  - 동기/비동기 Provider가 공통으로 사용할 수 있도록 파싱 로직과 프롬프트 생성 로직을 독립적인 순수 함수(Pure function)나 Mixin으로 분리(`_utils.py` 등).
  - Listwise 알고리즘의 순위 교환 로직(Sliding window 인덱스 계산 등)은 I/O가 없으므로 동기/비동기 클래스에서 공통 유틸리티로 추출하여 재사용합니다.

## 5. 에러 핸들링 (Error Handling)
- 기존 동기 구현과 동일한 `RerankProviderError`, `RerankInputError`, `RerankParseError` 사용.
- 비동기 태스크 취소(Cancellation) 및 Timeout 예외 발생 시 `RerankProviderError`로 래핑하여 `Fast Fail` 원칙 준수.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**: `pytest-asyncio`를 활용하여 `AsyncAzureOpenAIReranker`가 `await`로 정상 순위를 반환하는지 검증.
- **엣지/실패 케이스 (Edge & Failure Cases)**:
  - 비동기 Timeout 초과 시 즉시 실패하는지 검증.
  - JSON 파싱 에러 응답 모의(Mocking) 시 조용한 보정 없이 즉각 `RerankParseError`가 발생하는지 검증.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] 스펙 문서(본 문서) 상의 의사 코드 설계 검토 및 확정

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/_providers.py`: `AsyncLLMProvider`, `AsyncAzureOpenAIProvider` 구현
- [x] `src/ranksmith/strategies.py`: `AsyncRerankStrategy`, `AsyncListwiseStrategy` 구현 (공통 로직 분리 포함)
- [x] `src/ranksmith/azure.py`: `AsyncAzureOpenAIReranker` 구현

### Phase 3: 검증 (Verification)
- [x] `tests/test_async_providers.py` 등 비동기 전용 테스트 추가
- [x] 단위 테스트(Happy & Edge cases) 통과 확인
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] 필요 시 `docs/wiki/` 하위 문서 업데이트
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
