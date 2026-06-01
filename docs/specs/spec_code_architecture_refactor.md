# Spec: 코드 아키텍처 리팩터링

> **작성 가이드**: 이 문서는 코딩 어시스턴트의 작업 추적용이기도 하지만, **최우선적으로 사람(개발자)이 읽고 이해하기 가장 좋은 형태(가독성)**여야 합니다.
> 장황한 설명은 피하고, 핵심을 찌르는 간결한 문장, 명확한 목록(List), 구조화된 마크다운 포맷을 활용하세요.

## 1. 개요 (Overview)
- **작업 목적**: 기존 reranking method의 외부 동작을 유지하면서, 파일 크기와 중복을 줄이고 새 method/provider 추가가 쉬운 구조로 정리한다.
- **Reference**:
  - `docs/wiki/00_context.md`
  - `docs/wiki/01_decisions.md`
  - `docs/wiki/02_architecture.md`
  - `docs/wiki/06_verification_policy.md`
  - `docs/wiki/08_custom_strategy_extension.md`
  - 기존 코드 리뷰에서 확인된 구조 개선 항목
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**:
  - 기존 public API 입력 계약을 유지한다.
  - `AzureOpenAIReranker.rerank(query, documents, top_k=None)` 계약을 유지한다.
  - `AsyncAzureOpenAIReranker.rerank(query, documents, top_k=None)` 계약을 유지한다.
  - `ListwiseStrategy`, `PairwiseStrategy`, `TourRankStrategy`, `AcuRankStrategy` 및 async 대응 클래스의 생성자 파라미터를 유지한다.
- **출력 (Outputs)**:
  - 기존 `list[RerankResult]` 반환 계약을 유지한다.
  - `rank`는 1-based, `original_index`는 0-based를 유지한다.
  - 기존 metadata key를 제거하거나 이름 변경하지 않는다.
- **제약 사항 (Constraints)**:
  - Public API를 확장하지 않는다.
  - 기존 algorithm 의미를 바꾸지 않는다.
  - ranking/selection 응답을 조용히 보정하지 않는다.
  - 긴 문서를 조용히 자르지 않는다.
  - 리팩터링은 동작 보존을 우선한다.
  - 새 reference 기반 algorithm 구현은 이 spec 범위가 아니다.
  - 현재 추적되지 않은 Setwise PDF는 이 작업에서 처리하지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 동작 메커니즘
1. `strategies.py`에 모여 있는 strategy 구현을 method별 모듈로 분리한다.
2. `src/ranksmith/strategies.py` 파일은 `src/ranksmith/strategies/` 패키지로 전환한다. 같은 이름의 파일과 디렉터리는 공존할 수 없으므로, 이동 완료 후 기존 파일은 삭제한다.
3. 공통 입력 검증 helper를 `strategies/_common.py`로 이동한다.
4. parser는 ranking/selection/winner contract를 더 엄격히 검증한다.
5. provider 구현은 `providers/` 패키지로 분리하되, 기존 root import와 `_providers.py` 호환 경로를 유지한다.
6. sync/async 중복은 무리한 추상화보다 공통 helper 공유로 줄인다.
7. 기존 테스트를 먼저 보강한 뒤 파일 이동과 import 변경을 진행한다.

### 의사 알고리즘 (Pseudo-algorithm)
```text
for each public rerank entry:
  normalize documents
  validate top_k before model call
  delegate to selected Strategy

for each built-in strategy direct use:
  validate top_k before model call
  validate document length before model call
  run existing algorithm flow

top_k validation is intentionally duplicated:
  public reranker entry protects normal users
  built-in strategy validation protects direct Strategy users

for each parser:
  load JSON
  require expected object key
  require list/string shape
  reject bool values explicitly
  validate length, duplicate, range
  return validated contract value

for each strategy:
  keep existing ranking algorithm order
  use shared validation helper
  keep result construction local unless two strategies are truly identical
```

### 의사 코드 (Pseudo-code)
```python
def validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k < 0:
        raise RerankInputError("top_k must be greater than or equal to 0")


def _is_json_int(value: object) -> bool:
    return type(value) is int


def parse_ranking_response(raw_response: str, *, expected_count: int) -> list[int]:
    data = json.loads(raw_response)
    ranking = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(ranking, list):
        raise RerankParseError(
            'LLM response must contain a "ranking" list.',
            raw_response,
        )
    if not all(_is_json_int(item) for item in ranking):
        raise RerankParseError("ranking must contain only integers.", raw_response)
    validate_permutation(ranking, expected_count)
    return ranking
```

### 통합 지점 (Integration Points)
- `src/ranksmith/strategies.py`
  - 기존 구현을 method별 모듈로 이동한 뒤 파일을 삭제한다.
  - `src/ranksmith/strategies.py`와 `src/ranksmith/strategies/`는 동시에 존재할 수 없다.
- `src/ranksmith/strategies/__init__.py`
  - public strategy class를 재노출한다.
  - 기존 `from ranksmith.strategies import ListwiseStrategy` 형태의 import 경로를 이 패키지에서 유지한다.
- `src/ranksmith/strategies/_common.py`
  - `validate_top_k`, document length 검증, provider capability guard를 둔다.
- `src/ranksmith/strategies/_listwise.py`
  - `ListwiseStrategy`, `AsyncListwiseStrategy`.
- `src/ranksmith/strategies/_pairwise.py`
  - `PairwiseStrategy`, `AsyncPairwiseStrategy`, `_parse_pairwise_winner_response()` private helper.
- `src/ranksmith/strategies/_tourrank.py`
  - `TourRankStrategy`, `AsyncTourRankStrategy`, `TourRankStageConfig`.
- `src/ranksmith/strategies/_acurank.py`
  - `AcuRankStrategy`, `AsyncAcuRankStrategy`, TrueSkill helper.
- `src/ranksmith/parsing.py`
  - `bool`을 integer로 인정하지 않도록 수정한다.
  - `parse_pairwise_winner_response()`는 추가하지 않는다.
- `src/ranksmith/providers/_azure.py`
  - Azure sync/async provider 구현.
- `src/ranksmith/providers/_stubs.py`
  - OpenAI/Anthropic/Gemini 미구현 stub.
- `src/ranksmith/_providers.py`
  - 기존 import 경로 호환을 위해 provider class를 re-export한다.
- `src/ranksmith/__init__.py`
  - 기존 public import 표면을 유지한다.
- `__all__` contract
  - `src/ranksmith/__init__.py`, `src/ranksmith/strategies/__init__.py`, `src/ranksmith/providers/__init__.py`의 export 목록을 명시적으로 유지한다.
- `tests/`
  - 기존 테스트 import가 그대로 통과해야 한다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **공통 컴포넌트 식별 (Shared Components)**:
  - `validate_top_k(top_k)`
  - `validate_documents_max_chars(documents, max_document_chars)`
  - `ensure_model_client_method(model_client, method_name, label)`
  - JSON integer validator: `type(value) is int`
- **추상화 방안 (Abstraction Plan)**:
  - sync/async strategy를 억지로 하나의 base class로 합치지 않는다.
  - algorithm-specific flow는 각 method 모듈에 둔다.
  - 검증과 parser contract만 공유한다.
  - 결과 조립은 각 strategy에 남긴다. Listwise/Pairwise처럼 완전히 동일한 경우에만 후속 리팩터링에서 공통화를 검토한다.
  - provider는 vendor별 파일로 분리하고 `_providers.py`는 호환 layer로 남긴다.
  - evaluation/benchmark 모듈 이동은 후속 작업으로 분리한다. 이번 spec은 runtime reranking 구조에 집중한다.

## 5. 에러 핸들링 (Error Handling)
- `top_k < 0`
  - 모델 호출 전 `RerankInputError`.
- `max_document_chars < 1`, invalid config
  - 기존과 동일하게 `ValueError` 또는 기존 error type 유지.
- 문서 길이 초과
  - 모델 호출 전 `DocumentTooLongError`.
- ranking/selection에 `true`, `false` 포함
  - `RerankParseError`.
- ranking 중복, 누락, 범위 밖 값, 문자열 숫자
  - `RerankParseError`.
- provider capability 부족
  - `RerankInputError`.
- Azure SDK가 빈 content, 빈 choices, 비정상 response shape 반환
  - `RerankProviderError`.
- custom strategy 내부 임의 예외
  - 기존처럼 `RerankStrategyError` wrapping 유지.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**:
  - 기존 4개 sync strategy 결과가 리팩터링 전과 동일하다.
  - 기존 4개 async strategy 결과가 리팩터링 전과 동일하다.
  - 기존 root import가 모두 유지된다.
  - `_providers.py` 기존 import 경로가 유지된다.
- **엣지/실패 케이스 (Edge & Failure Cases)**:
  - `top_k=-1`은 provider/model client 호출 없이 `RerankInputError`.
  - `parse_ranking_response('{"ranking":[true,2]}')`는 `RerankParseError`.
  - `parse_selection_response('{"selected":[true]}')`는 `RerankParseError`.
  - MTEB parser도 bool rank를 invalid로 분류한다.
  - sync Azure provider에서 `choices=[]`이면 `RerankProviderError`.
  - async Azure provider에서 `choices=[]`이면 `RerankProviderError`.
  - sync Azure provider에서 `message.content is None`이면 기존처럼 `RerankProviderError`.
  - async Azure provider에서 `message.content is None`이면 기존처럼 `RerankProviderError`.
  - provider stub은 기존처럼 fast fail한다.
- **공통 Reranking Smoke/Benchmark**:
  - algorithm 순위 생성 로직 자체를 바꾸지 않으므로 fixture smoke 추가는 필수 아님.
  - 단, 리팩터링 중 ranking 결과가 바뀌지 않았음을 기존 `tests/test_ranksmith.py`, `tests/test_tourrank.py`, `tests/test_acurank.py`, `tests/test_async_providers.py`로 확인한다.
  - live provider benchmark는 실행하지 않는다.

---

## 7. 작업 태스크 추적 (Task Checklist)
> **코딩 어시스턴트 필수 지침**: 개발을 진행하면서 완료된 작업은 `[x]`로 표시하고, 필요시 하위 태스크를 추가하여 작업 내역을 관리하세요.

### Task 0: 승인 게이트

**Files:**
- Read: `docs/specs/spec_code_architecture_refactor.md`

- [x] **Step 0.1: 필수 컨텍스트 확인**
  - 확인한 문서:
    - `docs/wiki/00_context.md`
    - `docs/wiki/01_decisions.md`
    - `docs/wiki/02_architecture.md`
    - `docs/wiki/03_reference_processing.md`
    - `docs/wiki/04_references_index.md`
    - `docs/wiki/06_verification_policy.md`

- [x] **Step 0.2: 사용자 승인 확인**
  - 구현 전 사용자에게 본 spec 승인 여부를 확인한다.
  - 승인 문구 예: “승인”, “진행”, “이 spec대로 구현”.
  - 승인 전에는 `src/` 또는 `tests/`를 수정하지 않는다.

### Task 1: Regression Tests - Fast-fail 및 Parser Contract

**Files:**
- Modify: `tests/test_ranksmith.py`
- Modify: `tests/test_async_providers.py`
- Modify: `tests/test_tourrank.py`
- Modify: `tests/test_mteb_eval.py`

- [x] **Step 1.1: sync `top_k=-1` fast-fail 테스트 추가**
  - `tests/test_ranksmith.py`에 provider 호출 여부를 검증하는 테스트를 추가한다.
  - 테스트 이름: `test_negative_top_k_fast_fails_before_provider_call`
  - 기대: `RerankInputError` 발생, fake provider calls는 빈 리스트.
  - 실행: `uv run pytest tests/test_ranksmith.py::test_negative_top_k_fast_fails_before_provider_call -q`
  - 현재 기대: 실패. 이유는 기존 구현이 provider 호출 후 `top_k`를 검증하기 때문이다.

- [x] **Step 1.2: async `top_k=-1` fast-fail 테스트 추가**
  - `tests/test_async_providers.py`에 async provider 호출 여부를 검증하는 테스트를 추가한다.
  - 테스트 이름: `test_async_negative_top_k_fast_fails_before_provider_call`
  - 기대: `RerankInputError` 발생, fake provider calls는 빈 리스트.
  - 실행: `uv run pytest tests/test_async_providers.py::test_async_negative_top_k_fast_fails_before_provider_call -q`
  - 현재 기대: 실패.

- [x] **Step 1.3: ranking bool 거부 테스트 추가**
  - `tests/test_ranksmith.py`의 invalid ranking parametrized case에 `{"ranking": [true, 2]}`를 추가한다.
  - 실행: `uv run pytest tests/test_ranksmith.py::test_invalid_llm_ranking_fast_fails -q`
  - 현재 기대: 실패. 이유는 Python `bool`이 `int`로 통과하기 때문이다.

- [x] **Step 1.4: selection bool 거부 테스트 추가**
  - `tests/test_tourrank.py`에 `parse_selection_response('{"selected":[true]}', expected_count=2, selected_count=1)`가 `RerankParseError`를 내는 테스트를 추가한다.
  - 테스트 이름: `test_selection_parser_rejects_boolean_indexes`
  - 실행: `uv run pytest tests/test_tourrank.py::test_selection_parser_rejects_boolean_indexes -q`
  - 현재 기대: 실패.

- [x] **Step 1.5: MTEB parser bool invalid 테스트 추가**
  - `tests/test_mteb_eval.py`에 `parse_ranking_with_failure_type('{"ranking":[true,2]}', 2)`가 invalid를 반환하는 테스트를 추가한다.
  - 기대:
    - `valid is False`
    - `failure_type == "non_integer_rank"`
  - 실행: `uv run pytest tests/test_mteb_eval.py::test_parse_ranking_with_failure_type_rejects_boolean_rank -q`
  - 현재 기대: 실패.

### Task 2: Parser 및 Fast-fail 구현

**Files:**
- Modify: `src/ranksmith/parsing.py`
- Modify: `src/ranksmith/_mteb_eval.py`
- Modify: `src/ranksmith/azure.py`

- [x] **Step 2.1: JSON integer helper 추가**
  - `src/ranksmith/parsing.py`에 private helper를 추가한다.
  - 핵심 로직:
    ```python
    def _is_json_int(value: object) -> bool:
        return type(value) is int
    ```
  - `parse_ranking_response()`와 `parse_selection_response()`의 integer 검사에서 이 helper를 사용한다.

- [x] **Step 2.2: MTEB parser bool 거부 적용**
  - `src/ranksmith/_mteb_eval.py`의 `parse_ranking_with_failure_type()`에서 `isinstance(item, int)` 대신 `type(item) is int` 기준을 사용한다.
  - 실패 타입은 기존 `"non_integer_rank"`를 유지한다.

- [x] **Step 2.3: public reranker entry `top_k` 검증 추가**
  - `src/ranksmith/azure.py`의 `AzureOpenAIReranker.rerank()`와 `AsyncAzureOpenAIReranker.rerank()`에서 strategy 호출 전 `top_k`를 검증한다.
  - 에러 타입: `RerankInputError`.
  - 메시지: `"top_k must be greater than or equal to 0"`.

- [x] **Step 2.4: Task 1 테스트 통과 확인**
  - 실행:
    - `uv run pytest tests/test_ranksmith.py::test_negative_top_k_fast_fails_before_provider_call -q`
    - `uv run pytest tests/test_async_providers.py::test_async_negative_top_k_fast_fails_before_provider_call -q`
    - `uv run pytest tests/test_ranksmith.py::test_invalid_llm_ranking_fast_fails -q`
    - `uv run pytest tests/test_tourrank.py::test_selection_parser_rejects_boolean_indexes -q`
    - `uv run pytest tests/test_mteb_eval.py::test_parse_ranking_with_failure_type_rejects_boolean_rank -q`
  - 기대: 모두 통과.

### Task 3: Azure 응답 방어

**Files:**
- Modify: `tests/test_model_architecture.py`
- Modify: `src/ranksmith/_providers.py`

- [x] **Step 3.1: sync Azure malformed response 테스트 추가**
  - `tests/test_model_architecture.py`에 `choices=[]`가 `RerankProviderError`로 감싸지는 테스트를 추가한다.
  - 테스트 이름: `test_azure_aoai_provider_fast_fails_empty_choices`
  - 실행: `uv run pytest tests/test_model_architecture.py::test_azure_aoai_provider_fast_fails_empty_choices -q`
  - 현재 기대: 실패.

- [x] **Step 3.2: async Azure malformed response 테스트 추가**
  - `tests/test_model_architecture.py`에 async provider의 `choices=[]` 테스트를 추가한다.
  - 테스트 이름: `test_async_azure_aoai_provider_fast_fails_empty_choices`
  - 실행: `uv run pytest tests/test_model_architecture.py::test_async_azure_aoai_provider_fast_fails_empty_choices -q`
  - 현재 기대: 실패.

- [x] **Step 3.3: sync/async Azure response shape guard 구현**
  - `src/ranksmith/_providers.py`에서 `response.choices[0].message.content` 접근을 명시적 shape check로 감싼다.
  - `choices`가 비었거나 `message.content` 속성이 없으면 `RerankProviderError("Azure OpenAI returned an invalid response.")`로 실패한다.
  - `content is None` 또는 `content == ""`는 기존 empty response error를 유지한다.

- [x] **Step 3.4: Azure guard 테스트 통과 확인**
  - 실행:
    - `uv run pytest tests/test_model_architecture.py::test_azure_aoai_provider_fast_fails_empty_choices -q`
    - `uv run pytest tests/test_model_architecture.py::test_async_azure_aoai_provider_fast_fails_empty_choices -q`
  - 기대: 모두 통과.

### Task 4: Strategy 패키지 전환

**Files:**
- Delete: `src/ranksmith/strategies.py`
- Create: `src/ranksmith/strategies/__init__.py`
- Create: `src/ranksmith/strategies/_common.py`
- Create: `src/ranksmith/strategies/_listwise.py`
- Create: `src/ranksmith/strategies/_pairwise.py`
- Create: `src/ranksmith/strategies/_tourrank.py`
- Create: `src/ranksmith/strategies/_acurank.py`
- Modify: imports in `src/ranksmith/__init__.py` only if required by package conversion

- [x] **Step 4.1: 이동 단위 확정**
  - `src/ranksmith/strategies.py` 파일 삭제와 `src/ranksmith/strategies/` 디렉터리 생성을 같은 변경 단위로 처리한다.
  - 같은 경로에 `strategies.py`와 `strategies/`는 공존할 수 없다.

- [x] **Step 4.2: `_common.py` 생성**
  - 이동할 공통 요소:
    - `validate_top_k(top_k)`
    - document length validation helper
    - model client capability guard
  - 함수명은 기존 의미를 유지하되 `provider` 용어는 `model_client`로 정리한다.

- [x] **Step 4.3: `_listwise.py` 생성**
  - 이동 대상:
    - `_ListwiseConfigMixin`
    - `ListwiseStrategy`
    - `AsyncListwiseStrategy`
  - 기존 algorithm flow와 metadata를 바꾸지 않는다.
  - strategy 직접 사용자를 위해 model call 전 `validate_top_k(top_k)`를 호출한다.

- [x] **Step 4.4: `_pairwise.py` 생성**
  - 이동 대상:
    - `_PairwiseConfigMixin`
    - `PairwiseStrategy`
    - `AsyncPairwiseStrategy`
    - `_parse_pairwise_winner_response()` private helper
  - public `parse_pairwise_winner_response()`는 추가하지 않는다.
  - 기존 pair order 비교 순서와 swap 조건을 바꾸지 않는다.

- [x] **Step 4.5: `_tourrank.py` 생성**
  - 이동 대상:
    - `TourRankStageConfig`
    - `DEFAULT_TOURRANK_STAGE_CONFIGS`
    - `_TourRankConfigMixin`
    - `TourRankStrategy`
    - `AsyncTourRankStrategy`
  - group construction, shuffling seed, score aggregation semantics를 바꾸지 않는다.

- [x] **Step 4.6: `_acurank.py` 생성**
  - 이동 대상:
    - `_AcuRankConfigMixin`
    - `AcuRankStrategy`
    - `AsyncAcuRankStrategy`
    - `_AcuRankBatchRanking`
    - batch ranking helpers: `_chunks`, `_rank_and_apply_acurank_batches`, `_rank_and_apply_acurank_batches_async`, `_rank_acurank_batch`, `_rank_acurank_batch_async`
    - rating update helpers: `_apply_acurank_batch_ranking`, `_update_acurank_ratings`
    - probability helpers: `_has_call_budget`, `_acurank_probability_order`, `_acurank_topk_probabilities`, `_find_acurank_threshold`, `_normal_right_tail`
  - TrueSkill update, prior score handling, adaptive call budget semantics를 바꾸지 않는다.
  - helper 섹션 순서는 batch ranking, rating update, probability 순서로 유지한다.

- [x] **Step 4.7: `strategies/__init__.py` export 유지**
  - 기존 public import가 모두 유지되도록 아래 이름을 export한다.
  - 대상:
    - `ListwiseStrategy`, `AsyncListwiseStrategy`
    - `PairwiseStrategy`, `AsyncPairwiseStrategy`
    - `TourRankStrategy`, `AsyncTourRankStrategy`, `TourRankStageConfig`
    - `AcuRankStrategy`, `AsyncAcuRankStrategy`
  - `__all__`에 동일 이름을 명시한다.

- [x] **Step 4.8: strategy regression 테스트 실행**
  - 실행:
    - `uv run pytest tests/test_ranksmith.py tests/test_async_providers.py tests/test_tourrank.py tests/test_acurank.py -q`
  - 기대: 모두 통과.

### Task 5: Provider 패키지 전환

**Files:**
- Create: `src/ranksmith/providers/__init__.py`
- Create: `src/ranksmith/providers/_azure.py`
- Create: `src/ranksmith/providers/_stubs.py`
- Modify: `src/ranksmith/_providers.py`
- Modify: `src/ranksmith/__init__.py`
- Modify: `tests/test_model_architecture.py`

- [x] **Step 5.1: `_azure.py` 생성**
  - `AzureAOAIProvider`, `AsyncAzureAOAIProvider`, `_extract_usage`, `_to_openai_messages`를 이동한다.
  - Task 3에서 보강한 malformed response guard를 그대로 보존한다.

- [x] **Step 5.2: `_stubs.py` 생성**
  - 이동 대상:
    - `OpenAIProvider`, `AsyncOpenAIProvider`
    - `AnthropicProvider`, `AsyncAnthropicProvider`
    - `GeminiProvider`, `AsyncGeminiProvider`
  - 기존 fast fail message를 바꾸지 않는다.

- [x] **Step 5.3: provider export 호환 유지**
  - `src/ranksmith/providers/__init__.py`에서 provider class를 re-export하고 `__all__`을 명시한다.
  - `src/ranksmith/_providers.py`는 기존 import 경로 호환용 re-export layer로 축소한다.
  - `src/ranksmith/__init__.py`의 root import 표면을 유지한다.

- [x] **Step 5.4: provider regression 테스트 실행**
  - 실행:
    - `uv run pytest tests/test_model_architecture.py -q`
  - 기대: 통과.

### Task 6: Import 정리 및 Public Import 회귀 확인

**Files:**
- Read: `tests/test_model_architecture.py`
- Read: `tests/test_examples.py`

- [x] **Step 6.1: import 정리**
  - 파일 이동 후 사용하지 않는 import와 순서가 어긋난 import를 정리한다.
  - 자동 수정은 `uv run ruff check . --fix`까지만 허용한다.
  - formatter는 최종 `./scripts/verify.sh`의 `ruff format --check` 결과를 기준으로 한다.

- [x] **Step 6.2: public import 테스트 실행**
  - 실행:
    - `uv run pytest tests/test_model_architecture.py::test_model_api_is_publicly_importable -q`
    - `uv run pytest tests/test_model_architecture.py::test_removed_llm_provider_names_are_not_public -q`
  - 기대: 통과.

- [x] **Step 6.3: example 테스트 실행**
  - 실행:
    - `uv run pytest tests/test_examples.py -q`
  - 기대: 통과.

- [x] **Step 6.4: focused regression 실행**
  - 실행:
    - `uv run pytest tests/test_ranksmith.py tests/test_async_providers.py tests/test_tourrank.py tests/test_acurank.py tests/test_model_architecture.py tests/test_mteb_eval.py -q`
  - 기대: 통과.

### Task 7: 문서 정리

**Files:**
- Modify: `docs/wiki/02_architecture.md`
- Read: `docs/wiki/08_custom_strategy_extension.md`
- Modify when internal strategy module paths are mentioned: `docs/wiki/08_custom_strategy_extension.md`
- Modify: `docs/specs/spec_code_architecture_refactor.md`

- [x] **Step 7.1: architecture 문서 업데이트**
  - `docs/wiki/02_architecture.md`에 실제 파일 구조를 반영한다.
  - 반영 내용:
    - `strategies/` 패키지의 method별 모듈
    - `providers/` 패키지와 `_providers.py` compatibility layer
    - public API 표면은 유지된다는 점

- [x] **Step 7.2: custom strategy 확장 문서 확인**
  - `docs/wiki/08_custom_strategy_extension.md`의 import 예제가 새 구조에서도 맞는지 확인한다.
  - root import 기반 예제가 유지되면 문서 변경은 하지 않는다.
  - `ranksmith.strategies` 내부 모듈 경로를 직접 안내하는 문장이 있으면 새 구조에 맞게 수정한다.

- [x] **Step 7.3: spec 체크리스트 갱신**
  - 완료된 항목은 이 문서에서 `[x]`로 표시한다.
  - 모든 구현과 검증이 끝난 뒤 상태를 `[ ] Draft | [ ] In Progress | [x] Completed`로 바꾼다.

### Task 8: 최종 검증

**Files:**
- Read: `scripts/verify.sh`
- Build artifacts may change under `dist/`

- [x] **Step 8.1: 전체 검증 실행**
  - 실행:
    - `./scripts/verify.sh`
  - 기대:
    - pytest 전체 통과
    - ruff check 통과
    - ruff format check 통과
    - mypy 통과
    - build 성공

- [x] **Step 8.2: git 상태 확인**
  - 실행:
    - `git status --short`
  - 기대:
    - 의도한 source/test/docs 변경만 표시된다.
    - 기존 untracked Setwise PDF는 이 작업 범위 밖으로 유지된다.
