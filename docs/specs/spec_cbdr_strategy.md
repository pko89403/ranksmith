# Spec: CBDR Strategy

## 1. 개요 (Overview)
- **작업 목적**: query-only confidence가 충분히 높으면 context reranking을 건너뛰고, 낮으면 `Conf(Q+C)-Conf(Q)` confidence gain으로 documents를 rerank하는 CBDR 본체 Strategy를 추가한다.
- **Reference**:
  - `docs/wiki/references/parametric_post_retrieval_confidence.md`
  - `docs/specs/spec_confidence_runtime_readiness.md`
  - `docs/specs/spec_confidence_gain_reranking.md`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

이 스펙은 논문 원형을 그대로 재현하지 않는다.

논문 원형의 CBDR은 query-only confidence가 높으면 retrieval 자체를 skip한다.
ranksmith는 retriever, indexer, vector search를 소유하지 않으므로 true pre-retrieval skip은 구현하지 않는다.
이번 스펙은 **reranking-side CBDR router**만 구현한다.

즉, 이미 documents가 전달된 `rerank()` 호출 안에서 context reranking을 skip할 수 있다.
외부 retriever가 retrieval 호출 전에 사용할 query-only decision API는 이번 범위에 포함하지 않는다.

핵심 목표:

```text
base = Conf(Q)

if base >= skip_threshold:
  context scoring과 reranking을 수행하지 않음
  original order를 보존하고 cbdr_skipped metadata를 남김
else:
  after_i = Conf(Q + C_i)
  gain_i = after_i - base
  documents sorted by gain_i desc
```

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 포함 범위
- sync `CBDRStrategy` 추가.
- query-only confidence 기반 skip decision 추가.
- low-confidence query에서 confidence gain reranking 수행.
- `ConfidenceGainStrategy`와 공유 가능한 계산/검증 로직 정리.
- `AzureOpenAIReranker` sync facade에서 built-in strategy로 처리.
- `ranksmith.strategies` submodule export 추가.
- README/README.ko 및 architecture wiki 문서 반영.

### 제외 범위
- retriever integration.
- vector search/indexing.
- upstream retrieval 호출 자체를 중단하는 orchestration API.
- query-only `should_retrieve()` helper.
- async `CBDRStrategy`.
- closed model provider 병렬 호출.
- answer generation cache.
- confidence scorer 학습 또는 artifact 생성.
- benchmark 수치 README 반영.
- CBDR threshold 자동 튜닝.
- reranker fine-tuning.

### 입력 (Inputs)

```python
CBDRStrategy(
    base_estimator: ConfidenceEstimator,
    context_estimator: ConfidenceEstimator,
    answer_generator: AnswerGenerator,
    skip_threshold: float = 0.8,
    max_document_chars: int = 4000,
    algorithm: Literal["cbdr"] = "cbdr",
)
```

필수 task type:

```text
base_estimator.task_type == "query_answerability_confidence"
context_estimator.task_type == "query_context_answerability_confidence"
```

`answer_generator`는 기존 `ConfidenceGainStrategy`와 같은 contract를 사용한다.

```python
class AnswerGenerator(Protocol):
    def answer_query(self, query: str) -> str: ...
    def answer_with_context(self, query: str, context: str) -> str: ...
```

`model_client`는 `RerankStrategy` protocol 때문에 `rerank()` 인자로 받지만 사용하지 않는다.
answer generation은 `answer_generator` 책임이다.
CBDRStrategy는 `model_client` capability를 검사하지 않는다.

### 출력 (Outputs)

#### skip path

`Conf(Q) >= skip_threshold`이면 context answer generation과 context confidence scoring을 수행하지 않는다.
입력 documents의 original order를 보존해 `RerankResult`를 반환한다.
rank는 반환 결과 기준 1-based로 부여한다.
original_index는 입력 documents 기준 0-based를 보존한다.

```python
{
    "strategy": "cbdr",
    "algorithm": "cbdr",
    "cbdr_skipped": True,
    "base_confidence": 0.91,
    "skip_threshold": 0.8,
    "context_confidence": None,
    "confidence_gain": None,
}
```

#### rerank path

`Conf(Q) < skip_threshold`이면 confidence gain으로 정렬한다.
rank는 반환 결과 기준 1-based로 부여한다.
original_index는 입력 documents 기준 0-based를 보존한다.

```python
{
    "strategy": "cbdr",
    "algorithm": "cbdr",
    "cbdr_skipped": False,
    "base_confidence": 0.42,
    "skip_threshold": 0.8,
    "context_confidence": 0.81,
    "confidence_gain": 0.39,
}
```

### 제약 사항 (Constraints)
- hidden state, logits, attention에 직접 접근하지 않는다.
- runtime Strategy는 scorer artifact를 학습하지 않는다.
- `Conf(Q)`는 query당 한 번만 계산한다.
- 빈 documents 또는 `top_k == 0`이면 base answer generation과 confidence scoring을 수행하지 않고 `[]`를 반환한다.
- skip path에서는 `answer_with_context()`와 `context_estimator.score()`를 호출하지 않는다.
- skip path에서는 document text를 context로 읽지 않으므로 `max_document_chars` 검증도 수행하지 않는다.
- rerank path에서는 문서별로 `Conf(Q+C_i)`를 계산한다.
- skip 여부를 metadata에 반드시 남긴다.
- skip path에서 original order를 조용히 수정하지 않는다.
- rerank path에서 gain 동점은 original index 오름차순으로 안정 정렬한다.
- `top_k > 0`은 skip/rerank path 모두 최종 결과에 slicing으로만 적용한다.
- score는 finite float이며 `[0, 1]` 범위여야 한다.
- `skip_threshold`는 finite float이며 `[0, 1]` 범위여야 한다.
- bool은 numeric score 또는 `skip_threshold`로 인정하지 않는다.
- `skip_threshold=0.0`이면 non-empty documents에서 항상 skip된다.
- `skip_threshold=1.0`이면 `base_confidence == 1.0`일 때만 skip된다.
- skip path는 answer generation 1회와 confidence scoring 1회를 수행한다.
- rerank path는 문서 수가 `N`이면 answer generation `N + 1`회와 confidence scoring `N + 1`회를 수행한다.
- rerank path에서는 `top_k`가 작아도 모든 문서를 scoring한 뒤 slicing한다.
- confidence scoring 실패 시 기존 ranking으로 fallback하지 않는다.
- rerank path에서 `max_document_chars`를 초과하면 `DocumentTooLongError`로 실패한다.
- metadata에 answer text, scorer artifact path, HuggingFace token, raw hidden state, raw feature vector를 넣지 않는다.
- root import 확장은 사용자 승인 없이는 하지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 동작 메커니즘
1. query와 `top_k`를 검증한다.
2. `answer_generator.answer_query(query)`로 base answer를 생성한다.
3. `base_estimator.score(QueryAnswerabilityConfidenceInput(...))`로 `Conf(Q)`를 계산한다.
4. `base_confidence >= skip_threshold`이면 skip path로 간다.
5. skip path는 document length 검증과 context 호출 없이 original order를 보존해 결과를 만든다.
6. `base_confidence < skip_threshold`이면 rerank path로 간다.
7. rerank path에서 document length를 검증한다.
8. 문서별 `answer_with_context(query, document.text)`를 호출한다.
9. 문서별 `Conf(Q+C_i)`를 계산한다.
10. `gain_i = context_confidence_i - base_confidence`를 계산한다.
11. gain 내림차순, original index 오름차순으로 정렬한다.

### 의사 알고리즘 (Pseudo-algorithm)

```text
cbdr_rerank(query, documents, top_k):
  validate query, top_k

  if documents is empty or top_k == 0:
    return []

  base_answer = answer_generator.answer_query(query)
  base = score_base(query, base_answer)

  if base >= skip_threshold:
    results = original_order_results(
      cbdr_skipped=True,
      base_confidence=base,
      context_confidence=None,
      confidence_gain=None,
    )
    if top_k is not None:
      results = results[:top_k]
    assign rank from 1 over returned results
    return results

  validate document lengths
  scored = []
  for each document with original_index:
    context_answer = answer_generator.answer_with_context(query, document.text)
    context = score_context(query, document.text, context_answer)
    gain = context - base
    scored.append(original_index, context, gain)

  scored = sort scored by (-gain, original_index)
  results = scored_results(scored)
  if top_k is not None:
    results = results[:top_k]
  assign rank from 1 over returned results
  return results
```

### 의사 코드 (Pseudo-code)

```python
from ranksmith.strategies import CBDRStrategy

strategy = CBDRStrategy(
    base_estimator=query_estimator,
    context_estimator=query_context_estimator,
    answer_generator=answer_generator,
    skip_threshold=0.8,
)

reranker = AzureOpenAIReranker(
    model_client=model_client,
    strategy=strategy,
)

results = reranker.rerank(query, documents, top_k=10)
```

### 통합 지점 (Integration Points)

Strategy:
- `src/ranksmith/strategies/_cbdr.py`
- `src/ranksmith/strategies/_confidence_gain.py`
- `src/ranksmith/strategies/__init__.py`

Facade:
- `src/ranksmith/azure.py`

Tests:
- `tests/test_cbdr_strategy.py`

Docs:
- `docs/wiki/02_architecture.md`
- `docs/specs/spec_cbdr_strategy.md`
- `README.md`
- `README.ko.md`

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### Metadata contract
공통 confidence key:

```python
{
    "base_confidence": float,
    "context_confidence": float | None,
    "confidence_gain": float | None,
}
```

CBDR 전용 key:

```python
{
    "strategy": "cbdr",
    "algorithm": "cbdr",
    "cbdr_skipped": bool,
    "skip_threshold": float,
}
```

metadata에는 다음 값을 넣지 않는다.
- generated answer text
- scorer artifact path
- HuggingFace token
- raw hidden state
- raw structural feature vector

### 공통 컴포넌트 식별 (Shared Components)
- `AnswerGenerator` protocol은 `ConfidenceGainStrategy`와 공유한다.
- `ConfidenceEstimator` protocol은 `ConfidenceGainStrategy`와 공유한다.
- task type 상수는 중복 정의하지 않는다.
- answer validation과 confidence score validation은 공유 helper로 분리한다.
- context gain 계산은 CBDR과 confidence gain strategy가 같은 helper를 사용한다.

### 추상화 방안 (Abstraction Plan)
- `src/ranksmith/strategies/_confidence_gain.py`에 이미 있는 공통 요소를 private helper로 정리한다.
- `CBDRStrategy`는 `ConfidenceGainStrategy`를 상속하지 않는다.
  - 이유: skip path가 있는 router와 pure confidence gain reranker는 의미가 다르다.
- 대신 shared helper를 사용해 검증, answer 호출, confidence gain 계산의 중복을 줄인다.

권장 구조:

```text
_confidence_gain.py
  AnswerGenerator
  ConfidenceEstimator
  ConfidenceGainResult
  validate_answer
  validate_confidence_score
  score_base_answerability
  score_context_answerability

_cbdr.py
  CBDRStrategy
```

## 5. 에러 핸들링 (Error Handling)
- 빈 query: `RerankInputError`
- 빈 documents: `[]`
- `top_k == 0`: `[]`
- `top_k < 0`: `RerankInputError`
- `skip_threshold`가 finite probability가 아니거나 bool이면 `ValueError`
- `max_document_chars < 1`: `ValueError`
- low-confidence rerank path에서 document length 초과: `DocumentTooLongError`
- base/context estimator task mismatch: `RerankInputError`
- answer generator가 빈 answer 반환: `RerankProviderError`
- answer generator가 예외 발생: `RerankProviderError`로 래핑
- direct `CBDRStrategy.rerank()`에서 confidence estimator의 기존 `RerankError`는 그대로 전파
- direct `CBDRStrategy.rerank()`에서 confidence estimator의 unexpected exception은 그대로 전파
- `AzureOpenAIReranker` facade에서 built-in strategy의 unexpected exception은 `RerankProviderError`로 래핑
- confidence score가 finite probability가 아님: `RerankStrategyError`
- gain이 finite float가 아님: `RerankStrategyError`
- output ranking 보정 필요 상황: 조용히 보정하지 않고 실패

## 6. 테스트 계획 (Test Plan)

### 성공 케이스 (Happy Paths)
- 빈 documents이면 `[]` 반환.
- 빈 documents에서 `answer_query()`와 estimator가 호출되지 않음.
- `top_k == 0`이면 `[]` 반환.
- `top_k == 0`에서 `answer_query()`와 estimator가 호출되지 않음.
- `base_confidence >= skip_threshold`이면 original order 보존.
- skip path rank는 반환 결과 기준 1-based.
- skip path original_index는 입력 documents 기준 0-based.
- skip path에서 `answer_with_context()`가 호출되지 않음.
- skip path에서 `context_estimator.score()`가 호출되지 않음.
- skip path metadata에 `cbdr_skipped=True` 기록.
- skip path에서 `top_k` slicing 적용.
- skip path에서 긴 document가 있어도 `DocumentTooLongError`가 발생하지 않음.
- `base_confidence < skip_threshold`이면 confidence gain으로 rerank.
- rerank path metadata에 `cbdr_skipped=False`, base/context/gain 기록.
- gain 동점 시 original index 유지.
- rerank path에서 `top_k`가 작아도 모든 문서를 scoring한 뒤 slicing.
- `skip_threshold=0.0`이면 non-empty documents에서 skip.
- `skip_threshold=1.0`이면 `base_confidence == 1.0`일 때만 skip.
- `AzureOpenAIReranker` facade에서 `CBDRStrategy` 예외 surface가 built-in strategy와 동일함.

### 엣지/실패 케이스 (Edge & Failure Cases)
- 잘못된 `skip_threshold` 실패.
- bool `skip_threshold` 실패.
- 잘못된 estimator task type 실패.
- 빈 query 실패.
- low-confidence rerank path에서만 긴 document 실패.
- answer generator empty output 실패.
- base confidence score 범위 밖 실패.
- context confidence score 범위 밖 실패.
- base scoring 실패가 skip fallback으로 숨겨지지 않음.
- context scoring 실패가 original order fallback으로 숨겨지지 않음.

### 공통 Reranking Smoke/Benchmark
- fake answer generator와 fake confidence estimator로 deterministic smoke test를 추가한다.
- artifact load 기반 skip path E2E smoke를 추가해 `from_artifact()` -> `CBDRStrategy` -> `AzureOpenAIReranker` 경로를 검증한다.
- artifact load 기반 rerank path E2E smoke를 추가해 `from_artifact()` -> `CBDRStrategy` -> `AzureOpenAIReranker` 경로를 검증한다.
- 실제 closed model live test는 credential/cost 때문에 opt-in으로 분리한다.
- README benchmark 수치는 추가하지 않는다.

검증 명령:

```bash
uv run pytest tests/test_cbdr_strategy.py tests/test_confidence_gain_strategy.py -q
./scripts/verify.sh
```

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] CBDR을 ranksmith에서 reranking-side router로 정의
- [x] 스펙 문서 작성

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/strategies/_confidence_gain.py`: 공통 protocol/helper 정리
- [x] `src/ranksmith/strategies/_cbdr.py`: `CBDRStrategy` 구현
- [x] `src/ranksmith/strategies/__init__.py`: strategy export 추가
- [x] `src/ranksmith/azure.py`: built-in sync strategy 처리 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_cbdr_strategy.py`: skip path 정상 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: rerank path 정상 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: 엣지/실패 케이스 추가
- [x] `tests/test_cbdr_strategy.py`: Azure facade smoke 추가
- [x] `tests/test_cbdr_strategy.py`: artifact load 기반 skip path E2E smoke 추가
- [x] `tests/test_cbdr_strategy.py`: artifact load 기반 rerank path E2E smoke 추가
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] `docs/wiki/02_architecture.md`: CBDR strategy 위치 추가
- [x] `README.md` / `README.ko.md`: benchmark 없는 usage 문서 추가
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
