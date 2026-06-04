# Spec: Confidence Gain Reranking

## 1. 개요 (Overview)
- **작업 목적**: post-retrieval context가 closed model의 answerability confidence를 얼마나 올리는지 계산하고, 그 변화량으로 documents를 rerank한다.
- **Reference**:
  - `docs/wiki/references/parametric_post_retrieval_confidence.md`
  - `docs/wiki/references/structural_confidence.md`
  - `docs/specs/spec_structural_confidence.md`
  - `docs/specs/spec_confidence_runtime_readiness.md`
  - `docs/specs/spec_confidence_training_pipeline.md`
  - `docs/specs/spec_confidence_generation_pipeline.md`
- **상태**: `[x] Draft` | `[ ] In Progress` | `[ ] Completed`

이 스펙은 논문 원형을 그대로 재현하지 않는다.

논문 원형은 target LLM의 hidden state 기반 confidence detector와 reranker fine-tuning을 사용한다.
ranksmith는 closed model API와 runtime training-free reranking을 지향하므로, 기존 structural confidence module을 confidence proxy로 사용한다.
runtime reranking은 training-free다.
단, confidence scorer artifact는 사전에 generation/training pipeline으로 준비되어 있어야 한다.

핵심 목표:

```text
base = Conf(Q)
after_i = Conf(Q + C_i)
gain_i = after_i - base
documents sorted by gain_i desc
```

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 포함 범위
- query-only answerability confidence task 추가.
- query+context answerability confidence task 추가.
- confidence generation pipeline에서 두 task의 canonical JSONL 생성 지원.
- confidence training pipeline에서 두 task 학습 지원.
- confidence runtime에서 두 task inference 지원.
- confidence gain 계산 utility 추가.
- confidence gain 기반 sync reranking Strategy 추가.

### 제외 범위
- CBDR retrieval skip 구현.
- retriever integration.
- vector search/indexing.
- async confidence gain strategy.
- closed model provider 병렬 호출.
- answer generation cache.
- reranker fine-tuning.
- benchmark 수치 README 반영.
- semantic feature fusion.
- hidden state, logits, attention 직접 사용.

CBDR은 이 스펙의 직접 범위가 아니다.
다만 `Conf(Q)`와 `gain_i`를 metadata로 남겨 후속 CBDR spec에서 재사용 가능하게 한다.

### 입력 (Inputs)

#### 새 confidence task type
기존:

```python
TaskType = Literal["answer_confidence", "judgment_confidence"]
```

확장:

```python
TaskType = Literal[
    "answer_confidence",
    "judgment_confidence",
    "query_answerability_confidence",
    "query_context_answerability_confidence",
]
```

#### 새 runtime input type

```python
@dataclass(frozen=True)
class QueryAnswerabilityConfidenceInput:
    query: str
    answer: str

@dataclass(frozen=True)
class QueryContextAnswerabilityConfidenceInput:
    query: str
    context: str
    answer: str
```

`answer`는 closed model이 생성한 답변이다.
confidence scorer는 “이 입력 조건에서 생성된 answer가 맞을 가능성”을 추정한다.

#### generation raw JSONL

Query-only answerability input:

```json
{
  "id": "sample-1::base",
  "query": "who played karen in married to the mob?",
  "gold_answer": "Nancy Travis",
  "source": "nq",
  "group_id": "sample-1",
  "metadata": {}
}
```

Query+context answerability input:

```json
{
  "id": "sample-1::doc-1",
  "query": "who played karen in married to the mob?",
  "context": "Angela de Marco ... Karen (Nancy Travis).",
  "gold_answer": "Nancy Travis",
  "source": "nq",
  "group_id": "sample-1",
  "metadata": {}
}
```

### 출력 (Outputs)

#### canonical JSONL

`query_answerability_confidence`:

```json
{
  "id": "sample-1::base",
  "task_type": "query_answerability_confidence",
  "query": "...",
  "answer": "Nancy Travis",
  "gold_answer": "Nancy Travis",
  "label": 1,
  "source": "nq",
  "group_id": "sample-1",
  "metadata": {
    "generation": {},
    "input_metadata": {}
  }
}
```

`query_context_answerability_confidence`:

```json
{
  "id": "sample-1::doc-1",
  "task_type": "query_context_answerability_confidence",
  "query": "...",
  "context": "...",
  "answer": "Nancy Travis",
  "gold_answer": "Nancy Travis",
  "label": 1,
  "source": "nq",
  "group_id": "sample-1",
  "metadata": {
    "generation": {},
    "input_metadata": {}
  }
}
```

#### runtime confidence gain result

```python
@dataclass(frozen=True)
class ConfidenceGainResult:
    base_score: float
    context_score: float
    gain: float
    base_result: StructuralConfidenceResult
    context_result: StructuralConfidenceResult
```

#### reranking result metadata

`ConfidenceGainStrategy`가 반환하는 `RerankResult.metadata`:

```python
{
    "strategy": "confidence_gain",
    "algorithm": "confidence_gain",
    "base_confidence": 0.42,
    "context_confidence": 0.81,
    "confidence_gain": 0.39,
}
```

### 제약 사항 (Constraints)
- hidden state, logits, attention에 직접 접근하지 않는다.
- runtime Strategy는 scorer artifact를 학습하지 않는다.
- scorer artifact는 task type별로 별도 사용한다.
- `Conf(Q)`는 query당 한 번만 계산한다.
- `Conf(Q+C_i)`는 문서별로 계산한다.
- runtime reranking은 기본적으로 answer generation `N + 1`회와 confidence scoring `N + 1`회를 수행한다.
- 호출 수를 조용히 줄이기 위해 일부 문서를 skip하지 않는다.
- score는 finite float이며 `[0, 1]` 범위여야 한다.
- gain은 `[-1, 1]` 범위의 finite float이어야 한다.
- gain 동점은 original index 오름차순으로 안정 정렬한다.
- confidence scoring 실패 시 기존 ranking으로 fallback하지 않는다.
- `top_k`는 정렬 후 slicing만 수행한다.
- `max_document_chars`를 초과하면 `DocumentTooLongError`로 실패한다.
- root import 확장은 사용자 승인 없이는 하지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 동작 메커니즘
1. closed model generation pipeline이 query-only 답변을 생성한다.
2. 같은 pipeline이 query+context 답변을 문서별로 생성한다.
3. gold answer와 normalized exact match로 canonical label을 만든다.
4. training pipeline이 task type별 scorer artifact를 학습한다.
5. runtime에서 base estimator와 context estimator를 로드한다.
6. Strategy 호출 시 base answer를 생성하거나 caller가 제공한다.
7. Strategy 호출 시 각 context answer를 생성하거나 caller가 제공한다.
8. base confidence와 context confidence를 계산한다.
9. `gain = context_confidence - base_confidence`를 계산한다.
10. gain 내림차순, original index 오름차순으로 정렬한다.

### Runtime answer 생성 정책
confidence gain에는 answer text가 필요하다.

1차 구현은 Strategy 내부에서 closed model answer generation까지 수행하지 않는다.

대신 Strategy는 answer generator hook을 받는다.

```python
AnswerGenerator = Protocol:
    def answer_query(self, query: str) -> str: ...
    def answer_with_context(self, query: str, context: str) -> str: ...
```

이렇게 분리하면 confidence reranking과 answer generation 책임이 섞이지 않는다.
기존 `ModelClient.rank/compare/select` 계약도 변경하지 않는다.

호출 비용:
- 문서 수가 `N`이면 `answer_query` 1회, `answer_with_context` `N`회를 호출한다.
- confidence scoring도 base 1회, context `N`회를 수행한다.
- 호출 수 절감을 위한 cache/batching은 이번 범위에서 제외한다.

### 의사 알고리즘 (Pseudo-algorithm)

```text
confidence_gain_rerank(query, documents):
  validate inputs
  base_answer = answer_generator.answer_query(query)
  base_score = base_estimator.score(QueryAnswerabilityInput(query, base_answer))

  scored = []
  for each document with original_index:
    context_answer = answer_generator.answer_with_context(query, document.text)
    context_score = context_estimator.score(
      QueryContextAnswerabilityInput(query, document.text, context_answer)
    )
    gain = context_score - base_score
    scored.append(original_index, context_score, gain)

  order = sort scored by (-gain, original_index)
  return RerankResult list
```

### 의사 코드 (Pseudo-code)

```python
strategy = ConfidenceGainStrategy(
    base_estimator=query_estimator,
    context_estimator=query_context_estimator,
    answer_generator=answer_generator,
)

results = strategy.rerank(
    query="who played karen in married to the mob?",
    documents=documents,
    model_client=model_client,
    top_k=5,
)
```

`model_client`는 Strategy protocol 때문에 인자로 받지만, 1차 구현에서는 사용하지 않는다.
대신 `answer_generator`가 answer generation 책임을 가진다.

### 통합 지점 (Integration Points)

Confidence runtime:
- `src/ranksmith/confidence/_types.py`
- `src/ranksmith/confidence/_templates.py`
- `src/ranksmith/confidence/_scorer.py`
- `src/ranksmith/confidence/_structural.py`
- `src/ranksmith/confidence/__init__.py`

Confidence generation:
- `src/ranksmith/confidence_generation/_types.py`
- `src/ranksmith/confidence_generation/_io.py`
- `src/ranksmith/confidence_generation/_prompts.py`
- `src/ranksmith/confidence_generation/_pipeline.py`
- `src/ranksmith/confidence_generation/__init__.py`

Confidence training:
- `src/ranksmith/confidence_training/_types.py`
- `src/ranksmith/confidence_training/_dataset.py`
- `src/ranksmith/confidence_training/_features.py`
- `src/ranksmith/confidence_training/_artifact.py`

Strategy:
- `src/ranksmith/strategies/_confidence_gain.py`
- `src/ranksmith/strategies/__init__.py`

Docs:
- `docs/wiki/02_architecture.md`
- `docs/wiki/04_references_index.md`
- `README.md`
- `README.ko.md`

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### 공통 컴포넌트 식별 (Shared Components)
- normalized exact match labeling은 기존 confidence generation labeling helper를 재사용한다.
- canonical JSONL validation은 기존 confidence training dataset validator 패턴을 확장한다.
- structural feature extraction은 기존 `structural-v1`을 그대로 재사용한다.
- batch scoring은 기존 `StructuralConfidenceEstimator.score_batch()`를 재사용한다.

### 추상화 방안 (Abstraction Plan)
- answer generation은 Strategy 내부 private prompt로 직접 만들지 않는다.
- `AnswerGenerator` protocol로 분리한다.
- confidence gain 계산은 Strategy와 분리된 utility로 둔다.

후속 CBDR spec은 같은 utility를 사용해 `Conf(Q) >= beta` retrieval skip을 구현할 수 있다.

## 5. 에러 핸들링 (Error Handling)
- 빈 query: `RerankInputError`
- 빈 documents: `[]`
- `top_k < 0`: `RerankInputError`
- document length 초과: `DocumentTooLongError`
- base/context estimator task mismatch: `RerankInputError`
- answer generator가 빈 answer 반환: `RerankProviderError`
- answer generator가 예외 발생: `RerankProviderError`로 래핑
- confidence estimator score 실패: 해당 confidence error를 감싸지 않고 전파
- confidence score가 finite probability가 아님: `ConfidenceArtifactError`
- gain이 finite float가 아님: `RerankStrategyError`
- output ranking 보정 필요 상황: 조용히 보정하지 않고 실패

## 6. 테스트 계획 (Test Plan)

### 성공 케이스 (Happy Paths)
- query-only input template 생성.
- query+context input template 생성.
- 새 task type scorer metadata validation 성공.
- canonical JSONL validation 성공.
- base/context confidence로 gain 계산.
- gain 내림차순 정렬.
- gain 동점 시 original index 유지.
- `top_k` slicing.
- metadata에 base/context/gain 기록.
- answer generator 호출 수가 `N + 1`인지 확인.

### 엣지/실패 케이스 (Edge & Failure Cases)
- unsupported task type 실패.
- task type과 input dataclass mismatch 실패.
- scorer artifact task mismatch 실패.
- 빈 query/context/answer 실패.
- answer generator empty output 실패.
- context estimator를 query-only task로 주입하면 실패.
- base estimator를 query+context task로 주입하면 실패.
- score 범위 밖 실패.
- `max_document_chars` 초과 실패.

### 공통 Reranking Smoke/Benchmark
- synthetic answer generator와 fake confidence estimator로 deterministic reranking smoke test를 추가한다.
- 실제 LLM live test는 credential/cost 때문에 opt-in으로 분리한다.
- README benchmark 수치는 추가하지 않는다.

검증 명령:

```bash
uv run pytest tests/test_confidence_*.py tests/test_confidence_generation_*.py tests/test_confidence_training_*.py tests/test_confidence_gain_strategy.py -q
uv run ruff check src/ranksmith tests
uv run mypy src/ranksmith tests
./scripts/verify.sh
```

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] 스펙 문서(본 문서) 상의 의사 코드 설계 검토 및 확정

### Phase 2: Confidence task 확장
- [ ] `src/ranksmith/confidence/_types.py`: query-only/contextual answerability input type 추가
- [ ] `src/ranksmith/confidence/_templates.py`: 새 input template 추가
- [ ] `src/ranksmith/confidence/_scorer.py`: metadata task validation 확장
- [ ] `src/ranksmith/confidence/__init__.py`: submodule export 추가
- [ ] `tests/test_confidence_*.py`: 새 task runtime tests 추가

### Phase 3: Generation/Training 확장
- [ ] `src/ranksmith/confidence_generation/*`: answerability generation config/pipeline 추가
- [ ] `src/ranksmith/confidence_training/*`: canonical schema/task validation 확장
- [ ] `tests/test_confidence_generation_*.py`: 새 generation tests 추가
- [ ] `tests/test_confidence_training_*.py`: 새 training tests 추가

### Phase 4: Confidence gain Strategy
- [ ] `src/ranksmith/strategies/_confidence_gain.py`: `ConfidenceGainStrategy` 구현
- [ ] `src/ranksmith/strategies/__init__.py`: strategy export 추가
- [ ] `tests/test_confidence_gain_strategy.py`: deterministic unit tests 추가
- [ ] 필요 시 `tests/fixtures/reranking_smoke_fixture.jsonl` 기반 smoke test 추가

### Phase 5: 문서 및 검증
- [ ] `docs/wiki/02_architecture.md`: confidence gain strategy 위치 추가
- [ ] `docs/wiki/04_references_index.md`: reference 상태 갱신
- [ ] `README.md` / `README.ko.md`: benchmark 없는 usage 문서 추가
- [ ] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인
- [ ] 본 문서 최상단의 **상태**를 `Completed`로 변경
