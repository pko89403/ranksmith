# Spec: Pairwise Ranking Prompting

> 이 문서는 PRP(Pairwise Ranking Prompting)를 ranksmith에 추가하기 위한 설계 기준이다. 구현은 사용자 최종 승인 후 진행한다.

## 1. 개요 (Overview)
- **작업 목적**: LLM이 여러 문서의 전체 permutation을 한 번에 생성하지 않고, 두 문서 간 상대 비교만 수행하도록 하여 ranking task의 난도를 낮춘다.
- **알고리즘 명칭**: 1차 구현은 `prp_sliding_k`로 제한한다.
- **Reference**: `docs/wiki/references/pairwise_ranking_prompting.md`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 입력 (Inputs)
- `query`: 문서 관련성을 평가할 사용자 질의.
- `documents`: `Sequence[Document]`.
- `provider`: pairwise 비교 호출을 지원하는 provider.
- `algorithm`: 1차 구현에서는 `"prp_sliding_k"`만 허용한다.
- `passes`: 뒤에서 앞으로 pairwise swap pass를 몇 번 수행할지 결정한다. 기본값은 논문 기준과 맞춰 `10`으로 둔다.
- `max_document_chars`: 개별 문서 최대 길이. 기존 fast fail 정책을 유지한다.

### 출력 (Outputs)
- 최종 정렬된 `RerankResult` 리스트.
- `rank`는 1-based를 유지한다.
- `original_index`는 입력 문서 기준 0-based를 유지한다.
- metadata에는 최소한 다음 값을 포함한다.
  - `{"strategy": "pairwise", "algorithm": "prp_sliding_k"}`

### 제약 사항 (Constraints)
- 신규 public API로 `PairwiseStrategy`와 `AsyncPairwiseStrategy`를 추가한다.
- 기존 `ListwiseStrategy`의 provider 계약을 오염시키지 않는다.
- `PairwiseStrategy`는 `PairwiseLLMProvider.compare()`를 지원하는 provider만 받는다.
- `AsyncPairwiseStrategy`는 `AsyncPairwiseLLMProvider.compare()`를 지원하는 provider만 받는다.
- pairwise 응답 계약은 JSON으로 통일한다.
- 논문 원문처럼 invalid generation을 조용히 tie로 완화하지 않는다.
- A/B와 B/A가 모두 유효하지만 서로 충돌하는 경우만 tie로 처리한다.
- 문서 길이는 조용히 자르지 않고 `DocumentTooLongError`로 실패한다.
- provider 응답이 계약을 위반하면 `RerankParseError`로 실패한다.
- `top_k`는 최종 결과 slicing에만 적용하고, ranking 절차 자체를 임의로 축소하지 않는다.
- `passes`와 `top_k`는 독립이다. `top_k=10`이더라도 `passes`를 자동으로 `10`으로 바꾸지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 3.1 동작 메커니즘
`prp_sliding_k`는 초기 ranking 순서를 기준으로 뒤에서 앞으로 인접 문서 쌍을 비교한다.

1. 현재 순서를 `current_order = [0, 1, ..., n - 1]`로 둔다.
2. 한 pass는 리스트 맨 뒤에서 시작해 앞쪽으로 이동한다.
3. 인접한 두 문서 `(left, right)`를 pairwise provider에 비교 요청한다.
4. 위치 편향 완화를 위해 같은 쌍을 두 번 호출한다.
   - 첫 호출: `left`를 A, `right`를 B로 둔다.
   - 둘째 호출: `right`를 A, `left`를 B로 둔다.
5. 두 호출이 같은 실제 문서를 선호하면 그 문서를 winner로 확정한다.
6. winner가 현재 뒤쪽 문서라면 두 문서 위치를 swap한다.
7. 두 호출이 충돌하면 tie로 보고 현재 순서를 유지한다.
8. 위 과정을 `passes`번 반복한다.

### 3.2 Pairwise 응답 계약
provider는 pairwise 비교에서 다음 JSON만 반환해야 한다.

```json
{"winner": "A"}
```

허용 값은 `"A"` 또는 `"B"`뿐이다. 소문자, 다른 문자열, 누락 값, 잘못된 JSON은 모두 `RerankParseError`다.

### 3.3 Pairwise prompt 계약
Azure pairwise provider는 listwise prompt와 별도 prompt builder를 사용한다.

```text
Given a query, choose which passage is more relevant to the query.

Query:
{query}

Passage A:
{document_a.text}

Passage B:
{document_b.text}

Return JSON exactly like this shape:
{"winner": "A"}

Use "A" if Passage A is more relevant. Use "B" if Passage B is more relevant.
```

system message는 JSON만 반환하도록 요구한다. `response_format={"type": "json_object"}`와 `temperature=0`은 기존 listwise provider와 동일하게 유지한다.

### 3.4 의사 알고리즘 (Pseudo-algorithm)
```text
current_order = [0, 1, ..., n - 1]

repeat passes times:
  for right_pos from n - 1 down to 1:
    left_pos = right_pos - 1
    left_index = current_order[left_pos]
    right_index = current_order[right_pos]

    winner_1 = compare_pair(A=left_index, B=right_index)
    winner_2 = compare_pair(A=right_index, B=left_index)

    preferred_index =
      left_index  if winner_1 == left_index  and winner_2 == left_index
      right_index if winner_1 == right_index and winner_2 == right_index
      tie         otherwise

    if preferred_index == right_index:
      swap current_order[left_pos], current_order[right_pos]

return current_order
```

### 3.5 의사 코드 (Pseudo-code)
```python
def prp_sliding_k(query, documents, provider, passes=10):
    if not supports_pairwise_compare(provider):
        raise RerankInputError("provider must support pairwise compare()")

    current_order = list(range(len(documents)))

    for _ in range(passes):
        for right_pos in range(len(current_order) - 1, 0, -1):
            left_pos = right_pos - 1
            left_index = current_order[left_pos]
            right_index = current_order[right_pos]

            first = compare_pair(query, documents[left_index], documents[right_index])
            second = compare_pair(query, documents[right_index], documents[left_index])

            first_winner = left_index if first == "A" else right_index
            second_winner = right_index if second == "A" else left_index

            if first_winner != second_winner:
                continue

            if first_winner == right_index:
                current_order[left_pos], current_order[right_pos] = (
                    current_order[right_pos],
                    current_order[left_pos],
                )

    return current_order
```

### 3.6 호출 수와 비용 모델
`prp_sliding_k`는 각 인접 쌍을 A/B, B/A 두 번 비교한다.

```text
pairwise_provider_calls = 2 * passes * max(document_count - 1, 0)
```

예시:
- `document_count=20`, `passes=10`이면 query당 `380`회 호출한다.
- `document_count=100`, `passes=10`이면 query당 `1,980`회 호출한다.

benchmark와 live 실행 CLI는 실행 전 이 호출 수를 반드시 출력한다. `--allow-live`가 있더라도 호출 수가 사용자의 의도와 맞는지 확인할 수 있어야 한다.

### 3.7 통합 지점 (Integration Points)
- `src/ranksmith/_providers.py`
  - 기존 `LLMProvider.rank(query, documents)`는 listwise 전용 계약으로 유지한다.
  - 신규 `PairwiseLLMProvider.compare(query, document_a, document_b) -> str` Protocol을 추가한다.
  - 신규 `AsyncPairwiseLLMProvider.compare(...)` Protocol을 추가한다.
  - Azure provider에는 pairwise 전용 prompt builder를 분리한다.
- `src/ranksmith/strategies.py`
  - `PairwiseAlgorithm = Literal["prp_sliding_k"]` 추가.
  - `PairwiseStrategy(algorithm="prp_sliding_k", passes=10, max_document_chars=4000)` 추가.
  - `AsyncPairwiseStrategy(algorithm="prp_sliding_k", passes=10, max_document_chars=4000)` 추가.
  - `_parse_pairwise_winner()` 추가.
  - provider가 pairwise protocol을 만족하지 않으면 `RerankInputError`로 실패한다.
- `src/ranksmith/azure.py`
  - 기존 `RerankStrategy` / `AsyncRerankStrategy` 주입 구조는 유지하되, provider type은 listwise와 pairwise를 모두 수용할 수 있게 확장한다.
  - `AzureOpenAIReranker(strategy=PairwiseStrategy(...))`가 동작하도록 한다.
- `src/ranksmith/__init__.py`
  - `PairwiseStrategy`, `AsyncPairwiseStrategy`를 public API로 export한다.
- `README.md`, `README.ko.md`
  - PairwiseStrategy 사용 예시와 비용 특성을 추가한다.
- `scripts/compare_reranking.py`
  - live opt-in 비교 대상에 `prp_sliding_k`를 추가한다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### 공통 컴포넌트 식별 (Shared Components)
- `_ListwiseConfigMixin`의 문서 길이 검증은 pairwise에도 필요하다.
- 장기적으로 이름을 `_DocumentValidationMixin`처럼 일반화할 수 있다.
- `RerankResult` 생성 로직은 strategy별 중복을 줄일 수 있다.
- usage callback은 provider 호출 수가 늘어나는 pairwise에서 그대로 중요하다.

### 추상화 방안 (Abstraction Plan)
- `PairwiseStrategy`는 listwise 구현과 분리한다.
- pairwise 전용 parsing은 `_parse_pairwise_winner()`로 둔다.
- pairwise prompt 생성은 `_build_pairwise_prompt()`로 분리한다.
- 첫 구현은 `prp_sliding_k`만 포함한다.
- `prp_allpair`, `prp_sorting`은 같은 비교 단위를 재사용할 수 있도록 후속 algorithm 후보로 남긴다.

## 5. 에러 핸들링 (Error Handling)
- `passes < 1`
  - **Exception**: `ValueError`
  - **이유**: algorithm 구성 자체가 유효하지 않다.
- provider가 `compare()`를 지원하지 않음
  - **Exception**: `RerankInputError`
  - **이유**: `PairwiseStrategy`와 provider의 계약이 맞지 않는 사용자 구성 오류다.
- `max_document_chars < 1`
  - **Exception**: `ValueError`
  - **이유**: 문서 길이 검증 기준이 성립하지 않는다.
- `top_k < 0`
  - **Exception**: `RerankInputError`
  - **이유**: 기존 reranker 계약과 동일하게 입력 오류로 분류한다.
- 개별 문서 길이가 `max_document_chars` 초과
  - **Exception**: `DocumentTooLongError`
  - **이유**: 숨은 truncation 금지.
- pairwise 응답이 JSON이 아니거나 `winner`가 없음
  - **Exception**: `RerankParseError`
  - **이유**: provider 응답 계약 위반.
- `winner` 값이 `"A"` 또는 `"B"`가 아님
  - **Exception**: `RerankParseError`
  - **이유**: 조용한 보정 금지.
- A/B와 B/A 비교 결과가 서로 충돌
  - **Exception**: 없음
  - **동작**: 논문 방식에 맞춰 tie로 간주하고 현재 순서를 유지한다.
- provider 호출 실패
  - **Exception**: `RerankProviderError`
  - **이유**: provider 계층 오류로 분류한다.

## 6. 테스트 계획 (Test Plan)

### 성공 케이스 (Happy Paths)
- `PairwiseStrategy(algorithm="prp_sliding_k", passes=1)`가 인접 문서를 뒤에서 앞으로 비교하는지 검증한다.
- `PairwiseStrategy()`의 기본 `passes`가 `10`인지 검증한다.
- 각 인접 쌍마다 A/B, B/A 두 번 호출하는지 검증한다.
- 두 호출이 같은 실제 문서를 선호하면 swap 여부가 올바른지 검증한다.
- 충돌하는 선호는 tie로 처리되어 기존 순서가 유지되는지 검증한다.
- `rank`는 1-based, `original_index`는 0-based로 보존되는지 검증한다.
- `top_k`가 최종 결과만 slicing하는지 검증한다.
- async strategy도 sync와 같은 최종 순서를 내는지 검증한다.

### 엣지/실패 케이스 (Edge & Failure Cases)
- 빈 문서 목록은 빈 결과를 반환한다.
- 문서가 1개면 provider 호출 없이 그대로 반환한다.
- `passes < 1`은 `ValueError`.
- pairwise `compare()`를 지원하지 않는 provider는 `RerankInputError`.
- `top_k < 0`은 `RerankInputError`.
- malformed JSON, 누락된 `winner`, 잘못된 winner 값은 `RerankParseError`.
- 긴 문서는 provider 호출 전 `DocumentTooLongError`.
- provider 예외는 `RerankProviderError`로 wrapping된다.

### 공통 Reranking Smoke/Benchmark
- `tests/fixtures/reranking_smoke_fixture.jsonl` 기반 smoke test에 `prp_sliding_k`를 추가한다.
- `src/ranksmith/_metrics.py`의 NDCG@k, MRR@k, Recall@k 계산을 재사용한다.
- `scripts/compare_reranking.py`는 `prp_sliding_k`의 예상 LLM 호출 수를 출력한다.
- `prp_sliding_k` 호출 수는 `2 * passes * max(document_count - 1, 0)`로 계산한다.
- live Azure OpenAI 실행은 기존 정책대로 `--allow-live` opt-in에서만 수행한다.
- 완료 판단에는 synthetic provider 테스트와 fixture 기반 metric 검증을 모두 포함한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] `docs/wiki/00_context.md` 확인
- [x] `docs/wiki/01_decisions.md` 확인
- [x] `docs/wiki/02_architecture.md` 확인
- [x] `docs/wiki/03_reference_processing.md` 확인
- [x] `docs/wiki/04_references_index.md` 확인
- [x] `docs/wiki/06_verification_policy.md` 확인
- [x] `docs/wiki/references/pairwise_ranking_prompting.md` 작성 및 확인
- [x] 1차 범위를 `PairwiseStrategy` + `prp_sliding_k`로 확정
- [x] 사용자 최종 승인 확인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/_providers.py`: `PairwiseLLMProvider` / `AsyncPairwiseLLMProvider` protocol 추가
- [x] `src/ranksmith/_providers.py`: Azure pairwise prompt 및 `compare()` 구현
- [x] `src/ranksmith/strategies.py`: `PairwiseStrategy` 구현
- [x] `src/ranksmith/strategies.py`: `AsyncPairwiseStrategy` 구현
- [x] `src/ranksmith/strategies.py`: `_parse_pairwise_winner()` 구현
- [x] `src/ranksmith/strategies.py`: `compare()` 미지원 provider의 `RerankInputError` 처리
- [x] `src/ranksmith/azure.py`: 기존 strategy injection 구조와 pairwise provider 연결 확인
- [x] `src/ranksmith/__init__.py`: 신규 strategy export
- [x] `README.md`, `README.ko.md`: 사용 예시 및 비용 설명 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_ranksmith.py`: sync pairwise 정상/실패 케이스 추가
- [x] `tests/test_async_providers.py`: async pairwise 정상/실패 케이스 추가
- [x] `tests/test_benchmark_fixture.py`: fixture 기반 metric 검증에 `prp_sliding_k` 추가
- [x] `scripts/compare_reranking.py`: 비교 대상 및 호출 수 계산에 `prp_sliding_k` 추가
- [x] `scripts/evaluate_mteb_reranking.py`: MTEB method 및 호출 수 계산에 `prp_sliding_k@N` 추가
- [x] `./scripts/verify.sh` 실행 및 결과 기록

### Phase 4: 완료 및 정리
- [x] 필요 시 `docs/wiki/02_architecture.md`에 `PairwiseStrategy` 반영
- [x] 필요 시 `docs/wiki/01_decisions.md`에 public API 확장 결정 기록
- [x] `docs/wiki/05_open_questions.md`의 Q001 완료 상태 확인
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
