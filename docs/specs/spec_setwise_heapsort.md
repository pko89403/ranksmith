# Spec: Setwise Heapsort Strategy

## 1. 개요 (Overview)
- **작업 목적**: 여러 문서 중 1개를 선택하는 Setwise prompting을 heap sort에 연결해, Pairwise보다 적은 LLM 호출로 top-k reranking을 수행한다.
- **Reference**: `docs/wiki/references/setwise_ranking_prompting.md`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**
  - `query: str`
  - `documents: Sequence[Document]`
  - `top_k: int | None`
  - `set_size: int = 3`
  - `max_document_chars: int = 4000`
- **출력 (Outputs)**
  - `list[RerankResult]`
  - `rank`는 1-based, `original_index`는 0-based를 유지한다.
  - metadata에는 `strategy="setwise"`, `algorithm="setwise_heapsort"`, `set_size`를 포함한다.
- **제약 사항 (Constraints)**
  - `set_size`는 한 번의 LLM 호출에 넣는 문서 수다.
  - heap의 child arity는 `set_size - 1`이다.
  - LLM 호출은 기존 `ModelClient.select(query, documents, top_m=1)` 계약을 재사용한다.
  - logits 기반 `listwise.likelihood`는 구현하지 않는다.
  - `setwise.bubblesort`는 구현하지 않는다.
  - 사용자 승인 없이 public API 범위를 더 넓히지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 동작 메커니즘
1. 입력 문서 index를 초기 순서대로 heap 배열에 둔다.
2. bottom-up heapify로 max heap을 만든다.
3. root를 top 문서로 추출한다.
4. root와 heap 끝을 교환하고 heap 크기를 줄인다.
5. 줄어든 heap에 다시 heapify를 수행한다.
6. `top_k`가 있으면 `top_k`개를 추출한 뒤 멈춘다.
7. `top_k`가 없으면 모든 문서를 추출해 전체 순위를 만든다.

### 의사 알고리즘 (Pseudo-algorithm)
```text
set_size = c
child_arity = c - 1

heapify(heap, root, heap_size):
  while root has at least one child:
    candidates = [root] + children(root, child_arity, heap_size)
    selected = select(query, candidates, top_m=1)
    if selected is root:
      stop
    swap(root, selected_child)
    root = selected_child

build_heap(heap):
  for root from last_parent down to 0:
    heapify(heap, root, len(heap))

extract:
  limit = n if top_k is None else min(top_k, n)
  repeat limit times:
    append heap[0] to ranked
    swap heap[0] and heap[heap_size - 1]
    heap_size -= 1
    if more output is needed and heap_size > 1:
      heapify(heap, 0, heap_size)
```

### 의사 코드 (Pseudo-code)
```python
class SetwiseStrategy:
    algorithm = "setwise_heapsort"
    set_size = 3

    def rerank(self, query, documents, model_client, top_k=None):
        validate_top_k(top_k)
        validate_documents_max_chars(documents, max_document_chars)
        model_client = ensure_selection_model_client(model_client)

        heap = list(range(len(documents)))
        child_arity = self.set_size - 1
        build_heap(heap, child_arity)

        limit = len(heap) if top_k is None else min(top_k, len(heap))
        ranked = []
        heap_size = len(heap)
        for _ in range(limit):
            ranked.append(heap[0])
            heap[0], heap[heap_size - 1] = heap[heap_size - 1], heap[0]
            heap_size -= 1
            heapify(heap, 0, heap_size, child_arity)

        return results_from_indexes(ranked)
```

### 통합 지점 (Integration Points)
- `src/ranksmith/strategies/_setwise.py`
  - `SetwiseStrategy`
  - `AsyncSetwiseStrategy`
- `src/ranksmith/strategies/__init__.py`
  - 새 Strategy export
- `src/ranksmith/__init__.py`
  - root public API export
- `docs/wiki/00_context.md`
  - Public API 및 algorithm 목록 갱신
- `docs/wiki/02_architecture.md`
  - Strategy/Algorithm 목록 갱신

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- 기존 selection 계약을 재사용한다.
- 기존 `parse_selection_response()`를 재사용한다.
- 기존 `_common.py`의 `ensure_selection_model_client()`, `ensure_async_selection_model_client()`, `validate_documents_max_chars()`, `validate_top_k()`를 재사용한다.
- sync/async 구현은 heap index 계산을 공유 가능한 pure helper로 분리한다.

## 5. 에러 핸들링 (Error Handling)
- `set_size < 3`: `ValueError`
- `max_document_chars < 1`: `ValueError`
- `top_k < 0`: `RerankInputError`
- provider가 `select()` 미지원: `RerankInputError`
- `select()` 응답이 top-1 selection 계약 위반: `RerankParseError`
- 문서 길이 초과: `DocumentTooLongError`
- 빈 문서 목록: 빈 리스트 반환
- `top_k=0`: 빈 리스트 반환

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**
  - mock selection provider가 점수 높은 문서를 고를 때 `SetwiseStrategy`가 올바른 top-k를 반환한다.
  - `top_k=1`이면 필요한 개수만 추출하고 멈춘다.
  - `top_k=None`이면 전체 순위를 반환한다.
  - `AsyncSetwiseStrategy`도 동일한 순위를 반환한다.
- **엣지/실패 케이스 (Edge & Failure Cases)**
  - `set_size=2` 거부
  - `top_k=0` 빈 결과
  - provider `select()` 미지원 시 fast fail
  - invalid selection JSON fast fail
  - `max_document_chars` 초과 fast fail
- **공통 Reranking Smoke/Benchmark**
  - synthetic provider 테스트를 추가했다.
  - fixture 기반 smoke test를 추가했다.
  - live provider benchmark는 `setwise_hs_s10` 실용 설정으로 실행하고 README에 반영했다.
  - `setwise_hs_s10`은 public 기본값이 아니라 benchmark alias이며, `set_size=10`, `top_k=5`를 사용한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] reference 요약 문서 생성
- [x] 스펙 문서 초안 생성
- [x] 사용자 최종 승인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/strategies/_setwise.py`: sync/async setwise heapsort 구현
- [x] `src/ranksmith/strategies/__init__.py`: Strategy export 연결
- [x] `src/ranksmith/__init__.py`: root public API export 연결
- [x] `scripts/compare_reranking.py`: benchmark 비교 대상 및 호출 수 estimate 추가
- [x] `examples/setwise_heapsort.py`: live provider 없이 실행 가능한 예제 추가
- [x] `docs/wiki/00_context.md`: Public API 및 algorithm 목록 갱신
- [x] `docs/wiki/02_architecture.md`: Strategy/Algorithm 목록 갱신

### Phase 3: 검증 (Verification)
- [x] `tests/test_setwise.py`: 정상 케이스 단위 테스트 추가
- [x] `tests/test_setwise.py`: 엣지 케이스 및 에러 발생 단위 테스트 추가
- [x] `tests/test_benchmark_fixture.py`: fixture 기반 smoke test 추가
- [x] `tests/test_compare_reranking.py`: benchmark 선택/estimate 테스트 추가
- [x] `tests/test_examples.py`: Setwise example 실행 테스트 추가
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] `docs/wiki/references/setwise_ranking_prompting.md` 상태 확인
- [x] `docs/wiki/04_references_index.md` 상태 갱신
- [x] `docs/wiki/05_open_questions.md` Q003 resolved 처리
- [x] `README.md`, `README.ko.md`: benchmark 및 example 목록 반영
- [x] `docs/benchmarks/bm25_top20_reranking.md`: benchmark 설정 반영
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
