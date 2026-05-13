# Spec: RankGPT Sliding Window Algorithm

> 이 문서는 `rankgpt_sliding_window` 알고리즘의 설계, 구현 지점, 검증 기준을 사람과 코딩 어시스턴트가 함께 추적하기 위한 기준 문서다.

## 1. 개요 (Overview)
- **작업 목적**: LLM의 한 번의 컨텍스트에 모든 후보 문서를 넣기 어려운 상황에서, RankGPT 논문의 back-to-first sliding window와 bubble-up 메커니즘을 ranksmith의 listwise reranking 알고리즘으로 제공한다.
- **알고리즘 명칭**: `rankgpt_sliding_window`
- **Reference**: "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (Sun et al.), `docs/wiki/references/rankgpt.md`
- **상태**: `[ ] Draft` | `[x] In Progress` | `[ ] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**
  - `documents`: 랭킹 대상 `Document` 시퀀스.
  - `query`: 문서 관련성을 평가할 사용자 질의.
  - `provider`: `LLMProvider` 구현체. 각 윈도우에 대해 JSON ranking 응답을 반환해야 한다.
  - `window_size`: 한 번의 LLM 요청에 포함할 최대 문서 수. `1` 이상이어야 한다.
  - `stride`: 다음 앞쪽 윈도우로 이동할 때 제외되는 하위 문서 수. `1` 이상이고 `window_size` 이하여야 한다.
- **출력 (Outputs)**
  - 최종 정렬 순서에 따라 생성된 `RerankResult` 리스트.
  - `rank`는 1-based, `original_index`는 입력 문서 기준 0-based를 유지한다.
  - 결과 metadata에는 `{"strategy": "listwise", "algorithm": "rankgpt_sliding_window"}`를 포함한다.
- **제약 사항 (Constraints)**
  - `stride > window_size`는 bubble-up 전달 구간을 만들 수 없으므로 `RerankInputError`로 fast fail한다.
  - 문서를 조용히 자르지 않는다. 개별 문서가 `max_document_chars`를 넘으면 `DocumentTooLongError`로 실패한다.
  - RankGPT 원문 응답 형식(`[2] > [3] > [1]`)은 사용하지 않는다. ranksmith의 기존 LLM 응답 계약인 `{"ranking": [1, 2, 3]}` JSON permutation만 허용한다.
  - 잘못된 ranking을 보정하지 않는다. 누락, 중복, 범위 밖 값, 비정수 값, 잘못된 JSON은 `RerankParseError`로 실패한다.
  - Public API는 확장하지 않는다. 기존 `ListwiseStrategy.algorithm` 옵션만 사용한다.

## 3. 상세 설계 (Architecture & Design)

### 3.1 동작 메커니즘
전체 문서 순서를 뒤에서 앞으로(back-to-first) 이동하는 윈도우로 반복 평가한다.

1. 현재 전체 순서를 `current_order`로 보관한다. 값은 입력 문서의 0-based original index다.
2. 첫 윈도우 시작점은 `document_count - window_size`다.
3. 현재 윈도우의 original index 목록을 추출하고, 해당 문서들을 provider에 전달한다.
4. provider 응답의 1-based ranking permutation을 파싱한다.
5. 윈도우 내부 순서를 ranking 결과대로 `current_order`에 제자리 반영한다.
6. `start_pos == 0`이면 첫 윈도우까지 처리한 것이므로 종료한다.
7. 아니면 `start_pos -= stride`로 앞쪽 윈도우로 이동한다.

하위 `stride`개 문서는 다음 윈도우에서 제외되고, 상위 `window_size - stride`개 문서는 앞쪽 윈도우에 합류하여 다시 평가된다. 이 재평가 대상 전달이 bubble-up이다.

### 3.2 의사 알고리즘 (Pseudo-algorithm)
```text
current_order = [0, 1, 2, ..., n - 1]
start_pos = n - window_size

loop:
  start_pos = max(0, start_pos)
  window_indices = current_order[start_pos : start_pos + window_size]
  window_documents = documents[window_indices]
  ranking = parse(provider.rank(query, window_documents))
  current_order[start_pos : start_pos + window_size] =
    reorder(window_indices, ranking)

  if start_pos == 0:
    break

  start_pos = start_pos - stride

return current_order
```

### 3.3 의사 코드 (Pseudo-code)
```python
def rankgpt_sliding_window(query, documents, provider, window_size, stride):
    if stride > window_size:
        raise RerankInputError("stride must be less than or equal to window_size")

    current_order = list(range(len(documents)))
    start_pos = len(documents) - window_size

    while True:
        start_pos = max(0, start_pos)
        window_indices = current_order[start_pos : start_pos + window_size]
        window_documents = [documents[index] for index in window_indices]

        raw_response = provider.rank(query, window_documents)
        ranking = parse_ranking(raw_response, expected_count=len(window_documents))

        current_order[start_pos : start_pos + window_size] = [
            window_indices[number - 1] for number in ranking
        ]

        if start_pos == 0:
            break

        start_pos -= stride

    return current_order
```

### 3.4 Bubble-up 예시
- **Documents**: `[A, B, C, D, E]`
- **Parameters**: `window_size=3`, `stride=2`

1. 마지막 윈도우 `[C, D, E]`를 평가한다.
2. LLM ranking이 `E > C > D`이면 전체 순서는 `[A, B, E, C, D]`가 된다.
3. 하위 `stride`개인 `C`, `D`는 다음 윈도우에서 제외된다.
4. 앞쪽 윈도우 `[A, B, E]`를 평가한다.
5. LLM ranking이 `E > A > B`이면 최종 순서는 `[E, A, B, C, D]`가 된다.

### 3.5 통합 지점 (Integration Points)
- `src/ranksmith/strategies.py`
  - `Algorithm = Literal["direct", "sliding_window", "rankgpt_sliding_window"]`
  - `ListwiseStrategy.__post_init__`: algorithm 값과 sliding window 계열의 `stride <= window_size` 검증.
  - `ListwiseStrategy.rerank`: `algorithm == "rankgpt_sliding_window"`일 때 `_rank_rankgpt_sliding_windows()`로 분기.
  - `ListwiseStrategy._rank_rankgpt_sliding_windows`: back-to-first bubble-up 순서 생성.
  - `_parse_ranking`: provider 응답 JSON permutation 검증 재사용.
- `tests/test_ranksmith.py`
  - RankGPT bubble-up 순서와 provider 호출 윈도우를 검증한다.
  - `stride > window_size` 실패 경로를 검증한다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **공통 컴포넌트 식별 (Shared Components)**
  - `_parse_ranking`: `direct`, `sliding_window`, `rankgpt_sliding_window`가 모두 공유하는 JSON permutation 검증 로직.
  - `ListwiseStrategy._validate_documents`: 문서 길이 fast fail 정책을 알고리즘과 독립적으로 적용.
  - `RerankResult` 생성 경로: 모든 listwise 알고리즘은 original index 순서를 반환하고, 공통 `rerank()`가 결과 객체를 만든다.
  - `LLMProvider.rank`: provider 호출 인터페이스를 알고리즘 구현과 분리한다.
- **추상화 방안 (Abstraction Plan)**
  - RankGPT 알고리즘은 현재 `_rank_rankgpt_sliding_windows()` private method로 충분하다.
  - 추가 sliding-window 변형이 늘어나면 window traversal helper를 분리할 수 있다.
  - Public API 확장은 하지 않는다. 새 알고리즘은 `ListwiseStrategy.algorithm` 내부 옵션으로만 추가한다.
  - RankGPT 원문 파싱 형식은 도입하지 않고, 기존 `_parse_ranking`을 계속 재사용한다.

## 5. 에러 핸들링 (Error Handling)
- `stride > window_size`
  - **Exception**: `RerankInputError`
  - **이유**: 윈도우 간 겹침이 없어 bubble-up 메커니즘이 성립하지 않는다.
- `window_size < 1`, `stride < 1`, `max_document_chars < 1`
  - **Exception**: `ValueError`
  - **이유**: strategy 구성 자체가 유효하지 않다.
- 개별 문서 길이가 `max_document_chars` 초과
  - **Exception**: `DocumentTooLongError`
  - **이유**: 조용한 truncation 금지.
- provider 응답이 JSON이 아니거나 `ranking` 리스트가 없음
  - **Exception**: `RerankParseError`
  - **이유**: LLM 응답 계약 위반.
- `ranking`에 누락, 중복, 범위 밖 값, 비정수 값이 포함됨
  - **Exception**: `RerankParseError`
  - **이유**: ranking permutation 위반. 조용한 보정 금지.
- LLM API 통신 실패, 권한 오류, timeout 등 provider 호출 실패
  - **Exception**: `RerankProviderError`
  - **이유**: provider 계층 오류로 분류한다.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**
  - `window_size=3`, `stride=2`, 문서 `[a, b, c, d, e]`에서 provider 응답을 `{"ranking": [3, 1, 2]}`로 두 번 반환하도록 구성한다.
  - provider 호출 윈도우가 `[c, d, e]`, `[a, b, e]` 순서인지 확인한다.
  - 최종 문서 순서가 `[e, a, b, c, d]`인지 확인한다.
  - 결과의 `rank`는 1-based, `original_index`는 입력 기준 0-based인지 기존 공통 테스트와 함께 보장한다.
- **엣지/실패 케이스 (Edge & Failure Cases)**
  - `stride > window_size`이면 `RerankInputError`.
  - malformed JSON, 누락, 중복, gap, 비정수 ranking은 `RerankParseError`.
  - 긴 문서는 `DocumentTooLongError`.
  - provider 예외는 `RerankProviderError`.
  - 문서 수가 `window_size` 이하이면 단일 윈도우 경로로 처리된다.
- **실제 데이터 검증 (Real-data Validation)**
  - 단위 테스트의 synthetic 문서와 `FakeProvider`만으로 완료 판단을 하지 않는다.
  - benchmark 데이터셋은 repository 안에 상주시켜 개발 후 반복 검증과 메소드별 비교에 사용한다.
  - 데이터셋은 실제 query, candidate documents, relevance judgment(qrels)를 포함해야 한다.
  - 데이터 파일은 작고 재현 가능한 fixture로 유지한다. 대용량 원본 corpus 전체를 vendoring하지 않는다.
  - 권장 위치:
    - `tests/fixtures/reranking_smoke_fixture.jsonl`: query, candidate documents, qrels를 담은 작은 smoke fixture.
    - `scripts/compare_reranking.py`: `direct`, `sliding_window`, `rankgpt_sliding_window`를 같은 fixture에서 비교하는 수동 실행 스크립트.
  - `examples/`는 repository 내부 fixture 검증이 아니라 패키지 사용자가 public API를 따라 할 수 있는 코드만 둔다.
  - benchmark fixture는 출처, 라이선스, 가공 방식을 파일 또는 인접 문서에 기록해야 한다.
  - 공개 benchmark를 사용할 경우, repository 포함이 허용되는 라이선스인지 확인한 뒤 최소 샘플만 저장한다.
  - 같은 실제 데이터에 대해 `direct`, 기존 `sliding_window`, `rankgpt_sliding_window`를 비교한다.
  - 검증 항목:
    - provider가 호출한 윈도우 순서가 back-to-first인지 확인.
    - 각 윈도우 응답이 strict JSON permutation인지 확인.
    - 최종 `rank`와 `original_index`가 입력 문서와 정확히 매핑되는지 확인.
    - 기대 상위 문서가 상위권으로 bubble-up되는지 확인.
    - NDCG@k, MRR@k, Recall@k를 산출한다.
  - 실제 Azure OpenAI 호출이 필요한 비교는 API credential과 비용이 필요하므로 기본 `./scripts/verify.sh`에는 포함하지 않는다.
  - 기본 검증은 fixture schema와 metric 계산처럼 네트워크가 필요 없는 부분만 자동화한다.
  - live provider 비교는 명시적 opt-in 환경 변수나 CLI flag가 있을 때만 실행한다.
  - 실제 benchmark fixture의 출처 또는 라이선스가 확정되지 않으면 완료로 표시하지 않는다.

---

## 7. 작업 태스크 추적 (Task Checklist)
> 개발 진행 중 완료된 작업은 `[x]`로 표시하고, 필요시 하위 태스크를 추가한다.

### Phase 1: 컨텍스트 및 설계 확인
- [x] `docs/wiki/00_context.md` 확인
- [x] `docs/wiki/01_decisions.md` 확인
- [x] `docs/wiki/02_architecture.md` 확인
- [x] `docs/wiki/03_reference_processing.md` 확인
- [x] `docs/wiki/04_references_index.md` 확인
- [x] `docs/wiki/06_verification_policy.md` 확인
- [x] `docs/wiki/references/rankgpt.md` 확인
- [x] 스펙 문서의 의사 알고리즘과 기존 설계 정합성 확인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/strategies.py`: `Algorithm`에 `rankgpt_sliding_window` 추가
- [x] `src/ranksmith/strategies.py`: `ListwiseStrategy.rerank()` 분기 연결
- [x] `src/ranksmith/strategies.py`: `_rank_rankgpt_sliding_windows()` 구현
- [x] `src/ranksmith/strategies.py`: `stride <= window_size` 방어 로직 추가
- [x] `src/ranksmith/strategies.py`: 기존 `_parse_ranking` 기반 strict JSON permutation 검증 재사용

### Phase 3: 검증 (Verification)
- [x] `tests/test_ranksmith.py`: bubble-up 정상 케이스 단위 테스트 추가
- [x] `tests/test_ranksmith.py`: `stride > window_size` 실패 케이스 단위 테스트 추가
- [x] 기존 parse/input/provider error 테스트와의 정합성 확인
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인
- [x] `tests/fixtures/reranking_smoke_fixture.jsonl`: 실제 query/documents/qrels 기반 smoke fixture 추가
- [x] `tests/fixtures/reranking_smoke_fixture.SOURCES.md`: fixture의 출처, 라이선스, 가공 방식 기록
- [x] `tests/test_benchmark_fixture.py`: fixture schema 검증 및 NDCG@k, MRR@k, Recall@k metric 계산 테스트 추가
- [x] `src/ranksmith/_metrics.py`: method comparison용 ranking metric helper 추가
- [x] `examples/rankgpt_sync.py`: public API 기반 동기 사용 예제 유지 및 `.env` 로드 지원
- [x] `examples/rankgpt_async.py`: public API 기반 비동기 사용 예제 유지 및 `.env` 로드 지원
- [x] `scripts/compare_reranking.py`: 실제 데이터에서 `direct`, `sliding_window`, `rankgpt_sliding_window` 비교 실행 스크립트 추가
- [x] live Azure OpenAI 비교 실행을 `--allow-live` opt-in 방식으로 분리
- [ ] 실제 데이터 기준 metric 결과 기록

### Phase 4: 완료 및 정리
- [x] `docs/wiki/references/rankgpt.md` reference summary 정리
- [ ] 실제 데이터 검증 완료 후 본 문서 최상단의 **상태**를 `Completed`로 변경
