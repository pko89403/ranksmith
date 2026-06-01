# Spec: AcuRank

> **Historical Spec**
> 이 문서는 과거 구현 당시의 설계 기록입니다.
> 현재 code/API 기준은 `docs/wiki/02_architecture.md`와
> `docs/wiki/08_custom_strategy_extension.md`를 따릅니다.
> 아래 파일 경로는 현재 구조에 맞게 최소 보정했습니다.

## 1. 개요 (Overview)
- **작업 목적**: AcuRank를 ranksmith의 built-in uncertainty-aware adaptive reranking Strategy로 추가한다.
- **Reference**:
  - `docs/wiki/references/acurank.md`
  - `docs/wiki/references/AcuRank- Uncertainty-Aware Adaptive Computation for Listwise Reranking.pdf`
  - https://github.com/soyoung97/AcuRank
- **상태**: `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**:
  - `query: str`
  - `documents: Sequence[str | Document]`
  - `model_client.rank(query, documents)`를 지원하는 listwise model client
  - 선택적 `top_k`
- **출력 (Outputs)**:
  - `list[RerankResult]`
  - `rank`는 1-based
  - `original_index`는 0-based
- **제약 사항 (Constraints)**:
  - 기존 `ListwiseStrategy`, `PairwiseStrategy`, `TourRankStrategy` 동작을 바꾸지 않는다.
  - 공식 repo 코드는 복사하지 않고, 논문 알고리즘 설명을 ranksmith 구조에 맞게 독립 구현한다.
  - 공식 repo와의 재현성은 high-level algorithm flow 기준으로 맞춘다. 구현 코드, prompt/parser, cleanup 로직은 재사용하지 않는다.
  - invalid listwise JSON은 보정하지 않고 `RerankParseError`로 실패한다.
  - 문서 text를 조용히 자르지 않는다. 기존 `max_document_chars` 정책을 유지한다.
  - Public API 확장은 이 spec 승인 후 진행한다.
  - `trueskill>=0.4.5` 의존성을 사용한다.

- **라이선스 / 독립 구현 경계**:
  - 공식 Repo license가 확정되지 않은 상태로 확인되었으므로 repository 구현 코드는 복사하지 않는다.
  - 구현 근거는 논문에 공개된 알고리즘 설명, 수식, hyperparameter, 그리고 공식 Repo의 관찰 가능한 단계 흐름으로 제한한다.
  - ranksmith 구현은 기존 public abstraction(`Document`, `ModelClient.rank()`, `parse_ranking_response()`, `RerankError`) 위에서 독립 작성한다.
  - 공식 Repo와 다르게 처리하는 부분은 `docs/wiki/references/acurank.md`의 “재현성 및 의도적 차이”에 기록한다.

## 3. 상세 설계 (Architecture & Design)
- **새 Strategy**:
  - `AcuRankStrategy`
  - `AsyncAcuRankStrategy`
- **Algorithm 이름**:
  - `acurank`
- **기본 설정**:
  - `target_rank=10`
  - `window_size=20`
  - `tolerance=0.01`
  - `uncertain_threshold=10`
  - `max_adaptive_reranker_calls=None`
  - `batch_parallelism=1`
  - `initial_pass=True`
  - `score_metadata_key="score"`
  - `max_document_chars=4000`
- **Prior 정책**:
  - `Document.metadata[score_metadata_key]`에 numeric score가 있으면 논문 기본값을 따른다.
    - `mu_i = score`
    - `sigma_i = score / 3`
    - `bool` 값은 numeric score로 인정하지 않는다.
  - score가 없으면 standard TrueSkill prior를 사용한다.
    - `mu_i = 25`
    - `sigma_i = 25 / 3`
  - metadata score가 일부 문서에만 있으면 조용히 섞지 않고 `RerankInputError`로 실패한다.
- **동작 메커니즘**:
  1. 입력 문서를 검증한다.
  2. 문서별 TrueSkill `Rating(mu, sigma)` 상태를 초기화한다.
  3. `initial_pass=True`이면 입력 순서 기준으로 `window_size` 단위 batch를 한 번 rerank하고 rating을 갱신한다.
  4. `effective_target_rank = min(target_rank, len(documents))`를 사용한다.
  5. 각 반복에서 top-k threshold `t(effective_target_rank)`를 찾는다.
  6. 각 문서의 `s_i = P(x_i > t(effective_target_rank))`를 계산한다.
  7. probability 내림차순으로 candidate를 정렬한다.
  8. `tolerance < s_i < 1 - tolerance` 문서를 uncertain candidate로 선택한다.
  9. uncertain candidate 수가 `uncertain_threshold`보다 작으면 `s_i > tolerance` 후보를 final refinement 대상으로 선택하고, 이 iteration 이후 종료한다.
  10. 그렇지 않으면 `tolerance < s_i < 1 - tolerance` 후보를 adaptive refinement 대상으로 선택한다.
  11. 선택된 candidate를 probability 순서대로 `window_size` 단위 batch로 나눈다.
  12. 각 batch에 `model_client.rank()`를 호출한다. `batch_parallelism > 1`이면 같은 iteration 안의 독립 batch 호출만 병렬화한다.
  13. 반환 ranking을 `parse_ranking_response()`로 검증한다.
  14. batch ranking을 `trueskill.rate()` 입력으로 변환해 rating을 갱신한다. 병렬 호출을 사용해도 rating update는 batch order로 순차 적용한다.
  15. `max_adaptive_reranker_calls`가 있으면 adaptive refinement 호출만 제한한다. initial pass 호출은 이 budget에 포함하지 않는다.
  16. 최종 결과는 `mu_i` 내림차순, 동점은 `original_index` 오름차순으로 정렬한다.
- **의사 코드 (Pseudo-code)**:

```python
ratings = initialize_ratings(documents)

if initial_pass:
    for batch in chunks(input_order, window_size):
        ranking = rank_batch(batch)
        update_ratings(batch, ranking)

while True:
    should_break = False
    target_rank = min(target_rank, len(documents))
    threshold = find_topk_threshold(ratings, target_rank)
    probabilities = [normal_tail_probability(rating, threshold) for rating in ratings]
    probability_order = sort_by_probability_desc(probabilities)
    uncertain = [
        index
        for index in probability_order
        if tolerance < probabilities[index] < 1 - tolerance
    ]
    if len(uncertain) < uncertain_threshold:
        uncertain = [
            index for index in probability_order if probabilities[index] > tolerance
        ]
        should_break = True

    for batch in chunks(uncertain, window_size):
        if adaptive_budget_exhausted():
            break
        ranking = rank_batch(batch)
        update_ratings(batch, ranking)

    if should_break or adaptive_budget_exhausted():
        break

return sort_by_mu_desc_then_original_index(ratings)
```

- **통합 지점 (Integration Points)**:
  - `src/ranksmith/strategies/_acurank.py`
    - `AcuRankAlgorithm`
    - `_AcuRankConfigMixin`
    - `AcuRankStrategy`
    - `AsyncAcuRankStrategy`
    - threshold / probability helper
    - TrueSkill update helper
  - `src/ranksmith/__init__.py`
    - public import 추가
  - `src/ranksmith/azure.py`
    - built-in strategy error wrapping 목록에 추가
  - `scripts/compare_reranking.py`
    - live opt-in 비교 대상 추가
  - `src/ranksmith/_mteb_eval.py`
    - benchmark method parser 추가

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- 기존 `_validate_documents()` 패턴을 따른다.
- 기존 `parse_ranking_response()`를 재사용한다.
- listwise prompt 계약은 `ModelClient.rank()`를 그대로 재사용한다.
- threshold 계산은 pure helper로 분리해 단위 테스트한다.
- TrueSkill update 변환은 sync/async Strategy가 공유할 수 있는 helper로 분리한다.
- sync/async 차이는 model call scheduling에만 둔다.

## 5. 에러 핸들링 (Error Handling)
- `target_rank < 1`: `ValueError`
- `window_size < 1`: `ValueError`
- `tolerance <= 0` 또는 `tolerance >= 0.5`: `ValueError`
- `uncertain_threshold < 1`: `ValueError`
- `max_adaptive_reranker_calls < 0`: `ValueError`
- `batch_parallelism < 1`: `ValueError`
- `top_k < 0`: `RerankInputError`
- metadata score가 일부만 있거나 numeric이 아니거나 `bool`이면 `RerankInputError`
- provider가 `rank()`를 지원하지 않으면 `RerankInputError`
- invalid ranking JSON은 `RerankParseError`
- provider 실패는 `RerankProviderError`

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**:
  - `AcuRankStrategy` 기본값 public import
  - metadata score 기반 TrueSkill 초기화
  - score가 없을 때 standard TrueSkill prior 사용
  - deterministic fake rank provider로 `mu` 기준 최종 순위 생성
  - `top_k` 적용 시 rank와 original_index 보존
  - async strategy가 동일 결과를 반환
- **엣지/실패 케이스 (Edge & Failure Cases)**:
  - invalid constructor parameter
  - 일부 문서에만 score metadata 존재
  - non-numeric score metadata
  - provider without `rank()`
  - invalid listwise ranking JSON
  - `max_document_chars` 초과
  - empty documents
  - one-document input
- **공통 Reranking Smoke/Benchmark**:
  - `tests/fixtures/reranking_smoke_fixture.jsonl` 기반 smoke test 추가
  - `scripts/compare_reranking.py` 비교 대상 추가
  - live provider metric은 명시적 opt-in으로만 실행
- **검증 명령**:
  - `UV_NATIVE_TLS=true ./scripts/verify.sh`

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] AcuRank PDF 기반 reference 요약 작성
- [x] `trueskill>=0.4.5` 의존성 추가
- [x] 본 spec 사용자 검토 및 승인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/strategies/_acurank.py`: AcuRank sync/async strategy 추가
- [x] `src/ranksmith/strategies/_acurank.py`: threshold / probability / TrueSkill update helper 추가
- [x] `src/ranksmith/__init__.py`: public API 연결
- [x] `src/ranksmith/azure.py`: built-in strategy 목록 연결
- [x] `scripts/compare_reranking.py`: 비교 대상 추가
- [x] `src/ranksmith/_mteb_eval.py`: method parser 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_acurank.py`: sync/async AcuRank 테스트 추가
- [x] `tests/test_public_protocols.py`: public import 테스트 추가
- [x] `tests/test_benchmark_fixture.py`: fixture smoke test 추가
- [x] `tests/test_compare_reranking.py`: compare script 테스트 추가
- [x] `./scripts/verify.sh` 통과 확인

### Phase 4: 완료 및 정리
- [x] `docs/wiki/02_architecture.md` 업데이트
- [x] `docs/wiki/04_references_index.md` 구현 상태 업데이트
- [x] `README.md` / `README.ko.md` 업데이트
- [x] `examples/acurank.py`: score prior 기반 실행 예제 추가
- [x] 본 문서 상태를 `[x] Completed`로 변경
