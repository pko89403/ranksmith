# Spec: TourRank-r

## 1. 개요 (Overview)
- **작업 목적**: TourRank-r을 ranksmith의 공식 built-in Strategy로 추가한다.
- **Reference**: `docs/wiki/references/tourrank.md`
- **상태**: `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력**: `Sequence[str | Document]`, selection provider, 선택적 `top_k`.
- **출력**: `list[RerankResult]`. `rank`는 1-based, `original_index`는 0-based.
- **제약 사항**:
  - 기존 `ListwiseStrategy` / `PairwiseStrategy` 동작을 바꾸지 않는다.
  - 기본 `rounds=2`, 기본 stage는 논문 top-100 설정이다.
  - sync `TourRankStrategy`는 기본 `group_parallelism=1`로 직렬 실행한다.
  - sync `TourRankStrategy.group_parallelism`은 양의 정수만 허용한다.
  - async `AsyncTourRankStrategy`는 기본 `group_parallelism=None`으로 stage 내 group을 병렬 실행한다.
  - 기본 stage와 문서 수가 맞지 않으면 자동 보정하지 않고 fast fail한다.
  - provider 응답은 strict JSON `{"selected": [...]}`만 허용한다.

## 3. 상세 설계 (Architecture & Design)
- `TourRankStrategy`와 `AsyncTourRankStrategy`를 새 Strategy로 추가한다.
- `SelectionLLMProvider.select(query, documents, top_m)` 계약을 공개한다.
- 각 round에서 stage별 group selection을 수행하고, 선택된 문서 점수를 `+1`한다.
- 최종 순위는 누적 점수 내림차순, 동점은 입력 원래 순서로 정렬한다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- `parse_selection_response()`를 public parser helper로 추가한다.
- Azure provider는 기존 `rank()` / `compare()`와 별개로 `select()`를 제공한다.
- Sync strategy는 `group_parallelism > 1`이면 같은 stage의 group calls를 thread pool로 병렬 실행한다.
- Async strategy는 같은 stage의 group calls를 `asyncio.gather()`로 병렬 실행하고, `group_parallelism`이 지정되면 semaphore로 동시성을 제한한다.
- Sync 병렬 실행에서 한 group이 실패해도 이미 제출된 group 호출은 진행될 수 있다.

## 5. 에러 핸들링 (Error Handling)
- stage config와 문서 수 불일치: `RerankInputError`
- invalid selection JSON: `RerankParseError`
- 문서 길이 초과: `DocumentTooLongError`
- provider 실패: `RerankProviderError`

## 6. 테스트 계획 (Test Plan)
- public import 테스트
- `parse_selection_response()` 성공/실패 테스트
- sync/async TourRank 동작 테스트
- sync 기본 직렬, sync 병렬 opt-in, async 병렬 제한 테스트
- stage 불일치, provider protocol 불일치, invalid selection fast fail 테스트
- fixture 기반 smoke test
- example 실행 테스트
- `UV_NATIVE_TLS=true ./scripts/verify.sh`

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] TourRank PDF 기반 reference 요약 작성

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/protocols.py`: selection provider protocol 추가
- [x] `src/ranksmith/parsing.py`: selection parser 추가
- [x] `src/ranksmith/strategies.py`: TourRank sync/async strategy 추가
- [x] `src/ranksmith/_providers.py`: Azure `select()` 추가
- [x] `src/ranksmith/azure.py`, `src/ranksmith/__init__.py`: public API 연결

### Phase 3: 검증 (Verification)
- [x] parser/public import 테스트 추가
- [x] sync/async TourRank 테스트 추가
- [x] group_parallelism 테스트 추가
- [x] fixture smoke test 추가
- [x] example 실행 테스트 추가
- [x] `scripts/compare_reranking.py` 비교 대상 추가
- [x] `scripts/compare_reranking.py`: 100개 외 후보 수에는 명시 stage config 생성

### Phase 4: 완료 및 정리
- [x] README / README.ko 업데이트
- [x] wiki architecture / reference index 업데이트
