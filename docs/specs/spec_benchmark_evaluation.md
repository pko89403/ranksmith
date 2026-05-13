# Spec: Benchmark Evaluation

## 1. 개요 (Overview)
- **작업 목적**: reranking algorithm을 같은 데이터와 metric으로 비교하고, 향후 algorithm 개발 완료 판단에 재사용할 benchmark 실행 체계를 만든다.
- **Reference**: SciFact/BEIR 형식의 `corpus.jsonl`, `queries.jsonl`, `qrels/*.tsv`.
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**
  - 기존 smoke fixture JSONL.
  - BEIR/SciFact cache directory.
  - 실제 reranking benchmark에는 first-stage candidate file이 필요하다.
  - 비용이 발생하는 provider 실행은 `--allow-live`가 있을 때만 허용한다.
- **출력 (Outputs)**
  - per-query 결과와 aggregate metric을 포함하는 JSON.
  - `README.md`에 사람이 읽을 수 있는 benchmark 요약과 재현 명령.
- **제약 사항 (Constraints)**
  - 새 benchmark dataset은 repository에 commit하지 않는다.
  - qrels만으로 만든 후보군을 진짜 BEIR benchmark처럼 표현하지 않는다.
  - candidate file이 없으면 기본 benchmark mode는 fast fail한다.
  - diagnostic 후보군 생성은 명시 옵션일 때만 허용하고 결과 JSON에 표시한다.
  - public API를 확장하지 않는다.

## 3. 상세 설계 (Architecture & Design)
- **동작 메커니즘**
  1. CLI가 source를 선택한다: fixture JSONL 또는 BEIR/SciFact cache.
  2. loader가 query, candidate documents, qrels를 `BenchmarkCase`로 정규화한다.
  3. runner가 선택된 algorithm별로 reranking을 수행한다.
  4. metric layer가 NDCG@k, MRR@k, Recall@k를 per-query로 계산한다.
  5. aggregate layer가 macro average를 계산한다.
  6. JSON output에는 schema version, source metadata, algorithm, candidate strategy, per-query, aggregate를 포함한다.
- **의사 알고리즘 (Pseudo-algorithm)**
  - For each algorithm:
    - For each benchmark case:
      - rerank candidates.
      - compute metrics from ranked ids and qrels.
    - aggregate metric values by arithmetic mean over evaluated cases.
- **의사 코드 (Pseudo-code)**
  ```python
  cases = load_fixture(path) or load_beir_cache(cache_dir, candidates)
  results = []
  for algorithm in algorithms:
      for case in cases:
          ranked_ids = rerank(case.query, case.documents, algorithm)
          results.append(evaluate(ranked_ids, case.qrels, top_k))
  report = build_report(results)
  ```
- **통합 지점 (Integration Points)**
  - `src/ranksmith/_benchmark.py`: private benchmark schema, loader, metric aggregation.
  - `scripts/compare_reranking.py`: CLI 확장과 live Azure runner 연결.
  - `tests/test_benchmark_runner.py`: offline deterministic 검증.
  - `README.md`: 결과 요약과 재현 명령.

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **공통 컴포넌트 식별 (Shared Components)**
  - `src/ranksmith/_metrics.py`의 NDCG/MRR/Recall을 재사용한다.
  - benchmark case schema는 fixture와 BEIR cache loader가 공유한다.
  - aggregate report builder는 live/offline 실행에서 공유한다.
- **추상화 방안 (Abstraction Plan)**
  - script에는 argparse와 provider wiring만 둔다.
  - schema validation, BEIR parsing, metric aggregation은 private helper로 분리한다.

## 5. 에러 핸들링 (Error Handling)
- missing cache file: `ValueError` 또는 CLI `SystemExit`로 정확한 파일 경로를 알린다.
- malformed JSONL/TSV: line number와 함께 실패한다.
- candidate file에 없는 query/document id: fast fail한다.
- candidate file 없이 BEIR benchmark mode 실행: fast fail한다.
- diagnostic candidate strategy 사용: JSON report에 `benchmark_type="diagnostic_not_retrieval"`를 기록한다.
- invalid `top_k`, `candidate_count`, `max_cases`: 0 이하 값은 거부한다.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**
  - synthetic BEIR cache + candidate TSV를 loader가 `BenchmarkCase`로 변환한다.
  - deterministic provider로 per-query/aggregate metric을 계산한다.
  - fixture JSONL 경로도 기존과 동일하게 평가된다.
- **엣지/실패 케이스 (Edge & Failure Cases)**
  - candidate file 누락 시 BEIR mode는 실패한다.
  - malformed corpus/queries/qrels/candidates는 line number를 포함해 실패한다.
  - qrels가 없는 query 또는 candidate가 없는 query는 실패한다.
- **공통 Reranking Smoke/Benchmark**
  - `tests/fixtures/reranking_smoke_fixture.jsonl`은 빠른 smoke regression으로 유지한다.
  - live Azure benchmark는 자동 테스트에 포함하지 않는다.
  - README 수치는 재현 명령과 candidate strategy caveat를 함께 기록한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] benchmark scope와 저장 정책 확정
- [x] 스펙 문서(본 문서) 상의 의사 코드 설계 검토 및 확정

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/_benchmark.py`: schema, BEIR loader, aggregate metric 구현
- [x] `scripts/compare_reranking.py`: benchmark runner CLI 확장
- [x] `README.md`: benchmark 결과 요약과 재현 명령 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_benchmark_runner.py`: loader와 aggregate 테스트 추가
- [x] `tests/test_benchmark_fixture.py`: 기존 smoke fixture 검증 유지
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] README 중심으로 문서화하고 `docs/wiki/` 추가는 생략
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
