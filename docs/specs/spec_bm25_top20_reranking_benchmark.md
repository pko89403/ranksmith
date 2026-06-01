# Spec: BM25 Top-20 Reranking Benchmark

## 1. 개요 (Overview)
- **작업 목적**: 기존 benchmark dataset의 corpus/query/qrels를 유지하고, Pyserini BM25 top-20 후보를 first-stage로 고정해 ranksmith reranking method를 비교한다.
- **Reference**:
  - Pyserini BM25 / Lucene sparse retrieval
  - `docs/specs/spec_benchmark_evaluation.md`
  - `docs/specs/spec_acurank.md`
- **상태**: `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력**:
  - Benchmark cache: `corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`
  - Pyserini BM25 TREC run 또는 simple candidate TSV
  - query당 BM25 top-20 후보
- **출력**:
  - JSON benchmark report
  - method별 aggregate/per-query metric
- **제약 사항**:
  - ranksmith는 BM25 retrieval을 수행하지 않는다.
  - Pyserini는 top-20 후보 artifact를 만드는 upstream 도구다.
  - benchmark 표에서는 BM25 retrieval metric과 reranking metric을 섞지 않는다.
  - AcuRank는 BM25 score를 `Document.metadata["score"]` prior로 사용한다.
  - live LLM 호출은 `--allow-live` 명시 시에만 실행한다.

## 3. 상세 설계 (Architecture & Design)
- **평가 기준**:
  - Candidate scope: `N=20`
  - Final metric cutoff: `@5`
- **기본 method set**:
  - `original_bm25`: BM25 top-20 그대로
  - `single_call_listwise@20`: `N=20`
  - `rankgpt_sw_w5`: `N=20`, `window_size=5`, `stride=2`
  - `acurank_k5_b1`: `N=20`, `target_rank=5`, `window_size=20`, adaptive budget 1
  - `tourrank_r2`: `N=20`, `rounds=2`
  - `setwise_hs_s10`: `N=20`, `set_size=10`, `top_k=5`
  - `prp_sliding_p1`: `N=20`, `passes=1`
- **optional method**:
  - `acurank_k5_b4`: `N=20`, `target_rank=5`, `window_size=20`, adaptive budget 4
  - `acurank_b1`: `N=20`, `target_rank=10`, `window_size=20`, adaptive budget 1
  - `acurank_b4`: `N=20`, `target_rank=10`, `window_size=20`, adaptive budget 4
  - `tourrank_r10`: `N=20`, `rounds=10`
  - `prp_sliding_p3`: `N=20`, `passes=3`
- **예상 calls/query**:
  - `original_bm25`: 0
  - `single_call_listwise@20`: 1
  - `rankgpt_sw_w5`: 9
  - `acurank_k5_b1`: 2
  - `tourrank_r2`: 8
  - `setwise_hs_s10`: 12
  - `prp_sliding_p1`: 38
  - `acurank_k5_b4`: 5
  - `acurank_b1`: 2
  - `acurank_b4`: 5
  - `tourrank_r10`: 40
  - `prp_sliding_p3`: 114
- **통합 지점**:
  - `src/ranksmith/_benchmark.py`
    - Pyserini TREC candidate run parsing
    - candidate score 보존
    - `candidate_count` 기준 top-N truncation
    - generic benchmark cache dataset metadata 지원
  - `scripts/compare_reranking.py`
    - BM25 top-20 benchmark method alias 추가
    - `benchmark-cache` mode 추가
    - `original_bm25` no-op baseline 추가
    - AcuRank BM25 score metadata 주입
  - `tests/test_benchmark_runner.py`, `tests/test_compare_reranking.py`
    - parser, alias, call estimate, strategy config 검증

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- 기존 `BenchmarkCase`, `BenchmarkDocument`, metric aggregation을 재사용한다.
- candidate file loader는 simple TSV와 Pyserini TREC run을 모두 읽는다.
- method alias는 benchmark CLI 내부 개념으로 유지하고 public Strategy API에는 추가하지 않는다.

## 5. 에러 핸들링 (Error Handling)
- candidate row가 simple TSV 또는 TREC run으로 해석되지 않으면 `ValueError`.
- duplicate doc id per query는 `ValueError`.
- score가 필요한 TREC/simple score 칼럼이 숫자가 아니면 `ValueError`.
- benchmark mode에서 candidate file이 없으면 기존처럼 fast fail.

## 6. 테스트 계획 (Test Plan)
- Pyserini TREC run row를 `query_id`, `doc_id`, `rank`, `score`로 읽는다.
- `candidate_count=20`이면 query별 상위 20개만 사용한다.
- `benchmark-cache` mode는 기존 benchmark dataset label을 report에 보존한다.
- `original_bm25`는 입력 순서를 그대로 반환한다.
- `single_call_listwise@20`, `rankgpt_sw_w5`, `acurank_k5_b1`, `tourrank_r2`, `setwise_hs_s10`, `prp_sliding_p1` alias가 올바른 Strategy 설정으로 변환된다.
- optional `acurank_k5_b4`, `acurank_b1`, `acurank_b4`, `tourrank_r10`, `prp_sliding_p3` call estimate를 검증한다.
- AcuRank 문서에는 BM25 score metadata가 들어간다.
- `./scripts/verify.sh`를 통과한다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 benchmark 코드 확인
- [x] BM25 top-20, evaluate@5 benchmark method set 확정

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/_benchmark.py`: Pyserini TREC run 및 score parsing
- [x] `scripts/compare_reranking.py`: benchmark method alias 및 AcuRank score prior 연결
- [x] `docs/benchmarks/bm25_top20_reranking.md`: 실행 문서 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_benchmark_runner.py`: candidate score/TREC parser 테스트 추가
- [x] `tests/test_compare_reranking.py`: method alias 및 strategy config 테스트 추가
- [x] `./scripts/verify.sh` 통과

### Phase 4: 완료 및 정리
- [x] 본 문서 상태를 `[x] Completed`로 변경
