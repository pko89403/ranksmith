# Spec: Confidence-Aware Reranking (초안)

## 1. 개요 (Overview)
- **작업 목적**: 구현 완료된 confidence 신호(`ranksmith.confidence`)를 실제 reranking 결정에 연결한다. runtime readiness 스펙이 "CBDR-ready confidence signal"까지 완성해 두었으므로, 이 스펙은 그 신호를 소비하는 첫 모듈을 정의한다.
- **Reference**:
  - `docs/wiki/references/structural_confidence.md` (Trust in One Round — 구현된 confidence 신호의 근간)
  - `docs/wiki/references/cbdr_parametric_confidence_rag.md` (CBDR — 개념 로드맵. white-box 전제라 방식 직접 이식 불가)
  - `docs/wiki/references/llm_confidence_reranker.md` (LCR — training-free 정렬 규칙의 후보)
  - `docs/specs/spec_confidence_runtime_readiness.md` (소비할 signal contract)
- **상태**: `[x] Draft` | `[ ] In Progress` | `[ ] Completed`

이 스펙은 초안이다. §7의 열린 결정(Q004)이 해소되기 전에는 구현하지 않는다.

### 범위에 대한 사실 정리
- CBDR 논문의 CBDR 메커니즘 자체는 "retrieval을 할지 결정하는 트리거"다. ranksmith에는 retriever가 없으므로 **retrieval 트리거는 이 스펙의 범위 밖**이며 caller 애플리케이션 책임이다.
- CBDR 논문의 confidence 추출(LLM hidden state 직접 접근)과 reranker fine-tuning은 ranksmith의 폐쇄형 모델 / training-free 전제와 충돌한다. 이 스펙이 논문에서 가져오는 것은 **"confidence 신호로 문서 우선순위를 조정한다"는 설계 패턴**뿐이다.

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 포함 범위
- 기존 reranking 결과(`list[RerankResult]`) 또는 첫 단계 순서를 입력으로 받아, 후보별 confidence로 순위를 보정하는 모듈 1개.
- confidence 소스는 **경로 A로 확정**(Q004-1 resolved). 호출 비용 때문에 경로 B(LCR/MSCP)는 채택하지 않는다.
  - **경로 A (structural, 확정)**: `StructuralConfidenceEstimator.score_batch(...)`의 `judgment_confidence` 신호. runtime readiness 계약대로 candidate identity 결합은 이 모듈(adapter)의 책임. 추론 시 문서당 judgment 1회 + 로컬 CPU 특징 추출.
  - ~~경로 B (LCR/MSCP)~~: 문서당 K회+ 호출 비용으로 미채택.
- 정렬 규칙은 LCR 방식을 기본 후보로 한다: `T_upper`/`T_lower`로 High(+1)/Medium(0)/Low(−1) binning 후 [bin 내림차순, 기존 순위] StableSort. query-level gate(`T_query`) 채택 여부는 열린 결정.
- 기존 순위로의 환원성 보장: gate 조건에서 원래 순위를 그대로 반환한다 (LCR의 `T_query=0` 환원성과 동일한 안전장치).

### 제외 범위
- retrieval 트리거 (CBDR 본래 의미) — caller 책임.
- reranker/scorer fine-tuning — 경로 A의 artifact 학습은 기존 `confidence_training` 스펙 소관.
- 벤치마크 수치 주장 — 측정 전에는 README에 성능 claim을 쓰지 않는다.

### 제약
- fast fail 원칙 유지. silent fallback 금지 — confidence 계산 실패 시 원래 순위로 조용히 돌아가지 않고 에러를 낸다.
- 경로 B 채택 시 `ModelClient`의 temperature 0 / JSON-only 계약과 충돌한다. sampling 전용 요청 경로(예: `ModelRequest`에 sampling 파라미터 복원 또는 별도 client 메서드)가 선행 결정 사항이다.
- 경로 A 채택 시 학습된 scorer artifact가 선행 조건이다. 현재 리포에 커밋된 artifact는 없다 — generation → training 파이프라인을 한 번 실행해 만들어야 한다.

## 3. 상세 설계 (Architecture & Design) — 방향만, 확정 아님

### 동작 메커니즘 (공통 골격)
1. 입력: `query`, 순서 있는 후보 목록(기존 순위 = PrevRank), confidence 파라미터.
2. 후보별 confidence 계산 (경로 A: score_batch 순서 보존 결과를 zip / 경로 B: MSCP).
3. binning: `c >= T_upper → +1`, `c <= T_lower → −1`, 그 외 0.
4. StableSort by (bin desc, PrevRank asc). gate 조건이면 PrevRank 그대로.
5. 출력: `list[RerankResult]`, metadata에 `confidence_score`와 bin을 포함.

### 통합 지점 (후보)
- 형태 1: 새 Strategy (`docs/wiki/08_custom_strategy_extension.md` 패턴) — 다른 Strategy와 조합하려면 caller가 2단계 호출.
- 형태 2: `list[RerankResult]`를 받는 post-processor 함수 — Strategy 계약을 건드리지 않음. LCR이 "기존 reranker 뒤에 붙는" 구조라는 점과 정합.
- 어느 쪽인지 Q004-2에서 결정.

## 5. 에러 핸들링 (Error Handling)
- confidence 소스 실패(artifact 불일치, sampling 파싱 실패): 해당 confidence 에러로 fast fail.
- threshold 검증: `0 <= T_lower <= T_upper <= 1` 위반 시 `ValueError`.
- 빈 후보 목록: 빈 결과 반환 (기존 Strategy들과 동일).

## 6. 테스트 계획 (Test Plan)
- binning 경계값 (c == T_upper, c == T_lower).
- 환원성: gate 조건에서 입력 순위와 출력 순위가 동일.
- StableSort 안정성: 같은 bin 내에서 PrevRank 유지.
- 경로 A: score_batch 순서 보존 결과와 candidate zip 정합성.
- 경로 B: MSCP 계산(클러스터 비율), entailment 응답 파싱 fast fail.
- fixture 기반 smoke test (`tests/fixtures/reranking_smoke_fixture.jsonl` + `benchmarks/metrics.py`).

## 7. 열린 결정 (`docs/wiki/05_open_questions.md` Q004)
1. ~~confidence 소스~~ — **경로 A(structural + LightGBM)로 확정**. 호출 비용으로 경로 B 미채택.
2. **형태**: 새 Strategy vs post-processor. (미결)
3. **score → 순위 반영 규칙**: bin 방식 유지 여부와 threshold 기본값. (미결)
4. **artifact 학습 데이터**: score_batch가 소비할 scorer artifact를 만들 실제 라벨 데이터 ≥30개 확보. (미결)

## 스모크 검증 기록 (2026-07-05)
경로 A 파이프라인을 LM Studio(qwen3.5-9b) + libomp 설치 환경에서 부분 관통했다.
- generation: `generate_judgment_confidence_dataset`가 fixture 15개를 엔드투엔드 생성 (LM Studio provider).
- 추론 절반: `FrozenAutoEncoder`(bert-base-uncased) 로드 + 70차원 `structural-v1` 특징 추출 확인.
- 학습: `split.py`의 `MIN_TOTAL_SAMPLES = 30` 가드에 막힘 — 리포 fixture는 15개뿐. **artifact 생성에는 ≥30개 실제 라벨 데이터가 필요**하다(위 결정 4).

## 작업 태스크 추적 (Task Checklist)
### Phase 0: 결정
- [x] Q004-1 confidence 소스 = 경로 A
- [ ] Q004 나머지 (형태, 순위 반영 규칙, 학습 데이터)
### Phase 1 이후
- [ ] ≥30개 실제 라벨 데이터로 scorer artifact 생성
- [ ] 결정 반영해 본 스펙 확정(§3 의사 코드 구체화) 후 구현 착수
